import torch
from torch import nn
import torch.nn.functional as F


from .raft_utils.update import BasicUpdateBlock2, SepConvGRU
from .raft_utils.extractor import BasicEncoder
from .raft_utils.corr import CorrBlock
from .raft_utils.utils import coords_grid
try:
    from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
    FLASH_AVAIABLE = True
    print("flash attention is available!")
except:
    FLASH_AVAIABLE = False
    print("flash attention is not available!")


class SPOT(nn.Module):
    def __init__(self, args, mixed_precision=False, cfg=None):
        super().__init__()
        self.args = cfg
        self.fnet = BasicEncoder(output_dim=256, norm_fn=args.norm_fnet, dropout=0, patch_size=args.patch_size)
        self.cnet = BasicEncoder(output_dim=256, norm_fn=args.norm_cnet, dropout=0, patch_size=args.patch_size)

        self.update_block = BasicUpdateBlock2(args=self.args, hidden_dim=128, patch_size=args.patch_size, refine_alpha=args.refine_alpha)
        self.refine_alpha = args.refine_alpha
        self.patch_size = args.patch_size
        self.num_iter = args.num_iter
        self.mixed_precision = mixed_precision

        # additional module
        self.key_proj = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        nn.init.orthogonal_(self.key_proj.weight.data)
        nn.init.zeros_(self.key_proj.bias.data)
        
        self.fuser = nn.Conv2d(256 + 256, 256, kernel_size=3, padding=1)
        nn.init.zeros_(self.fuser.weight.data)
        nn.init.zeros_(self.fuser.bias.data)

        self.s_m_update = SepConvGRU(hidden_dim=128, input_dim=128)

    def initialize_flow(self, fmap, coarse_flow):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, _, h, w = fmap.shape
        src_pts = coords_grid(N, h, w, device=fmap.device)

        if coarse_flow is not None:
            print("coarse_flow shape: ", coarse_flow.shape)
            tgt_pts = src_pts + coarse_flow
        else:
            tgt_pts = src_pts

        return src_pts, tgt_pts

    def initialize_alpha(self, fmap, coarse_alpha):
        N, _, h, w = fmap.shape
        if coarse_alpha is None:
            alpha = torch.ones(N, 1, h, w, device=fmap.device)
        else:
            alpha = coarse_alpha[:, None]
        return alpha.logit(eps=1e-5)

    def postprocess_alpha(self, alpha):
        alpha = alpha[:, 0]
        return alpha

    def postprocess_flow(self, flow):
        flow = flow.permute(0, 2, 3, 1)
        return flow

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/P, W/P, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, self.patch_size, self.patch_size, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(self.patch_size * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, self.patch_size * H, self.patch_size * W)

    def upsample_alpha(self, alpha, mask):
        """ Upsample alpha field [H/P, W/P, 1] -> [H, W, 1] using convex combination """
        N, _, H, W = alpha.shape
        mask = mask.view(N, 1, 9, self.patch_size, self.patch_size, H, W)
        mask = torch.softmax(mask, dim=2)

        up_alpha = F.unfold(alpha, [3, 3], padding=1)
        up_alpha = up_alpha.view(N, 1, 9, 1, 1, H, W)

        up_alpha = torch.sum(mask * up_alpha, dim=2)
        up_alpha = up_alpha.permute(0, 1, 4, 2, 5, 3)
        return up_alpha.reshape(N, 1, self.patch_size * H, self.patch_size * W)

    def encode_features(self, frame, flow_init=None, alpha_init=None):
        # Determine input shape
        if len(frame.shape) == 5:
            # shape is b*t*c*h*w
            need_reshape = True
            b, t = frame.shape[:2]
            # flatten so that we can feed them into a 2D CNN
            frame = frame.flatten(start_dim=0, end_dim=1)
        elif len(frame.shape) == 4:
            # shape is b*c*h*w
            need_reshape = False
        else:
            raise NotImplementedError
        with torch.cuda.amp.autocast(enabled=self.mixed_precision, dtype=torch.bfloat16):
            fmaps = self.fnet(frame)
        fmaps = fmaps.float()
        key = self.key_proj(fmaps)
        if need_reshape:
            # B*T*C*H*W
            fmaps = fmaps.view(b, t, *fmaps.shape[-3:])
            # frame = frame.view(b, t, *frame.shape[-3:])
            key = key.view(b, t, *key.shape[-3:])
            coords0, coords1 = self.initialize_flow(fmaps[:, 0, ...], flow_init)
            alpha = self.initialize_alpha(fmaps[:, 0, ...], alpha_init) if self.refine_alpha else None
        else:
            coords0, coords1 = self.initialize_flow(fmaps, flow_init)
            alpha = self.initialize_alpha(fmaps, alpha_init) if self.refine_alpha else None

        return coords0, coords1, fmaps, alpha, key

    def encode_context(self, frame):
        # shape is b*c*h*w
        with torch.cuda.amp.autocast(enabled=self.mixed_precision, dtype=torch.bfloat16):
            cnet = self.cnet(frame)
        cnet = cnet.float()
        net, inp = torch.split(cnet, [128, 128], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        return net, inp

    def read_memory(self, query_key, query_value, memory_key=None, memory_value=None):
        # query_key: b c h w
        # memory_key/memory_value: b c t h w
        if memory_key is not None and memory_value is not None:
            with torch.cuda.amp.autocast(enabled=self.mixed_precision, dtype=torch.bfloat16):
                query_key = query_key.flatten(start_dim=2)
                memory_key = memory_key.flatten(start_dim=2)
                memory_value = memory_value.flatten(start_dim=2)
                if FLASH_AVAIABLE:
                    query_key = query_key.permute(0, 2, 1).unsqueeze(2).bfloat16()
                    memory_key = memory_key.permute(0, 2, 1).unsqueeze(2).bfloat16()
                    memory_value = memory_value.permute(0, 2, 1).unsqueeze(2).bfloat16()
                    memory1 = flash_attn_func(query_key, memory_key, memory_value[:, :, :, :memory_key.shape[-1]])
                    memory2 = flash_attn_func(query_key, memory_key, memory_value[:, :, :, memory_key.shape[-1]:])
                    memory = torch.cat([memory1, memory2], dim=-1).squeeze(2).permute(0, 2, 1)
                else:
                    scale = query_key.shape[1] ** -0.5

                    similarity = torch.einsum('b c l, b c t -> b t l', query_key, memory_key) * scale
                    maxes = torch.max(similarity, dim=1, keepdim=True)[0]
                    x_exp = torch.exp(similarity - maxes)
                    x_exp_sum = torch.sum(x_exp, dim=1, keepdim=True)
                    affinity = x_exp / x_exp_sum
                    memory = memory_value @ affinity

                memory = memory.reshape(*query_value.shape[:])
                if not self.args.ablate_fuser:
                    fused_fea = query_value + self.fuser(torch.cat([query_value, memory], dim=1))
                else:
                    fused_fea = memory

            return fused_fea.float()
        else:
            return query_value

    def predict_flow(self, net, inp, src_pts, tgt_pts, src_fmap, tgt_fmap, s_memory, alpha, is_train=False, num_iter=None,
                     alpha_thresh=0.8, attn=None):
        corr_fn = CorrBlock(src_fmap, tgt_fmap)

        flows_up = []
        alphas_up = []

        num_iter = num_iter if num_iter is not None else self.num_iter
        for itr in range(num_iter):
            tgt_pts = tgt_pts.detach()
            if self.refine_alpha:
                alpha = alpha.detach()

            corr = corr_fn(tgt_pts)

            flow = tgt_pts - src_pts
            with torch.cuda.amp.autocast(enabled=self.mixed_precision, dtype=torch.bfloat16):
                net, up_mask, delta_flow, up_mask_alpha, delta_alpha, motion_fea = self.update_block(net, inp, corr, flow, alpha, s_memory, attention=attn)

            # F(t+1) = F(t) + \Delta(t)
            tgt_pts = tgt_pts + delta_flow
            if self.refine_alpha:
                alpha = alpha + delta_alpha

            # upsample predictions
            flow_up = self.upsample_flow(tgt_pts - src_pts, up_mask)
            if self.refine_alpha:
                alpha_up = self.upsample_alpha(alpha, up_mask_alpha)

            if is_train or (itr == self.num_iter - 1):
                flows_up.append(self.postprocess_flow(flow_up))
                if self.refine_alpha:
                    alphas_up.append(self.postprocess_alpha(alpha_up))
        s_memory = self.s_m_update(s_memory, motion_fea)

        flows_up = torch.stack(flows_up, dim=1)
        alphas_up = torch.stack(alphas_up, dim=1) if self.refine_alpha else None
        if not is_train:
            flows_up = flows_up[:, 0]
            alphas_up = alphas_up[:, 0] if self.refine_alpha else None
        return {"flow": flows_up, "alpha": alphas_up, "alpha_low": alpha, "flow_low": tgt_pts - src_pts,
                "s_memory": s_memory, "net": net}

    def forward(self, mode, **kwargs):
        if mode == "encode_features":
            return self.encode_features(**kwargs)
        elif mode == "encode_context":
            return self.encode_context(**kwargs)
        elif mode == "value_reinforce":
            return self.value_reinforce(**kwargs)
        elif mode == "read_memory":
            return self.read_memory(**kwargs)
        elif mode == "predict_flow":
            return self.predict_flow(**kwargs)
        else:
            raise NotImplementedError(f"{mode} not implemented yet.")
