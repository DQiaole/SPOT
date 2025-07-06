from inference.memory_manager import MemoryManager
from spot.models.shelf.raft_utils.corr import CorrBlock
import torch
from spot.softsplat import FunctionSoftsplat as forward_warping


class InferenceCore:
    def __init__(self, network, config):
        self.config = config
        self.model = network
        self.mem_every = config.mem_every

        self.clear_memory()

    def clear_memory(self):
        self.curr_ti = -1
        self.last_mem_ti = - self.config.mem_every
        self.memory = MemoryManager(config=self.config)

    def set_first_frame(self, frame, flow_init=None, alpha_init=None):
        net, inp = self.model.encode_context(frame)
        attention = None
        self.net = net
        self.inp = inp
        self.attention = attention
        coords0, coords1, fmaps, alpha, _ = self.model.encode_features(frame)
        self.coords0 = coords0
        self.coords1 = coords1
        self.alpha = alpha
        self.src_fmap = fmaps
        self.sensor_m = torch.zeros((fmaps.shape[0], 128, *fmaps.shape[-2:])).to(fmaps)

    def step(self, frame, end=False):
        # image: 1*3*H*W
        self.curr_ti += 1

        is_mem_frame = (self.curr_ti - self.last_mem_ti >= self.mem_every) and (not end)

        # B, C, H, W
        init_tgt_fmap = self.model.fnet(frame).float()
        key = self.model.key_proj(init_tgt_fmap)
        
        # mem readout
        memory_readout = self.memory.match_memory(key)
        if memory_readout is not None:
            tgt_fmap = init_tgt_fmap + self.model.fuser(torch.cat([init_tgt_fmap, memory_readout], dim=1))
        else:
            tgt_fmap = init_tgt_fmap

        num_iter = self.config.infer_num_iter

        # predict flow
        pred = self.model.predict_flow(self.net, self.inp, self.coords0, self.coords1, self.src_fmap, tgt_fmap,
                                       self.sensor_m, self.alpha, is_train=False, num_iter=num_iter, attn=self.attention)

        # net of GRU and flow/alpha warm up
        self.net = pred["net"]
        self.coords1 = (pred["flow_low"] + self.coords0 - self.coords1) * self.config.extrapolate + self.coords1
        self.alpha = (pred["alpha_low"] - self.alpha) * 2 + self.alpha
        # s_m update
        self.sensor_m = pred["s_memory"]

        # save as memory if needed
        if is_mem_frame:
            init_value = self.src_fmap
            # forward splatting
            current_value = forward_warping(init_value, pred["flow_low"], tenMetric=pred["alpha_low"].sigmoid(),
                                            strType=self.config.splatting_type)  # b c h w
            self.memory.add_memory(key, current_value)
            self.last_mem_ti = self.curr_ti

        return pred
