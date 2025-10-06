import os
import torch
import torch.nn as nn
import argparse
from datetime import datetime
# torch.backends.cudnn.benchmark = True
from spot.softsplat import FunctionSoftsplat as forward_warping
from torch.optim.lr_scheduler import OneCycleLR
from spot.data.movi_f_dataset import create_point_tracking_dataset
from spot.utils.io import create_folder, read_config
from spot.utils.log import Logger
from spot.models.shelf import SPOT
from spot.utils.options.base_options import str2bool
from torch.cuda.amp import GradScaler
import numpy as np
from einops import repeat, rearrange
import torch.distributed as dist
import torch.multiprocessing as mp


def cycle(dl):
    while True:
        for data in dl:
            yield data


def checkpoint(model, optimizer, path, name):
    create_folder(path)
    model_path = os.path.join(path, f"{name}.pth")
    optimizer_path = os.path.join(path, f"{name}_optimizer.pth")
    torch.save(model.module.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path)


def sample(flow_preds, alpha_preds, gt):
    # b i t c h w
    # b i t h w
    B, I, T, _, H, W = flow_preds.shape
    dense = torch.cat([flow_preds, alpha_preds.unsqueeze(3)], dim=3)
    dense = rearrange(dense, "b i t c h w -> b (i t c) h w")
    src_pos = gt["out_src_points"][..., :2]
    grid = src_pos[:, None] * 2 - 1
    sparse = torch.nn.functional.grid_sample(dense, grid, mode="nearest", align_corners=True, padding_mode="border")
    sparse = rearrange(sparse, "b (i t c) h w -> b i (h w) t c", i=I, t=T)
    delta_pos, tgt_alpha = sparse[..., :2], sparse[..., 2:]
    delta_pos[..., 0] = delta_pos[..., 0] / (W - 1)
    delta_pos[..., 1] = delta_pos[..., 1] / (H - 1)
    tgt_pos = src_pos[:, None, :, None] + delta_pos
    out_tgt_points = torch.cat([tgt_pos, tgt_alpha], dim=-1)
    pred = {
        "out_tgt_points": out_tgt_points,
    }
    return pred


def calculate_sparse_corr_loss(args, flow_preds, alpha_preds, gt, loss):
    pred = sample(flow_preds, alpha_preds, gt)
    motion_loss = torch.tensor(0., requires_grad=True).cuda()
    visibility_loss = torch.tensor(0., requires_grad=True).cuda()
    gamma = 0.8
    num_iter = pred["out_tgt_points"].size(1)
    for i in range(num_iter):
        weight = gamma ** (num_iter - i - 1)
        pred_pos, pred_vis = pred["out_tgt_points"][:, i][..., :2], pred["out_tgt_points"][:, i][..., 2]
        gt_pos, gt_vis = gt["out_tgt_points"][..., :2], gt["out_tgt_points"][..., 2]
        motion_loss += weight * (gt_pos - pred_pos).abs().mean()
        visibility_loss += weight * torch.nn.functional.binary_cross_entropy_with_logits(pred_vis, gt_vis)
    if args.movif_cvo:
        loss += motion_loss * args.lambda_motion_loss * 1000
    else:
        loss += motion_loss * args.lambda_motion_loss
    loss += visibility_loss * args.lambda_visibility_loss
    return loss, motion_loss, visibility_loss


def calculate_dense_flow_loss(args, flow_preds, alpha_preds, gt, loss):
    B, I, T, _, H, W = flow_preds.shape
    gamma = 0.8
    n_predictions = I
    motion_loss = torch.tensor(0., requires_grad=True).cuda()
    visibility_loss = torch.tensor(0., requires_grad=True).cuda()
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        i_loss = (flow_preds[:, i] - gt["flow"]).abs()
        motion_loss += i_weight * i_loss.mean()
        visibility_loss += i_weight * torch.nn.functional.binary_cross_entropy_with_logits(alpha_preds[:, i], gt["alpha"])

    loss += motion_loss * args.lambda_motion_loss
    loss += visibility_loss * args.lambda_visibility_loss

    epe = torch.sum((flow_preds[:, -1] - gt["flow"]) ** 2, dim=2).sqrt()
    epe = epe.view(-1).mean()

    return loss, motion_loss, visibility_loss, epe


def step(gt, model, optimizer, scheduler, logger, global_iter, args, scaler=None):
    loss = torch.tensor(0., requires_grad=True).cuda()

    for key, value in gt.items():
        if torch.is_tensor(value):
            # b t c h w
            gt[key] = value.cuda()
    # normalize
    gt["imgs"] = gt["imgs"] * 2 - 1
    b = gt["imgs"].shape[0]
    if args.enable_gma:
        net, inp, attention = model(mode="encode_context", frame=gt["imgs"][:, 0, ...])
    else:
        net, inp = model(mode="encode_context", frame=gt["imgs"][:, 0, ...])
        attention = None
    # b t c h w
    coords0, coords1, fmaps, alpha, key = model(mode="encode_features", frame=gt["imgs"])
    key = key.transpose(1, 2).contiguous()  # b c t h w
    fmaps = fmaps.transpose(1, 2).contiguous()  # b c t h w

    video_flow_predictions = []
    video_alpha_predictions = []
    sensor_m = torch.zeros((b, 128, *key.shape[-2:])).to(key)

    values = None
    for ti in range(1, gt["imgs"].shape[1]):
        # mem readout
        if ti == 1:
            ref_values = ref_keys = None
        elif ti <= args.num_ref_frames + 1:
            ref_values = values
            ref_keys = key[:, :, 1:ti]
        elif args.disable_random_sample:
            ref_values = values[:, :, ti-1-args.num_ref_frames:]
            ref_keys = key[:, :, ti-args.num_ref_frames:ti]
        else:
            indices = [torch.randperm(ti - 1)[:args.num_ref_frames] + 1 for _ in range(b)]
            ref_values = torch.stack([
                values[bi, :, indices[bi] - 1] for bi in range(b)
            ], 0)
            ref_keys = torch.stack([
                key[bi, :, indices[bi]] for bi in range(b)
            ], 0)
        tgt_fmap = model(mode="read_memory", query_key=key[:, :, ti], query_value=fmaps[:, :, ti], memory_key=ref_keys,
                         memory_value=ref_values)

        # predict flow from frame 0 to frame ti
        pred = model(mode="predict_flow", net=net, inp=inp, src_pts=coords0, tgt_pts=coords1, src_fmap=fmaps[:, :, 0],
                     tgt_fmap=tgt_fmap, s_memory=sensor_m, alpha=alpha, is_train=True, num_iter=args.num_iter,
                     attn=attention)
        video_flow_predictions.append(pred["flow"].permute(0, 1, 4, 2, 3))  # b i 2 h w
        video_alpha_predictions.append(pred["alpha"])  # b i h w
        # net of GRU and flow/alpha warm up
        if not args.ablate_net_warmup:
            net = pred["net"]
        if not args.ablate_flow_warmup:
            coords1 = (pred["flow_low"] + coords0 - coords1) * 2 + coords1
        if not args.disable_occ_warmup:
            alpha = (pred["alpha_low"] - alpha) * 2 + alpha
        else:
            alpha = torch.clamp(pred["alpha_low"], min=-11, max=11)
        # s_m update
        sensor_m = pred["s_memory"]
        if args.detach_splat:
            splat_flow = pred["flow_low"].detach()
            splat_alpha = pred["alpha_low"].sigmoid().detach()
        else:
            splat_flow = pred["flow_low"]
            splat_alpha = pred["alpha_low"].sigmoid()
        if args.enable_value_reinforce:
            # value reinforce
            init_value = model(mode="value_reinforce", fea=fmaps[:, :, 0], visible_mask=splat_alpha)
        else:
            init_value = fmaps[:, :, 0]
        # forward splatting
        current_value = forward_warping(init_value, splat_flow, tenMetric=splat_alpha,
                                        strType=args.splatting_type)  # b c h w
        current_value = current_value.unsqueeze(2)
        values = current_value if values is None else torch.cat([values, current_value], dim=2)

    # loss function
    video_flow_predictions = torch.stack(video_flow_predictions, dim=2)  # b, i, t, 2, H, W
    video_alpha_predictions = torch.stack(video_alpha_predictions, dim=2)  # b i t h w
    if gt["type"][0] == "sparse":
        loss, motion_loss, visibility_loss = calculate_sparse_corr_loss(args, video_flow_predictions,
                                                                        video_alpha_predictions, gt, loss)
        epe = 0.0
    elif gt["type"][0] == "dense":
        loss, motion_loss, visibility_loss, epe = calculate_dense_flow_loss(args, video_flow_predictions,
                                                                            video_alpha_predictions, gt, loss)
    else:
        raise NotImplementedError

    if optimizer is not None:
        loss = loss / args.accumulation_steps
        scaler.scale(loss).backward()
        if global_iter % args.accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            scaler.step(optimizer)
            optimizer.zero_grad()
            scaler.update()
        scheduler.step()

    if logger is not None:
        logger.log_scalar("loss/motion_loss", motion_loss, global_iter)
        logger.log_scalar("loss/visibility_loss", visibility_loss, global_iter)
        logger.log_scalar("loss/loss", loss, global_iter)
        logger.log_scalar("scale", scaler.get_scale(), global_iter)
        logger.log_scalar("epe", epe, global_iter)
        logger.log_scalar("lr", scheduler.get_last_lr()[0], global_iter)

        if global_iter % args.print_iter == 0:
            losses = f"Loss: {loss.item():.3E} ("
            losses += f"motion: {motion_loss.item():.3E}, "
            losses += f"visibility: {visibility_loss.item():.3E}, "
            losses += f"scale: {scaler.get_scale():.3E})"
            print(f"[Iter {global_iter}/{args.train_iter}] {losses}")

        if global_iter % args.log_iter == 0:
            logger.log_image("flow/pred_flow", pred["flow"][:, -1], "flow", 2, global_iter)
            logger.log_image("alpha/pred_alpha", (pred["alpha"][:, -1].sigmoid() > 0.8).float(), "mask", 2, global_iter)
            logger.log_image("img/tgt_frame", (gt["imgs"][:, -1] + 1) / 2, "rgb", 2, global_iter)
            logger.log_image("img/src_frame", (gt["imgs"][:, 0] + 1) / 2, "rgb", 2, global_iter)
            if gt["type"][0] == "dense":
                logger.log_image("flow/gt_flow", gt["flow"][:, -1].permute(0, 2, 3, 1), "flow", 2, global_iter)
                logger.log_image("alpha/gt_alpha", gt["alpha"][:, -1], "mask", 2, global_iter)

    return loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def torch_init_model(model, total_dict, key):
    if key in total_dict:
        state_dict = total_dict[key]
    else:
        state_dict = total_dict
    missing_keys = []
    unexpected_keys = []
    error_msgs = []
    # copy state_dict so _load_from_state_dict can modify it
    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(prefix[:-1], {})
        module._load_from_state_dict(state_dict=state_dict, prefix=prefix, local_metadata=local_metadata, strict=True,
                                     missing_keys=missing_keys, unexpected_keys=unexpected_keys, error_msgs=error_msgs)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(model, prefix='')

    print("missing keys:{}".format(missing_keys))
    print('unexpected keys:{}'.format(unexpected_keys))
    print('error msgs:{}'.format(error_msgs))


def main(gpu, args):
    rank = args.node_rank * args.gpus + gpu
    torch.cuda.set_device(rank)

    if args.DDP:
        dist.init_process_group(backend='nccl',
                                init_method='env://',
                                world_size=args.world_size,
                                rank=rank,
                                group_name='mtorch')
        model = nn.SyncBatchNorm.convert_sync_batchnorm(SPOT(read_config(args.refiner_config), mixed_precision=args.mixed_precision, cfg=args)).cuda()
        model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])

    if rank == 0:
        logger = Logger(args)
        print("Parameter Count: %d" % count_parameters(model))
    else:
        logger = None

    if args.refiner_path is not None:
        # model.load_state_dict(torch.load(args.refiner_path, map_location='cpu'))
        print("[Loading ckpt from {}]".format(args.refiner_path))
        ckpt = torch.load(args.refiner_path, map_location='cpu')
        torch_init_model(model.module, ckpt, key='model')

    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0, 0.99),
                                 weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = OneCycleLR(optimizer, args.lr, args.train_iter + 100,
                           pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    if args.optimizer_path is not None:
        optimizer.load_state_dict(torch.load(args.optimizer_path))
        for _ in range(args.restore_steps):
            scheduler.step()

    # Prepare data
    if args.movif:
        if args.DDP:
            train_sampler, train_loader = create_point_tracking_dataset(
                args,
                batch=args.batch_size,
                split="train",
                workers=args.num_workers,
                verbose=True,
                DDP=args.DDP,
                rank=rank
            )
        else:
            raise NotImplementedError

    scaler = GradScaler(enabled=args.mixed_precision)
    epoch = 0
    global_iter = args.restore_steps + 1
    should_keep_training = True
    while should_keep_training:
        epoch += 1
        if args.DDP:
            train_sampler.set_epoch(epoch)
        for i_batch, data_blob in enumerate(train_loader):
            step(data_blob, model, optimizer, scheduler, logger, global_iter, args, scaler)

            if global_iter % args.save_iter == 0:
                if rank == 0:
                    checkpoint(model, optimizer, args.checkpoint_path, str(global_iter) + '_it')
                dist.barrier()

            global_iter += 1
            if global_iter >= args.train_iter + 1:
                should_keep_training = False
                break

    logger.close()
    cleanup()
    return


def cleanup():
    dist.destroy_process_group()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--datetime", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="datasets/kubric/movi_f")
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--aspect_ratio", type=float, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_tracks", type=int, default=2048)
    parser.add_argument("--alpha_thresh", type=float, default=0.8)
    parser.add_argument("--is_train", type=str2bool, nargs='?', const=True, default=False)

    ##### DDP ##############
    parser.add_argument('--nodes', type=int, default=1, help='how many machines')
    parser.add_argument('--gpus', type=int, default=1, help='how many GPUs in one node')
    parser.add_argument('--GPU_ids', type=str, default='0')
    parser.add_argument('--node_rank', type=int, default=0, help='the id of this machine')
    parser.add_argument('--DDP', action='store_true', help='DDP')
    parser.add_argument("--num_workers", type=int, default=16)

    # Optical flow estimator
    parser.add_argument("--refiner_config", type=str, default="configs/raft_patch_4_alpha.json")
    parser.add_argument("--refiner_path", type=str, default=None)
    parser.add_argument("--num_iter", type=int, default=12)

    # Mem
    parser.add_argument("--input_frames", type=int, default=7)
    parser.add_argument("--num_ref_frames", type=int, default=3)
    # # args for inference
    parser.add_argument("--mem_every", type=int, default=1)
    parser.add_argument('--enable_long_term', action='store_true')
    parser.add_argument('--enable_long_term_count_usage', action='store_true')
    parser.add_argument("--max_mid_term_frames", type=int, default=4)
    parser.add_argument("--min_mid_term_frames", type=int, default=3)
    parser.add_argument("--num_prototypes", type=int, default=128)
    parser.add_argument("--max_long_term_elements", type=int, default=10000)

    # training
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument("--out_track_name", type=str, default="ground_truth")
    parser.add_argument("--num_out_tracks", type=int, default=2048)
    parser.add_argument("--train_iter", type=int, default=100000)
    parser.add_argument("--log_iter", type=int, default=500)
    parser.add_argument("--log_factor", type=float, default=1.)
    parser.add_argument("--print_iter", type=int, default=1)
    parser.add_argument("--save_iter", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument('--wdecay', type=float, default=0.00001)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument("--valid_ratio", type=float, default=0.01)
    parser.add_argument("--lambda_motion_loss", type=float, default=1.)
    parser.add_argument("--lambda_visibility_loss", type=float, default=1.)
    parser.add_argument("--optimizer_path", type=str, default=None)
    parser.add_argument("--restore_steps", type=int, default=0)
    parser.add_argument('--movif', action='store_true')
    parser.add_argument('--movif_cvo', action='store_true')
    parser.add_argument("--movif_stride", type=int, default=1)
    parser.add_argument('--alternate_corr', action='store_true', help='use efficent correlation implementation')
    parser.add_argument("--accumulation_steps", type=int, default=1)

    # evaluation
    parser.add_argument("--split", type=str, choices=["clean", "final", "extended"], default="final")
    parser.add_argument("--filter", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--filter_indices', type=int, nargs="+",
                        default=[70, 77, 93, 96, 140, 143, 162, 172, 174, 179, 187, 215, 236, 284, 285, 293, 330,
                                 358, 368, 402, 415, 458, 483, 495, 534])
    parser.add_argument('--plot_indices', type=int, nargs="+", default=[])
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--resize2train_size', action='store_true')
    parser.add_argument("--vis_only", action='store_true')

    # abla
    parser.add_argument('--disable_sensory_mem', action='store_true')
    parser.add_argument('--enable_value_reinforce', action='store_true')
    parser.add_argument('--disable_occ_warmup', action='store_true')
    parser.add_argument('--enable_gma', action='store_true')
    parser.add_argument('--detach_splat', action='store_true')
    parser.add_argument('--disable_random_sample', action='store_true')
    parser.add_argument('--ablate_key_proj', action='store_true')
    parser.add_argument('--ablate_net_warmup', action='store_true')
    parser.add_argument('--ablate_flow_warmup', action='store_true')
    parser.add_argument("--infer_num_iter", type=int, default=16)
    parser.add_argument("--splatting_type", type=str, choices=['summation', 'average', 'linear', 'softmax'], default="linear")
    parser.add_argument('--ablate_fuser', action='store_true')
    parser.add_argument('--extrapolate', type=float, default=2)

    args = parser.parse_args()

    os.environ['CUDA_VISIBLE_DEVICES'] = args.GPU_ids
    if args.DDP:
        args.world_size = args.nodes * args.gpus
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '22324'
    else:
        args.world_size = 1

    torch.manual_seed(1234)
    np.random.seed(1234)

    if args.datetime is None:
        args.datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    name = f"{args.name}_{args.datetime}"
    if hasattr(args, 'split'):
        name += f"_{args.split}"
    args.checkpoint_path = f"ckpts/{name}"
    args.log_path = f"logs/{name}"
    args.result_path = f"results/{name}"
    # main(args)
    mp.spawn(main, nprocs=args.world_size, args=(args,))
    print("Done.")
