import os
import torch
import torch.nn as nn
import argparse
from datetime import datetime
torch.backends.cudnn.benchmark = True
from einops import repeat, rearrange
from torch.optim.lr_scheduler import OneCycleLR
from spot.data.cvo_dataset import create_cvo_dataset
from spot.utils.io import create_folder, read_config
from spot.utils.torch import to_device
from spot.utils.log import Logger
from spot.models.shelf import RAFT
from spot.utils.options.base_options import str2bool
from torch.cuda.amp import GradScaler
import numpy as np


def checkpoint(model, optimizer, path, name):
    create_folder(path)
    model_path = os.path.join(path, f"{name}.pth")
    optimizer_path = os.path.join(path, f"{name}_optimizer.pth")
    torch.save(model.module.state_dict(), model_path)
    torch.save(optimizer.state_dict(), optimizer_path)


def sample(pred, gt):
    B, I, H, W, _ = pred["flow"].shape
    dense = torch.cat([pred["flow"], pred["alpha"][..., None]], dim=-1)
    dense = rearrange(dense, "b i h w c -> b (i c) h w")
    src_pos = gt["out_src_points"][..., :2]
    grid = src_pos[:, None] * 2 - 1
    sparse = torch.nn.functional.grid_sample(dense, grid, mode="nearest", align_corners=True, padding_mode="border")
    sparse = rearrange(sparse, "b (i c) h w -> b i (h w) c", i=I)
    delta_pos, tgt_alpha = sparse[..., :2], sparse[..., 2:]
    delta_pos[..., 0] = delta_pos[..., 0] / (W - 1)
    delta_pos[..., 1] = delta_pos[..., 1] / (H - 1)
    tgt_pos = src_pos[:, None] + delta_pos
    out_tgt_points = torch.cat([tgt_pos, tgt_alpha], dim=-1)
    pred = {
        "flow": pred["flow"][:, -1],
        "alpha": pred["alpha"][:, -1],
        "out_tgt_points": out_tgt_points
    }
    return pred


def calculate_sparse_corr_loss(args, pred, gt, loss):
    pred = sample(pred, gt)
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
    loss += motion_loss * args.lambda_motion_loss
    loss += visibility_loss * args.lambda_visibility_loss
    return loss, motion_loss, visibility_loss


def calculate_dense_flow_loss(args, pred, gt, loss):
    B, I, H, W, _ = pred["flow"].shape
    gamma = 0.8
    n_predictions = pred["flow"].size(1)
    motion_loss = torch.tensor(0., requires_grad=True).cuda()
    visibility_loss = torch.tensor(0., requires_grad=True).cuda()
    for i in range(n_predictions):
        i_weight = gamma ** (n_predictions - i - 1)
        flow_pre = pred["flow"][:, i].permute(0, 3, 1, 2)
        i_loss = (flow_pre - gt["flow"]).abs()
        motion_loss += i_weight * i_loss.mean()
        visibility_loss += i_weight * torch.nn.functional.binary_cross_entropy_with_logits(pred["alpha"][:, i], gt["alpha"])

    loss += motion_loss * args.lambda_motion_loss
    loss += visibility_loss * args.lambda_visibility_loss

    epe = torch.sum((pred["flow"][:, -1].permute(0, 3, 1, 2) - gt["flow"]) ** 2, dim=1).sqrt()
    epe = epe.view(-1).mean()

    return loss, motion_loss, visibility_loss, epe


def step(gt, model, optimizer, scheduler, logger, global_iter, args, scaler=None):
    if optimizer is not None:
        optimizer.zero_grad()

    loss = torch.tensor(0., requires_grad=True).cuda()

    for key, value in gt.items():
        if torch.is_tensor(value):
            gt[key] = value.flatten(start_dim=0, end_dim=1).cuda()

    pred = model(src_frame=gt["src_frame"], tgt_frame=gt["tgt_frame"], is_train=True, num_iter=args.num_iter)

    if gt["type"][0] == "sparse":
        loss, motion_loss, visibility_loss = calculate_sparse_corr_loss(args, pred, gt, loss)
        epe = 0.0
    elif gt["type"][0] == "dense":
        loss, motion_loss, visibility_loss, epe = calculate_dense_flow_loss(args, pred, gt, loss)
    else:
        raise NotImplementedError

    if optimizer is not None:
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(optimizer)
        scheduler.step()
        scaler.update()

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
            losses += f"visibility: {visibility_loss.item():.3E})"
            print(f"[Iter {global_iter}/{args.train_iter}] {losses}")

        if global_iter % args.log_iter == 0:
            logger.log_image("flow/pred_flow", pred["flow"][:, -1], "flow", 2, global_iter)
            logger.log_image("alpha/pred_alpha", (pred["alpha"][:, -1].sigmoid() > 0.8).float(), "mask", 2, global_iter)
            logger.log_image("img/tgt_frame", gt["tgt_frame"], "rgb", 2, global_iter)
            logger.log_image("img/src_frame", gt["src_frame"], "rgb", 2, global_iter)
            logger.log_image("flow/gt_flow", gt["flow"].permute(0, 2, 3, 1), "flow", 2, global_iter)
            logger.log_image("alpha/gt_alpha", gt["alpha"], "mask", 2, global_iter)

    return loss


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def main(args):
    logger = Logger(args)

    # Load model and optimizer
    model = RAFT(read_config(args.refiner_config), mixed_precision=args.mixed_precision)

    if args.refiner_path is not None:
        model.load_state_dict(torch.load(args.refiner_path, map_location='cpu'))

    model = nn.DataParallel(model, device_ids=args.gpus)
    print("Parameter Count: %d" % count_parameters(model))
    model.cuda()
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0, 0.99),
                                 weight_decay=args.wdecay, eps=args.epsilon)
    scheduler = OneCycleLR(optimizer, args.lr, args.train_iter + 100,
                           pct_start=0.05, cycle_momentum=False, anneal_strategy='linear')

    if args.optimizer_path is not None:
        optimizer.load_state_dict(torch.load(args.optimizer_path))
        assert args.restore_steps > 0
        for _ in range(args.restore_steps):
            scheduler.step()

    train_loader = create_cvo_dataset(
        args,
        keys=None,
        batch=args.batch_size,
        crop_size=(args.height, args.width),
        split="clean+final",
        workers=args.num_workers,
    )

    scaler = GradScaler(enabled=args.mixed_precision)

    global_iter = args.restore_steps + 1
    should_keep_training = True
    while should_keep_training:

        for i_batch, data_blob in enumerate(train_loader):
            # if global_iter % 2 == 0:
            #     step(movif_train_dataset, model, optimizer, scheduler, logger, global_iter, args, scaler)
            # else:
            step(data_blob, model, optimizer, scheduler, logger, global_iter, args, scaler)

            if global_iter % args.save_iter == 0:
                checkpoint(model, optimizer, args.checkpoint_path, str(global_iter) + '_it')

            global_iter += 1
            if global_iter >= args.train_iter + 1:
                should_keep_training = False
                break

    logger.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", type=str)
    parser.add_argument("--datetime", type=str, default=None)
    parser.add_argument("--data_root", type=str, default="datasets/kubric/movi_f")
    parser.add_argument("--height", type=int, default=384)
    parser.add_argument("--width", type=int, default=384)
    parser.add_argument("--aspect_ratio", type=float, default=1)
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--num_tracks", type=int, default=2048)
    parser.add_argument("--alpha_thresh", type=float, default=0.8)
    parser.add_argument("--is_train", type=str2bool, nargs='?', const=True, default=False)

    # Parallelization
    parser.add_argument('--worker_idx', type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])

    # Optical flow estimator
    parser.add_argument("--refiner_config", type=str, default="configs/raft_patch_4_alpha.json")
    parser.add_argument("--refiner_path", type=str, default=None)
    parser.add_argument("--num_iter", type=int, default=12)

    # training
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
    parser.add_argument("--out_track_name", type=str, default="ground_truth")
    parser.add_argument("--num_out_tracks", type=int, default=2048)
    parser.add_argument("--train_iter", type=int, default=100000)
    parser.add_argument("--log_iter", type=int, default=500)
    parser.add_argument("--log_factor", type=float, default=1.)
    parser.add_argument("--print_iter", type=int, default=100)
    parser.add_argument("--save_iter", type=int, default=5000)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument('--wdecay', type=float, default=0.00001)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument("--valid_ratio", type=float, default=0.01)
    parser.add_argument("--lambda_motion_loss", type=float, default=1000.)
    parser.add_argument("--lambda_visibility_loss", type=float, default=1.)
    parser.add_argument("--optimizer_path", type=str, default=None)
    parser.add_argument("--restore_steps", type=int, default=0)

    # evaluation
    parser.add_argument("--split", type=str, choices=["clean", "final", "extended"], default="final")
    parser.add_argument("--filter", type=str2bool, nargs='?', const=True, default=True)
    parser.add_argument('--filter_indices', type=int, nargs="+",
                        default=[70, 77, 93, 96, 140, 143, 162, 172, 174, 179, 187, 215, 236, 284, 285, 293, 330,
                                 358, 368, 402, 415, 458, 483, 495, 534])
    parser.add_argument('--plot_indices', type=int, nargs="+", default=[])
    parser.add_argument('--eval_only', action='store_true')
    parser.add_argument('--resize2train_size', action='store_true')
    args = parser.parse_args()

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
    main(args)
    print("Done.")
