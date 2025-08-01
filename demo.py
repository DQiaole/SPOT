import torch
import argparse
torch.backends.cudnn.benchmark = True
from spot.utils.io import read_config
from spot.models.shelf import SPOT
from spot.utils.options.base_options import str2bool
from datetime import datetime
from inference import inference_core
import torch
from torch import nn
import os.path as osp
from tqdm import tqdm
from matplotlib import colormaps
import numpy as np
import scipy
import math
from spot.utils.io import create_folder, write_video, read_video, read_frame
from spot.utils.torch import to_device, get_grid



class Visualizer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.save_mode = args.save_mode
        self.overlay_factor = args.overlay_factor
        self.spaghetti_radius = args.spaghetti_radius
        self.spaghetti_length = args.spaghetti_length
        self.spaghetti_grid = args.spaghetti_grid
        self.spaghetti_scale = args.spaghetti_scale
        self.spaghetti_every = args.spaghetti_every
        self.spaghetti_dropout = args.spaghetti_dropout

    def forward(self, data, mode, name):
        if "overlay" in mode:
            video = self.plot_overlay(data, mode)
        elif "spaghetti" in mode:
            video = self.plot_spaghetti(data, mode)
        else:
            raise ValueError(f"Unknown mode {mode}")
        save_path = osp.join(name, mode) + ".mp4" if self.save_mode == "video" else name
        write_video(video, save_path)

    def plot_overlay(self, data, mode):
        T, C, H, W = data["video"].shape
        mask = data["mask"] if "mask" in mode else torch.ones_like(data["mask"])
        tracks = data["tracks"]

        if tracks.ndim == 4:
            col = get_rainbow_colors(int(mask.sum())).cuda()
        else:
            col = get_rainbow_colors(tracks.size(1)).cuda()

        video = []
        for tgt_step in tqdm(range(T), leave=False, desc="Plot target frame"):
            tgt_frame = data["video"][tgt_step]
            tgt_frame = tgt_frame.permute(1, 2, 0)

            # Plot rainbow points
            tgt_pos = tracks[tgt_step, ..., :2]
            tgt_vis = tracks[tgt_step, ..., 2]
            if tracks.ndim == 4:
                tgt_pos = tgt_pos[mask]
                tgt_vis = tgt_vis[mask]
            rainbow, alpha = draw(tgt_pos, tgt_vis, col, H, W)

            # Plot rainbow points with white stripes in occluded regions
            if "stripes" in mode:
                rainbow_occ, alpha_occ = draw(tgt_pos, 1 - tgt_vis, col, H, W)
                stripes = torch.arange(H).view(-1, 1) + torch.arange(W).view(1, -1)
                stripes = stripes % 9 < 3
                rainbow_occ[stripes] = 1.
                rainbow = alpha * rainbow + (1 - alpha) * rainbow_occ
                alpha = alpha + (1 - alpha) * alpha_occ

            # Overlay rainbow points over target frame
            tgt_frame = self.overlay_factor * alpha * rainbow + (1 - self.overlay_factor * alpha) * tgt_frame

            # Convert from H W C to C H W
            tgt_frame = tgt_frame.permute(2, 0, 1)
            video.append(tgt_frame)
        video = torch.stack(video)
        return video

    def plot_spaghetti(self, data, mode):
        bg_color = 1.
        T, C, H, W = data["video"].shape
        G, S, R, L = self.spaghetti_grid, self.spaghetti_scale, self.spaghetti_radius, self.spaghetti_length
        D = self.spaghetti_dropout

        # Extract a grid of tracks
        mask = data["mask"] if "mask" in mode else torch.ones_like(data["mask"])
        mask = mask[G // 2:-G // 2 + 1:G, G // 2:-G // 2 + 1:G]
        tracks = data["tracks"]
        if tracks.ndim == 4:
            tracks = tracks[:, G // 2:-G // 2 + 1:G, G // 2:-G // 2 + 1:G]
            tracks = tracks[:, mask]
        elif D > 0:
            N = tracks.size(1)
            assert D < 1
            samples = np.sort(np.random.choice(N, int((1 - D) * N), replace=False))
            tracks = tracks[:, samples]
        col = get_rainbow_colors(tracks.size(1)).cuda()

        # Densify tracks over temporal axis
        tracks = spline_interpolation(tracks, length=L)

        video = []
        cur_frame = None
        cur_alpha = None
        grid = get_grid(H, W).cuda()
        grid[..., 0] *= (W - 1)
        grid[..., 1] *= (H - 1)
        for tgt_step in tqdm(range(T), leave=False, desc="Plot target frame"):
            for delta in range(L):
                # Plot rainbow points
                tgt_pos = tracks[tgt_step * L + delta, :, :2]
                tgt_vis = torch.ones_like(tgt_pos[..., 0])
                tgt_pos = project(tgt_pos, tgt_step * L + delta, T * L, H, W)
                tgt_col = col.clone()
                rainbow, alpha = draw(S * tgt_pos, tgt_vis, tgt_col, int(S * H), int(S * W), radius=R)
                rainbow, alpha = rainbow.cpu(), alpha.cpu()

                # Overlay rainbow points over previous points / frames
                if cur_frame is None:
                    cur_frame = rainbow
                    cur_alpha = alpha
                else:
                    cur_frame = alpha * rainbow + (1 - alpha) * cur_frame
                    cur_alpha = 1 - (1 - cur_alpha) * (1 - alpha)

                plot_first = "first" in mode and tgt_step == 0 and delta == 0
                plot_last = "last" in mode and delta == 0
                plot_every = "every" in mode and delta == 0 and tgt_step % self.spaghetti_every == 0
                if delta == 0:
                    if plot_first or plot_last or plot_every:
                        # Plot target frame
                        tgt_col = data["video"][tgt_step].permute(1, 2, 0).reshape(-1, 3)
                        tgt_pos = grid.view(-1, 2)
                        tgt_vis = torch.ones_like(tgt_pos[..., 0])
                        tgt_pos = project(tgt_pos, tgt_step * L + delta, T * L, H, W)
                        tgt_frame, alpha_frame = draw(S * tgt_pos, tgt_vis, tgt_col, int(S * H), int(S * W))
                        tgt_frame, alpha_frame = tgt_frame.cpu(), alpha_frame.cpu()

                        # Overlay target frame over previous points / frames
                        tgt_frame = alpha_frame * tgt_frame + (1 - alpha_frame) * cur_frame
                        alpha_frame = 1 - (1 - cur_alpha) * (1 - alpha_frame)

                        # Add last points on top
                        tgt_frame = alpha * rainbow + (1 - alpha) * tgt_frame
                        alpha_frame = 1 - (1 - alpha_frame) * (1 - alpha)

                        # Set background color
                        tgt_frame = alpha_frame * tgt_frame + (1 - alpha_frame) * torch.ones_like(tgt_frame) * bg_color

                        if plot_first or plot_every:
                            cur_frame = tgt_frame
                            cur_alpha = alpha_frame
                    else:
                        tgt_frame = cur_alpha * cur_frame + (1 - cur_alpha) * torch.ones_like(cur_frame) * bg_color

                    # Convert from H W C to C H W
                    tgt_frame = tgt_frame.permute(2, 0, 1)

                    # Translate everything to make the target frame look static
                    if "static" in mode:
                        end_pos = project(torch.tensor([[0, 0]]), T * L, T * L, H, W)[0]
                        cur_pos = project(torch.tensor([[0, 0]]), tgt_step * L + delta, T * L, H, W)[0]
                        delta_pos = S * (end_pos - cur_pos)
                        tgt_frame = translation(tgt_frame, delta_pos[0], delta_pos[1], bg_color)
                    video.append(tgt_frame)
        video = torch.stack(video)
        return video


def translation(frame, dx, dy, pad_value):
    C, H, W = frame.shape
    grid = get_grid(H, W, device=frame.device)
    grid[..., 0] = grid[..., 0] - (dx / (W - 1))
    grid[..., 1] = grid[..., 1] - (dy / (H - 1))
    frame = frame - pad_value
    frame = torch.nn.functional.grid_sample(frame[None], grid[None] * 2 - 1, mode='bilinear', align_corners=True)[0]
    frame = frame + pad_value
    return frame


def spline_interpolation(x, length=10):
    if length != 1:
        T, N, C = x.shape
        x = x.view(T, -1).cpu().numpy()
        original_time = np.arange(T)
        cs = scipy.interpolate.CubicSpline(original_time, x)
        new_time = np.linspace(original_time[0], original_time[-1], T * length)
        x = torch.from_numpy(cs(new_time)).view(-1, N, C).float().cuda()
    return x


def get_rainbow_colors(size):
    col_map = colormaps["jet"]
    col_range = np.array(range(size)) / (size - 1)
    col = torch.from_numpy(col_map(col_range)[..., :3]).float()
    col = col.view(-1, 3)
    return col


def draw(pos, vis, col, height, width, radius=1):
    H, W = height, width
    frame = torch.zeros(H * W, 4, device=pos.device)
    pos = pos[vis.bool()]
    col = col[vis.bool()]
    if radius > 1:
        pos, col = get_radius_neighbors(pos, col, radius)
    else:
        pos, col = get_cardinal_neighbors(pos, col)
    inbound = (pos[:, 0] >= 0) & (pos[:, 0] <= W - 1) & (pos[:, 1] >= 0) & (pos[:, 1] <= H - 1)
    pos = pos[inbound]
    col = col[inbound]
    pos = pos.round().long()
    idx = pos[:, 1] * W + pos[:, 0]
    idx = idx.view(-1, 1).expand(-1, 4)
    frame.scatter_add_(0, idx, col)
    frame = frame.view(H, W, 4)
    frame, alpha = frame[..., :3], frame[..., 3]
    nonzero = alpha > 0
    frame[nonzero] /= alpha[nonzero][..., None]
    alpha = nonzero[..., None].float()
    return frame, alpha


def get_cardinal_neighbors(pos, col, eps=0.01):
    pos_nw = torch.stack([pos[:, 0].floor(), pos[:, 1].floor()], dim=-1)
    pos_sw = torch.stack([pos[:, 0].floor(), pos[:, 1].floor() + 1], dim=-1)
    pos_ne = torch.stack([pos[:, 0].floor() + 1, pos[:, 1].floor()], dim=-1)
    pos_se = torch.stack([pos[:, 0].floor() + 1, pos[:, 1].floor() + 1], dim=-1)
    w_n = pos[:, 1].floor() + 1 - pos[:, 1] + eps
    w_s = pos[:, 1] - pos[:, 1].floor() + eps
    w_w = pos[:, 0].floor() + 1 - pos[:, 0] + eps
    w_e = pos[:, 0] - pos[:, 0].floor() + eps
    w_nw = (w_n * w_w)[:, None]
    w_sw = (w_s * w_w)[:, None]
    w_ne = (w_n * w_e)[:, None]
    w_se = (w_s * w_e)[:, None]
    col_nw = torch.cat([w_nw * col, w_nw], dim=-1)
    col_sw = torch.cat([w_sw * col, w_sw], dim=-1)
    col_ne = torch.cat([w_ne * col, w_ne], dim=-1)
    col_se = torch.cat([w_se * col, w_se], dim=-1)
    pos = torch.cat([pos_nw, pos_sw, pos_ne, pos_se], dim=0)
    col = torch.cat([col_nw, col_sw, col_ne, col_se], dim=0)
    return pos, col


def get_radius_neighbors(pos, col, radius):
    R = math.ceil(radius)
    center = torch.stack([pos[:, 0].round(), pos[:, 1].round()], dim=-1)
    nn = torch.arange(-R, R + 1)
    nn = torch.stack([nn[None, :].expand(2 * R + 1, -1), nn[:, None].expand(-1, 2 * R + 1)], dim=-1)
    nn = nn.view(-1, 2).cuda()
    in_radius = nn[:, 0] ** 2 + nn[:, 1] ** 2 <= radius ** 2
    nn = nn[in_radius]
    w = 1 - nn.pow(2).sum(-1).sqrt() / radius + 0.01
    w = w[None].expand(pos.size(0), -1).reshape(-1)
    pos = (center.view(-1, 1, 2) + nn.view(1, -1, 2)).view(-1, 2)
    col = col.view(-1, 1, 3).repeat(1, nn.size(0), 1)
    col = col.view(-1, 3)
    col = torch.cat([col * w[:, None], w[:, None]], dim=-1)
    return pos, col


def project(pos, t, time_steps, heigh, width):
    T, H, W = time_steps, heigh, width
    pos = torch.stack([pos[..., 0] / (W - 1), pos[..., 1] / (H - 1)], dim=-1)
    pos = pos - 0.5
    pos = pos * 0.25
    t = 1 - torch.ones_like(pos[..., :1]) * t / (T - 1)
    pos = torch.cat([pos, t], dim=-1)
    M = torch.tensor([
        [0.8, 0, 0.5],
        [-0.2, 1.0, 0.1],
        [0.0, 0.0, 0.0]
    ])
    pos = pos @ M.t().to(pos.device)
    pos = pos[..., :2]
    pos[..., 0] += 0.25
    pos[..., 1] += 0.45
    pos[..., 0] *= (W - 1)
    pos[..., 1] *= (H - 1)
    return pos


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


def main(args):
    # Load model
    model = SPOT(read_config(args.raft_config), mixed_precision=args.mixed_precision, cfg=args)
    if args.ckpt_path is not None:
        print("[Loading ckpt from {}]".format(args.ckpt_path))
        ckpt = torch.load(args.ckpt_path, map_location='cpu')
        torch_init_model(model, ckpt, key='model')

    print("Parameter Count: %d" % count_parameters(model))
    model.cuda()
    model.eval()

    visualizer = Visualizer(args).cuda()
    resolution = (args.height, args.width)
    create_folder(args.vis_dir)

    video = read_video(args.video_path, resolution=resolution).cuda()[None]
    print('video shape: ', video.shape)
    with torch.no_grad():
        # normalize
        video = video * 2 - 1
        B, T, C, H, W = video.shape
        grid = get_grid(H, W, device=video.device)
        grid[..., 0] *= (W - 1)
        grid[..., 1] *= (H - 1)

        tracks_from_src = []
        # current step
        flow = torch.zeros(B, H, W, 2, device=video.device)
        alpha = torch.ones(B, H, W, device=video.device)
        tracks_from_src.append(torch.cat([flow + grid, alpha[..., None]], dim=-1))
        # steps after current step
        processor = inference_core.InferenceCore(model, config=args)
        processor.set_first_frame(video[:, 0])
        for ti in tqdm(range(1, T)):
            pred = processor.step(video[:, ti], end=(ti == T - 1))
            alpha = (pred["alpha"].sigmoid() > 0.8).float()
            flow = pred["flow"]
            tracks_from_src.append(torch.cat([flow + grid, alpha[..., None]], dim=-1))

        # concat all time steps
        tracks_from_src = torch.stack(tracks_from_src, dim=1)
        tracks = tracks_from_src[0]
        

    mask_path = args.mask_path
    if any(["mask" in mode] for mode in args.visualization_modes) and osp.exists(mask_path):
        mask = torch.sum(read_frame(mask_path, resolution=resolution), dim=0) > 0.5
    else:
        mask = torch.ones(args.height, args.width).bool()

    data = {
        "video": (video[0] + 1) / 2,
        "tracks": tracks,
        "mask": mask
    }

    data = to_device(data, "cuda")

    if data["tracks"].ndim == 4 and args.rainbow_mode == "left_right":
        data["mask"] = data["mask"].permute(1, 0)
        data["tracks"] = data["tracks"].permute(0, 2, 1, 3)
    elif data["tracks"].ndim == 3:
        points = data["tracks"][0]
        x, y = points[..., 0].long(), points[..., 1].long()
        x, y = x - x.min(), y - y.min()
        if args.rainbow_mode == "left_right":
            idx = y + x * y.max()
        else:
            idx = x + y * x.max()
        order = idx.argsort(dim=0)
        data["tracks"] = data["tracks"][:, order]

    for mode in args.visualization_modes:
        visualizer(data, mode=mode, name=args.vis_dir)
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--vis_dir", type=str)
    parser.add_argument("--datetime", type=str, default=None)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--aspect_ratio", type=float, default=1)
    parser.add_argument("--batch_size", type=int, default=1)

    # Optical flow estimator
    parser.add_argument("--raft_config", type=str, default="configs/raft_patch_4_alpha.json")
    parser.add_argument("--ckpt_path", type=str, default=None)

    # args for inference
    parser.add_argument("--mem_every", type=int, default=1)
    parser.add_argument("--max_mid_term_frames", type=int, default=4)
    parser.add_argument("--min_mid_term_frames", type=int, default=3)

    # training
    parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')

    # abla
    parser.add_argument("--infer_num_iter", type=int, default=16)
    parser.add_argument("--splatting_type", type=str, choices=['summation', 'average', 'linear', 'softmax'],
                        default="linear")
    parser.add_argument('--extrapolate', type=float, default=2)

    # vis
    parser.add_argument("--visualization_modes", type=str, nargs="+", default=["overlay", "spaghetti_last_static"])
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--mask_path", type=str)
    parser.add_argument("--save_tracks", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--recompute_tracks", type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument("--overlay_factor", type=float, default=0.75)
    parser.add_argument("--rainbow_mode", type=str, default="left_right", choices=["left_right", "up_down"])
    parser.add_argument("--save_mode", type=str, default="video", choices=["image", "video"])
    parser.add_argument("--spaghetti_radius", type=float, default=1.5)
    parser.add_argument("--spaghetti_length", type=int, default=40)
    parser.add_argument("--spaghetti_grid", type=int, default=30)
    parser.add_argument("--spaghetti_scale", type=float, default=2)
    parser.add_argument("--spaghetti_every", type=int, default=10)
    parser.add_argument("--spaghetti_dropout", type=float, default=0)
    parser.set_defaults(vis_dir="demo", batch_size=1, height=480, width=856)

    args = parser.parse_args()
    if args.datetime is None:
        args.datetime = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    torch.manual_seed(1234)
    np.random.seed(1234)

    main(args)
    print("Done.")
