import os
from glob import glob
import random
import numpy as np
import torch
from torch.utils import data
from torch.nn import functional as F
from spot.utils.io import read_video, read_tracks
from torch.utils.data.distributed import DistributedSampler


def create_point_tracking_dataset(args, batch=1, split="train", workers=None, verbose=False, DDP=False,
                                  rank=0):
    dataset = Dataset(args, split, verbose)
    if DDP:
        train_sampler = DistributedSampler(dataset, num_replicas=args.world_size,
                                           rank=rank, shuffle=True)
        train_loader = data.DataLoader(dataset, batch_size=args.batch_size // args.world_size,
                                       pin_memory=True, shuffle=False, num_workers=workers, sampler=train_sampler)
        return train_sampler, train_loader
    else:
        dataloader = data.DataLoader(dataset, batch_size=batch, pin_memory=True, shuffle=True, num_workers=workers,
                                     drop_last=True)
        return dataloader


def get_correspondences(track_path, src_step, tgt_step, num_tracks, crop_y0, crop_x0, target_H, target_W,
                        vis_src_only, stride=1):
    tracks = torch.from_numpy(read_tracks(track_path))
    tracks[..., 0] = (tracks[..., 0] - crop_x0) / (target_W - 1)
    tracks[..., 1] = (tracks[..., 1] - crop_y0) / (target_H - 1)
    src_points = tracks[:, src_step]
    tgt_points = []
    if src_step < tgt_step:
        for i in range(src_step+stride, tgt_step+stride, stride):
            tgt_points.append(tracks[:, i])
    else:
        for i in range(src_step-stride, tgt_step-stride, -stride):
            tgt_points.append(tracks[:, i])
    tgt_points = torch.stack(tgt_points, dim=1)

    if vis_src_only:
        src_alpha = src_points[..., 2]
        vis_idx = torch.nonzero(src_alpha * (src_points[..., 0] >= 0) * (src_points[..., 0] <= 1) *
                                (src_points[..., 1] >= 0) * (src_points[..., 1] <= 1), as_tuple=True)[0]
        num_vis = vis_idx.shape[0]
        if num_vis == 0:
            return False, None
        samples = np.random.choice(num_vis, num_tracks, replace=num_tracks > num_vis)
        idx = vis_idx[samples]
    else:
        idx = np.random.choice(tracks.size(0), num_tracks, replace=num_tracks > tracks.size(0))
    return True, (src_points[idx], tgt_points[idx])


class Dataset(data.Dataset):
    def __init__(self, args, split="train", verbose=False):
        super().__init__()
        self.input_frames = args.input_frames
        self.video_folder = os.path.join(args.data_root, "video")
        self.out_track_folder = os.path.join(args.data_root, args.out_track_name)
        self.num_out_tracks = args.num_out_tracks
        self.height = args.height
        self.width = args.width
        self.stride = args.movif_stride
        num_videos = len(glob(os.path.join(self.video_folder, "*")))
        self.video_steps = [
            len(glob(os.path.join(self.video_folder, str(video_idx), "*"))) for video_idx in range(num_videos)
        ]
        video_indices = list(range(num_videos))
        if split == "valid":
            video_indices = video_indices[:int(num_videos * args.valid_ratio)]
        elif split == "train":
            video_indices = video_indices[int(num_videos * args.valid_ratio):]
        self.video_indices = video_indices
        self.num_videos = len(video_indices)
        if verbose:
            print(f"Created {split} dataset of length {self.num_videos}")

    def __len__(self):
        return self.num_videos

    def __getitem__(self, idx):
        idx = idx % self.num_videos
        video_idx = self.video_indices[idx]
        time_steps = self.video_steps[video_idx]
        stride = random.randrange(self.stride) + 1
        src_step = random.randrange(time_steps - (self.input_frames - 1) * stride)
        tgt_step = src_step + (self.input_frames - 1) * stride

        video_path = os.path.join(self.video_folder, str(video_idx))
        imgs = read_video(video_path, start_step=src_step, time_steps=(self.input_frames - 1) * stride + 1)
        imgs = imgs[::stride]
        if np.random.rand() < 0.5:
            src_step, tgt_step = tgt_step, src_step
            imgs = imgs.flip(0)

        T, _, H, W = imgs.shape
        assert T == self.input_frames

        if np.random.rand() < 0.25 and (H != self.height or W != self.width):
            imgs = F.interpolate(imgs, size=(self.height, self.width), mode="bilinear")
            resize = True
            target_H, target_W = H, W
        else:
            resize = False
            target_H, target_W = self.height, self.width

        if resize or H == self.height:
            y0 = 0
        else:
            y0 = np.random.randint(0, H - self.height)
        if resize or W == self.width:
            x0 = 0
        else:
            x0 = np.random.randint(0, W - self.width)

        imgs = imgs[:, :, y0: y0 + self.height, x0: x0 + self.width]

        out_track_path = os.path.join(self.out_track_folder, f"{video_idx}.npy")

        vis_src_only = True
        success, corr = get_correspondences(out_track_path, src_step, tgt_step, self.num_out_tracks, y0,
                                            x0, target_H, target_W, vis_src_only, stride=stride)
        if not success:
            return self[idx + 1]
        out_src_points, out_tgt_points = corr

        data = {
            "imgs": imgs,
            "out_src_points": out_src_points,
            "out_tgt_points": out_tgt_points,
            "type": "sparse",
        }

        return data
