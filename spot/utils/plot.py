import os.path as osp
import matplotlib
import matplotlib.pyplot as plt
import torch
from spot.utils.io import create_folder
import numpy as np


def make_colorwheel():
    """
    Generates a color wheel for optical flow visualization as presented in:
        Baker et al. "A Database and Evaluation Methodology for Optical Flow" (ICCV, 2007)
        URL: http://vision.middlebury.edu/flow/flowEval-iccv07.pdf

    Code follows the original C++ source code of Daniel Scharstein.
    Code follows the the Matlab source code of Deqing Sun.

    Returns:
        np.ndarray: Color wheel
    """

    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))
    col = 0

    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255*np.arange(0,RY)/RY)
    col = col+RY
    # YG
    colorwheel[col:col+YG, 0] = 255 - np.floor(255*np.arange(0,YG)/YG)
    colorwheel[col:col+YG, 1] = 255
    col = col+YG
    # GC
    colorwheel[col:col+GC, 1] = 255
    colorwheel[col:col+GC, 2] = np.floor(255*np.arange(0,GC)/GC)
    col = col+GC
    # CB
    colorwheel[col:col+CB, 1] = 255 - np.floor(255*np.arange(CB)/CB)
    colorwheel[col:col+CB, 2] = 255
    col = col+CB
    # BM
    colorwheel[col:col+BM, 2] = 255
    colorwheel[col:col+BM, 0] = np.floor(255*np.arange(0,BM)/BM)
    col = col+BM
    # MR
    colorwheel[col:col+MR, 2] = 255 - np.floor(255*np.arange(MR)/MR)
    colorwheel[col:col+MR, 0] = 255
    return colorwheel


def flow_uv_to_colors(u, v, convert_to_bgr=False):
    """
    Applies the flow color wheel to (possibly clipped) flow components u and v.

    According to the C++ source code of Daniel Scharstein
    According to the Matlab source code of Deqing Sun

    Args:
        u (np.ndarray): Input horizontal flow of shape [H,W]
        v (np.ndarray): Input vertical flow of shape [H,W]
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    flow_image = np.zeros((u.shape[0], u.shape[1], 3), np.uint8)
    colorwheel = make_colorwheel()  # shape [55x3]
    ncols = colorwheel.shape[0]
    rad = np.sqrt(np.square(u) + np.square(v))
    a = np.arctan2(-v, -u)/np.pi
    fk = (a+1) / 2*(ncols-1)
    k0 = np.floor(fk).astype(np.int32)
    k1 = k0 + 1
    k1[k1 == ncols] = 0
    f = fk - k0
    for i in range(colorwheel.shape[1]):
        tmp = colorwheel[:,i]
        col0 = tmp[k0] / 255.0
        col1 = tmp[k1] / 255.0
        col = (1-f)*col0 + f*col1
        idx = (rad <= 1)
        col[idx]  = 1 - rad[idx] * (1-col[idx])
        col[~idx] = col[~idx] * 0.75   # out of range
        # Note the 2-i => BGR instead of RGB
        ch_idx = 2-i if convert_to_bgr else i
        flow_image[:,:,ch_idx] = np.floor(255 * col)
    return flow_image


def flow_to_image(flow_uv, clip_flow=None, convert_to_bgr=False):
    """
    Expects a two dimensional flow image of shape.

    Args:
        flow_uv (np.ndarray): Flow UV image of shape [H,W,2]
        clip_flow (float, optional): Clip maximum of flow values. Defaults to None.
        convert_to_bgr (bool, optional): Convert output image to BGR. Defaults to False.

    Returns:
        np.ndarray: Flow visualization image of shape [H,W,3]
    """
    assert flow_uv.ndim == 3, 'input flow must have three dimensions'
    assert flow_uv.shape[2] == 2, 'input flow must have shape [H,W,2]'
    if clip_flow is not None:
        flow_uv = np.clip(flow_uv, 0, clip_flow)
    u = flow_uv[:,:,0]
    v = flow_uv[:,:,1]
    rad = np.sqrt(np.square(u) + np.square(v))
    rad_max = np.max(rad)
    epsilon = 1e-5
    u = u / (rad_max + epsilon)
    v = v / (rad_max + epsilon)
    return flow_uv_to_colors(u, v, convert_to_bgr)


def to_rgb(tensor, mode, tracks=None, is_video=False, to_torch=True, reshape_as_video=False):
    if isinstance(tensor, list):
        tensor = torch.stack(tensor)
    tensor = tensor.cpu().numpy()
    if is_video:
        batch_size, time_steps = tensor.shape[:2]
    if mode == "flow":
        height, width = tensor.shape[-3: -1]
        tensor = np.reshape(tensor, (-1, height, width, 2))

        tensor = [flow_to_image(flow) for flow in tensor]
        tensor = np.stack(tensor, axis=0)
        tensor = np.moveaxis(tensor, -1, -3)
        # tensor = flow_to_rgb(tensor)
    elif mode == "mask":
        height, width = tensor.shape[-2:]
        tensor = np.reshape(tensor, (-1, 1, height, width))
        tensor = np.repeat(tensor, 3, axis=1)
    else:
        height, width = tensor.shape[-2:]
        tensor = np.reshape(tensor, (-1, 3, height, width))
        if tracks is not None:
            samples = tracks.size(-2)
            tracks = tracks.cpu().numpy()
            tracks = np.reshape(tracks, (-1, samples, 3))
            traj, occ = tracks[..., :2], 1 - tracks[..., 2]
            if is_video:
                tensor = np.reshape(tensor, (-1, time_steps, 3, height, width))
                traj = np.reshape(traj, (-1, time_steps, samples, 2))
                occ = np.reshape(occ, (-1, time_steps, samples))
                new_tensor = []
                for t in range(time_steps):
                    pos_t = traj[:, t]
                    occ_t = occ[:, t]
                    new_tensor.append(plot_tracks(tensor[:, t], pos_t, occ_t, tracks=traj[:, :t + 1]))
                tensor = np.stack(new_tensor, axis=1)
            else:
                tensor = plot_tracks(tensor, traj, occ)
    if is_video and reshape_as_video:
        tensor = np.reshape(tensor, (batch_size, time_steps, 3, height, width))
    else:
        tensor = np.reshape(tensor, (-1, 3, height, width))
    if to_torch:
        tensor = torch.from_numpy(tensor)
    return tensor


def flow_to_rgb(flow, transparent=False):
    flow = np.copy(flow)
    H, W = flow.shape[-3: -1]
    mul = 20.
    scaling = mul / (H ** 2 + W ** 2) ** 0.5
    direction = (np.arctan2(flow[..., 0], flow[..., 1]) + np.pi) / (2 * np.pi)
    norm = np.linalg.norm(flow, axis=-1)
    magnitude = np.clip(norm * scaling, 0., 1.)
    saturation = np.ones_like(direction)
    if transparent:
        hsv = np.stack([direction, saturation, np.ones_like(magnitude)], axis=-1)
    else:
        hsv = np.stack([direction, saturation, magnitude], axis=-1)
    rgb = matplotlib.colors.hsv_to_rgb(hsv)
    rgb = np.moveaxis(rgb, -1, -3)
    if transparent:
        return np.concatenate([rgb, np.expand_dims(magnitude, axis=-3)], axis=-3)
    return rgb


def plot_tracks(rgb, points, occluded, tracks=None, trackgroup=None):
    """Plot tracks with matplotlib.
    Adapted from: https://github.com/google-research/kubric/blob/main/challenges/point_tracking/dataset.py"""
    rgb = rgb.transpose(0, 2, 3, 1)
    _, height, width, _ = rgb.shape
    points = points.transpose(1, 0, 2).copy()  # clone, otherwise it updates points array
    # points[..., 0] *= (width - 1)
    # points[..., 1] *= (height - 1)
    if tracks is not None:
        tracks = tracks.copy()
        # tracks[..., 0] *= (width - 1)
        # tracks[..., 1] *= (height - 1)
    if occluded is not None:
        occluded = occluded.transpose(1, 0)
    disp = []
    cmap = plt.cm.hsv

    z_list = np.arange(points.shape[0]) if trackgroup is None else np.array(trackgroup)
    # random permutation of the colors so nearby points in the list can get different colors
    np.random.seed(0)
    z_list = np.random.permutation(np.max(z_list) + 1)[z_list]
    colors = cmap(z_list / (np.max(z_list) + 1))
    figure_dpi = 64

    for i in range(rgb.shape[0]):
        fig = plt.figure(
            figsize=(width / figure_dpi, height / figure_dpi),
            dpi=figure_dpi,
            frameon=False,
            facecolor='w')
        ax = fig.add_subplot()
        ax.axis('off')
        ax.imshow(rgb[i])

        valid = points[:, i, 0] > 0
        valid = np.logical_and(valid, points[:, i, 0] < rgb.shape[2] - 1)
        valid = np.logical_and(valid, points[:, i, 1] > 0)
        valid = np.logical_and(valid, points[:, i, 1] < rgb.shape[1] - 1)

        if occluded is not None:
            colalpha = np.concatenate([colors[:, :-1], 1 - occluded[:, i:i + 1]], axis=1)
        else:
            colalpha = colors[:, :-1]
        # Note: matplotlib uses pixel coordinates, not raster.
        ax.scatter(
            points[valid, i, 0] - 0.5,
            points[valid, i, 1] - 0.5,
            s=3,
            c=colalpha[valid],
        )

        if tracks is not None:
            for j in range(tracks.shape[2]):
                track_color = colors[j]  # Use a different color for each track
                x = tracks[i, :, j, 0]
                y = tracks[i, :, j, 1]
                valid_track = x > 0
                valid_track = np.logical_and(valid_track, x < rgb.shape[2] - 1)
                valid_track = np.logical_and(valid_track, y > 0)
                valid_track = np.logical_and(valid_track, y < rgb.shape[1] - 1)
                ax.plot(x[valid_track] - 0.5, y[valid_track] - 0.5, color=track_color, marker=None)

        if occluded is not None:
            occ2 = occluded[:, i:i + 1]

            colalpha = np.concatenate([colors[:, :-1], occ2], axis=1)

            ax.scatter(
                points[valid, i, 0],
                points[valid, i, 1],
                s=20,
                facecolors='none',
                edgecolors=colalpha[valid],
            )

        plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        fig.canvas.draw()
        width, height = fig.get_size_inches() * fig.get_dpi()
        img = np.frombuffer(
            fig.canvas.tostring_rgb(),
            dtype='uint8').reshape(int(height), int(width), 3)
        disp.append(np.copy(img))
        plt.close("all")

    return np.stack(disp, axis=0).astype(float).transpose(0, 3, 1, 2) / 255 # TODO : inconsistent


def plot_points(src_frame, tgt_frame, src_points, tgt_points, save_path, max_points=256):
    _, H, W = src_frame.shape
    src_frame = src_frame.permute(1, 2, 0).cpu().numpy()
    tgt_frame = tgt_frame.permute(1, 2, 0).cpu().numpy()
    src_points = src_points.cpu().numpy()
    tgt_points = tgt_points.cpu().numpy()
    src_pos, src_alpha = src_points[..., :2], src_points[..., 2]
    tgt_pos, tgt_alpha = tgt_points[..., :2], tgt_points[..., 2]
    src_pos = np.stack([src_pos[..., 0] * (W - 1), src_pos[..., 1] * (H - 1)], axis=-1)
    tgt_pos = np.stack([tgt_pos[..., 0] * (W - 1), tgt_pos[..., 1] * (H - 1)], axis=-1)

    plt.figure()
    ax = plt.gca()
    P = 10
    plt.imshow(np.concatenate((src_frame, np.ones_like(src_frame[:, :P]), tgt_frame), axis=1))
    indices = np.random.choice(len(src_pos), size=min(max_points, len(src_pos)), replace=False)
    for i in indices:
        if src_alpha[i] == 1:
            ax.scatter(src_pos[i, 0], src_pos[i, 1], s=5, c="black", marker='x')
        else:
            ax.scatter(src_pos[i, 0], src_pos[i, 1], s=5, linewidths=1.5, c="black", marker='o')
            ax.scatter(src_pos[i, 0], src_pos[i, 1], s=2.5, c="white", marker='o')
        if tgt_alpha[i] == 1:
            ax.scatter(W + P + tgt_pos[i, 0], tgt_pos[i, 1], s=5, c="black", marker='x')
        else:
            ax.scatter(W + P + tgt_pos[i, 0], tgt_pos[i, 1], s=5, linewidths=1.5, c="black", marker='o')
            ax.scatter(W + P + tgt_pos[i, 0], tgt_pos[i, 1], s=2.5, c="white", marker='o')

        plt.plot([src_pos[i, 0], W + P + tgt_pos[i, 0]], [src_pos[i, 1], tgt_pos[i, 1]], linewidth=0.5, c="black")

    # Save
    ax.axis('off')
    plt.tight_layout()
    plt.subplots_adjust(wspace=0)
    create_folder(osp.dirname(save_path))
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()