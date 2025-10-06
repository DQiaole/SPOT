import os
import os.path as osp
from collections import OrderedDict
import lmdb
import numpy as np
import pyarrow as pa
import torch
import torch.utils.data as data
from spot.utils.torch import get_alpha_consistency
from torch.nn import functional as F


class FlowAugmentor:
    def __init__(self, size):
        # spatial augmentation params
        self.crop_size = (size, size) if isinstance(size, int) else size

    def spatial_transform(self, sample_dict):
        # randomly crop
        ht, wd = list(sample_dict.values())[0].shape[:2]
        if ht == self.crop_size[0]:
            y0 = 0
        else:
            y0 = np.random.randint(0, ht - self.crop_size[0])
        if wd == self.crop_size[1]:
            x0 = 0
        else:
            x0 = np.random.randint(0, wd - self.crop_size[1])

        def crop_fn(x):
            return x[y0 : y0 + self.crop_size[0], x0 : x0 + self.crop_size[1], :]

        for k, v in sample_dict.items():
            sample_dict[k] = crop_fn(v)

        return sample_dict

    def __call__(self, sample_dict):
        # random crop
        sample_dict = self.spatial_transform(sample_dict)
        return sample_dict


def totensor(x):
    return torch.from_numpy(x).permute(2, 0, 1).float()


class CVO_sampler_lmdb:
    """Data sampling"""

    all_keys = ["imgs", "imgs_blur", "fflows", "bflows", "delta_fflows", "delta_bflows", "flow", "alpha"]

    def __init__(self, is_training=True, keys=None, path=None):
        dst_dir = os.path.join("./datasets", "kubric", "cvo")
        if is_training:
            self.db_path = osp.join(dst_dir, "cvo_train.lmdb") if path is None else osp.join(dst_dir, path)
        else:
            self.db_path = osp.join(dst_dir, "cvo_test.lmdb")

        self.env = lmdb.open(
            self.db_path,
            subdir=os.path.isdir(self.db_path),
            readonly=True,
            lock=False,
            readahead=False,
            meminit=False,
        )
        with self.env.begin(write=False) as txn:
            self.samples = pa.deserialize(txn.get(b"__samples__"))
            self.length = len(self.samples)

        self.keys = self.all_keys if keys is None else [x.lower() for x in keys]
        self._check_keys(self.keys)

    def _check_keys(self, keys):
        # check keys are supported:
        for k in keys:
            assert k in self.all_keys, f"Invalid key value: {k}"

    def __len__(self):
        return self.length

    def sample(self, index):
        sample = OrderedDict()
        with self.env.begin(write=False) as txn:
            for k in self.keys:
                key = "{:05d}_{:s}".format(index, k)
                value = pa.deserialize(txn.get(key.encode()))
                if "flow" in key:  # Convert Int to Floating
                    value = value.astype(np.float32)
                    value = (value - 2**15) / 128.0
                if "alpha" in key:
                    value = value.astype(np.float32)
                sample[k] = value
        return sample


class CVO(data.Dataset):
    all_keys = ["fflows", "bflows", "delta_fflows", "delta_bflows"]

    def __init__(self, keys=None, split="clean", is_training=True, crop_size=256):
        self.augmentor = FlowAugmentor(crop_size) if is_training else None

        keys = self.all_keys if keys is None else [x.lower() for x in keys]
        self._check_keys(keys)
        if split == "clean":
            keys.append("imgs")
        else:
            keys.append("imgs_blur")

        self.sampler = CVO_sampler_lmdb(is_training, keys)

    def __getitem__(self, index):
        sample_dict = self.sampler.sample(index)
        if self.augmentor is not None:
            sample_dict = self.augmentor(sample_dict)

        out_dict = {"type": "dense",
                    }
        for k, v in sample_dict.items():
            v_ = totensor(np.ascontiguousarray(v).copy())
            if "imgs" in k:
                value = v_ / 255.0
                # value = value.split(3, dim=0)
                # assert len(value) == 7, len(value)
                out_dict["imgs"] = value.reshape(7, 3, value.shape[1], value.shape[2])
            elif "flow" in k:
                # value = v_.split(2, dim=0)
                # assert len(value) in [5, 6], len(value)
                out_dict[k] = v_.reshape(-1, 2, v_.shape[1], v_.shape[2])
            else:
                raise ValueError()

        indx_f = np.random.randint(0, 6)
        indx_b = np.random.randint(0, 6)

        out_dict["src_frame"] = torch.cat([out_dict["imgs"][indx_f:indx_f+1], out_dict["imgs"][indx_b+1:indx_b+2],
                                           out_dict["imgs"][:1].repeat(5, 1, 1, 1), out_dict["imgs"][2:]], dim=0)
        out_dict["tgt_frame"] = torch.cat([out_dict["imgs"][indx_f+1:indx_f+2], out_dict["imgs"][indx_b:indx_b+1],
                                           out_dict["imgs"][2:], out_dict["imgs"][:1].repeat(5, 1, 1, 1)], dim=0)
        out_dict["flow"] = torch.cat([out_dict["delta_fflows"][indx_f:indx_f+1], out_dict["delta_bflows"][indx_b:indx_b+1],
                                      out_dict["fflows"], out_dict["bflows"]], dim=0)

        thresh_1 = 0.01
        thresh_2 = 0.5
        alpha = get_alpha_consistency(out_dict["flow"].permute(0, 2, 3, 1),
                                      torch.cat([out_dict["delta_bflows"][indx_f:indx_f+1], out_dict["delta_fflows"][indx_b:indx_b+1],
                                                 out_dict["bflows"], out_dict["fflows"]], dim=0).permute(0, 2, 3, 1),
                                      thresh_1=thresh_1, thresh_2=thresh_2)

        out_dict["alpha"] = alpha
        out_dict.pop("imgs")
        out_dict.pop("delta_bflows")
        out_dict.pop("delta_fflows")
        out_dict.pop("fflows")
        out_dict.pop("bflows")
        return out_dict

    def _check_keys(self, keys):
        # check keys are supported:
        for k in keys:
            assert k in self.all_keys, f"Invalid key value: {k}"

    def __len__(self):
        return len(self.sampler)


class CVO_video(data.Dataset):
    all_keys = ["flow", "alpha"]

    def __init__(self, keys=None, split="clean", is_training=True, crop_size=256, args=None):
        self.augmentor = FlowAugmentor(crop_size) if is_training else None
        if is_training:
            self.crop_size = (crop_size, crop_size) if isinstance(crop_size, int) else crop_size

        keys = self.all_keys if keys is None else [x.lower() for x in keys]
        self._check_keys(keys)
        if split == "clean":
            keys.append("imgs")
        else:
            keys.append("imgs_blur")

        self.sampler = CVO_sampler_lmdb(is_training, keys, path="cvo_train_dql.lmdb")

    def __getitem__(self, index):
        sample_dict = self.sampler.sample(index)
        resize = False
        if self.augmentor is not None:
            if np.random.rand() < 0.25:
                resize = True
            else:
                sample_dict = self.augmentor(sample_dict)

        out_dict = {"type": "dense",
                    }
        for k, v in sample_dict.items():
            v_ = totensor(np.ascontiguousarray(v).copy())
            if "imgs" in k:
                value = v_ / 255.0
                out_dict["imgs"] = value.reshape(7, 3, value.shape[1], value.shape[2])
                if resize:
                    out_dict["imgs"] = F.interpolate(out_dict["imgs"], size=self.crop_size, mode="bilinear")
            elif "flow" in k:
                out_dict[k] = v_.reshape(-1, 2, v_.shape[1], v_.shape[2])
                if resize:
                    resize_ratio = torch.tensor([self.crop_size[1] / v_.shape[2], self.crop_size[0] / v_.shape[1]]).view(1, 2, 1, 1)
                    out_dict[k] = F.interpolate(out_dict[k], size=self.crop_size, mode="bilinear") * resize_ratio
            elif "alpha" in k:
                out_dict[k] = v_
                if resize:
                    out_dict[k] = F.interpolate(out_dict[k].unsqueeze(1), size=self.crop_size, mode="bilinear").squeeze(1)
            else:
                raise ValueError()

        return out_dict

    def _check_keys(self, keys):
        # check keys are supported:
        for k in keys:
            assert k in self.all_keys, f"Invalid key value: {k}"

    def __len__(self):
        return len(self.sampler)


def create_cvo_dataset(args, keys, batch=16, crop_size=256, split="clean", workers=0):
    """Create the data loader"""
    if "+" in split:
        dataset_clean = CVO(
            keys=keys,
            split="clean",
            is_training=True,
            crop_size=crop_size,
        )
        dataset_final = CVO(
            keys=keys,
            split="final",
            is_training=True,
            crop_size=crop_size,
        )
        dataset = dataset_clean + dataset_final
    else:
        dataset = CVO(
            keys=keys,
            split=split,
            is_training=True,
            crop_size=crop_size,
        )

    # train_sampler = DistributedSampler(dataset, num_replicas=args.world_size, rank=args.rank, shuffle=True)
    # train_loader = data.DataLoader(dataset, batch_size=batch,
    #                                pin_memory=True, shuffle=False, num_workers=workers, sampler=train_sampler)
    train_loader = data.DataLoader(dataset, batch_size=batch,
                                   pin_memory=True, shuffle=True, num_workers=workers, drop_last=True)
    return train_loader


def create_cvo_dataset_video(args, keys, batch=16, crop_size=256, split="clean", workers=0):
    """Create the data loader"""
    if "+" in split:
        dataset_clean = CVO_video(
            keys=keys,
            split="clean",
            is_training=True,
            crop_size=crop_size,
            args=args,
        )
        dataset_final = CVO_video(
            keys=keys,
            split="final",
            is_training=True,
            crop_size=crop_size,
            args=args,
        )
        dataset = dataset_clean + dataset_final
    else:
        dataset = CVO_video(
            keys=keys,
            split=split,
            is_training=True,
            crop_size=crop_size,
        )

    train_loader = data.DataLoader(dataset, batch_size=batch,
                                   pin_memory=True, shuffle=True, num_workers=workers, drop_last=True)
    return train_loader
