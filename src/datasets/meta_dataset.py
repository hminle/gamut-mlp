from pathlib import Path

import numpy as np
import torch
from torchvision.transforms import Compose, Normalize

from src.datasets.img_dataset import ImgDataset
from src.utils import TORCH_PI
from src.utils.color import (
    decode_prop,
    decode_srgb,
    encode_srgb,
    prop_to_srgb_cat02,
    srgb_to_prop_cat02,
    to_single,
    to_uint8,
)
from src.utils.image_func import get_wh_mgrid, read_image


class GMATinyCudnnMetaDataset(ImgDataset):
    def __init__(
        self,
        data_root: str,
        dataset_name: str,
    ):
        super(GMATinyCudnnMetaDataset, self).__init__(data_root, dataset_name)

    def __getitem__(self, idx):
        prop_img = read_image(self.image_paths[idx])
        prop_img = to_single(prop_img)
        ln_prop_img = decode_prop(prop_img)
        input, gt = self._prepocess_data(ln_prop_img)
        return input, gt

    def _prepocess_data(self, gt_ln_prop_img):
        sample = 1  # self.sample
        ln_srgb = prop_to_srgb_cat02(gt_ln_prop_img)
        bf_gma_mask = (
            ((ln_srgb < 0) | (ln_srgb > 1)).any(axis=2)
            & ~(ln_srgb < 0).all(axis=2)
            & ~(ln_srgb > 1).all(axis=2)
        )
        srgb = decode_srgb(to_single(to_uint8(encode_srgb(np.clip(ln_srgb, 0, 1)))))

        ln_prop = srgb_to_prop_cat02(srgb)

        ln_prop_tensor = torch.from_numpy(ln_prop)
        gt_ln_prop_tensor = torch.from_numpy(gt_ln_prop_img)

        x = get_wh_mgrid(gt_ln_prop_tensor.shape[0], gt_ln_prop_tensor.shape[1])
        x = x.float()

        # normalize
        transform = Compose([Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        ln_prop_tensor = ln_prop_tensor.permute(2, 0, 1)
        ln_prop_tensor = transform(ln_prop_tensor)
        ln_prop_tensor = ln_prop_tensor.permute(1, 2, 0)

        gt_ln_prop_tensor = gt_ln_prop_tensor.permute(2, 0, 1)
        gt_ln_prop_tensor = transform(gt_ln_prop_tensor)
        gt_ln_prop_tensor = gt_ln_prop_tensor.permute(1, 2, 0)

        coords = torch.cat(
            [x[bf_gma_mask.reshape(-1)], x[~bf_gma_mask.reshape(-1)][::sample, :]], dim=0
        )

        rgb = torch.cat(
            [ln_prop_tensor[bf_gma_mask], ln_prop_tensor[~bf_gma_mask][::sample, :]], dim=0
        )
        # Encoding input
        input = torch.cat(
            [
                coords,
                rgb,
            ],
            dim=1,
        )

        # gt = gt_ln_prop_tensor.view(-1, 3)
        gt = torch.cat(
            [gt_ln_prop_tensor[bf_gma_mask], gt_ln_prop_tensor[~bf_gma_mask][::sample, :]],
            dim=0,
        )
        return input, gt
