from pathlib import Path

import cv2
import numpy as np
import torch
from imageio import imread
from torch.utils.data import Dataset

from src.utils.color import (
    decode_prop,
    decode_srgb,
    encode_srgb,
    prop_to_srgb_cat02,
    srgb_to_prop_cat02,
    to_single,
    to_uint8,
)
from src.utils.image_func import crop_image, read_image


class ImgDataset(Dataset):
    def __init__(
        self,
        data_root: str,
        dataset_name: str,
        img_loader: str = "cv2",  # or imageio
        img_extension: str = "png",
    ):
        self.image_root = Path(data_root) / dataset_name
        self.image_paths = list(self.image_root.glob(f"*.{img_extension}"))
        self.dataset_name = dataset_name
        if img_loader == "cv2":
            self.loader = read_image
        elif img_loader == "imageio":
            self.loader = imread

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img_name = img_path.name
        img = self.loader(img_path)
        img = to_single(img)
        return img, img_name

    def __str__(self):
        return self.gma_dataset_name


class CroppedImgDataset(ImgDataset):
    def __init__(
        self, data_root: str, dataset_name: str, crop_size: int, img_extension: str = "png"
    ):
        super(CroppedImgDataset, self).__init__(data_root, dataset_name)
        self.crop_size = crop_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx].name
        img = imread(self.image_paths[idx])
        img = crop_image(img, crop_size=self.crop_size)
        img = to_single(img)
        return img, img_name

    def __str__(self):
        return self.gma_dataset_name


class CroppedImg16bDataset(ImgDataset):
    def __init__(
        self, data_root: str, dataset_name: str, crop_size: int, img_extension: str = "png"
    ):
        super(CroppedImg16bDataset, self).__init__(data_root, dataset_name)
        self.crop_size = crop_size

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx].name
        img_path = self.image_paths[idx]
        img = cv2.imread(str(img_path), cv2.IMREAD_UNCHANGED)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = crop_image(img, crop_size=self.crop_size)
        img = to_single(img)
        return img, img_name

    def __str__(self):
        return self.gma_dataset_name


class CV2ImgDataset(ImgDataset):
    def __init__(self, data_root: str, dataset_name: str, img_extension: str = "png"):
        super(CV2ImgDataset, self).__init__(data_root, dataset_name)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_name = self.image_paths[idx].name
        img_path = self.image_paths[idx]
        img = read_image(img_path)
        img = to_single(img)
        return img, img_name

    def __str__(self):
        return self.gma_dataset_name


class GMADatasetMLP(ImgDataset):
    def __init__(self, data_root: str, gma_dataset_name: str, crop_size: int):
        super(GMADatasetMLP, self).__init__(data_root, gma_dataset_name)
        self.crop_size = crop_size

    def __getitem__(self, idx):
        prop_img = imread(self.image_paths[idx])
        h, w, _ = prop_img.shape
        ch, cw = (h - self.crop_size) // 2, (w - self.crop_size) // 2
        prop_img = prop_img[ch : ch + self.crop_size, cw : cw + self.crop_size]
        gt_ln_prop_img = decode_prop(to_single(prop_img))

        ln_srgb = prop_to_srgb_cat02(gt_ln_prop_img)
        srgb = decode_srgb(to_single(to_uint8(encode_srgb(np.clip(ln_srgb, 0, 1)))))
        ln_prop = srgb_to_prop_cat02(srgb)

        ln_prop_tensor = torch.from_numpy(ln_prop)
        gt_ln_prop_tensor = torch.from_numpy(gt_ln_prop_img)

        g0, g1 = torch.meshgrid(
            [
                torch.arange(-1, 1, step=2 / gt_ln_prop_tensor.shape[0]),
                torch.arange(-1, 1, step=2 / gt_ln_prop_tensor.shape[1]),
            ]
        )
        x = torch.cat([g0.flatten().unsqueeze(1), g1.flatten().unsqueeze(1)], dim=1)
        x = x.float()

        input = torch.cat([x, ln_prop_tensor.view(-1, 3)], dim=1)
        return input, gt_ln_prop_tensor.view(-1, 3)


class GMADatasetMLPv2(ImgDataset):
    def __init__(self, data_root: str, gma_dataset_name: str):
        super(GMADatasetMLPv2, self).__init__(data_root, gma_dataset_name)

    def __getitem__(self, idx):
        prop_img = imread(self.image_paths[idx])
        gt_ln_prop_img = decode_prop(to_single(prop_img))

        ln_srgb = prop_to_srgb_cat02(gt_ln_prop_img)
        srgb = decode_srgb(to_single(to_uint8(encode_srgb(np.clip(ln_srgb, 0, 1)))))
        ln_prop = srgb_to_prop_cat02(srgb)

        ln_prop_tensor = torch.from_numpy(ln_prop)
        gt_ln_prop_tensor = torch.from_numpy(gt_ln_prop_img)

        g0, g1 = torch.meshgrid(
            [
                torch.arange(-1, 1, step=2 / gt_ln_prop_tensor.shape[0]),
                torch.arange(-1, 1, step=2 / gt_ln_prop_tensor.shape[1]),
            ]
        )
        x = torch.cat([g0.flatten().unsqueeze(1), g1.flatten().unsqueeze(1)], dim=1)
        x = x.float()

        input = torch.cat([x, ln_prop_tensor.view(-1, 3)], dim=1)
        return input, gt_ln_prop_tensor.view(-1, 3)


class GMADatasetMLPRGB(ImgDataset):
    def __init__(self, data_root: str, gma_dataset_name: str, crop_size: int):
        super(GMADatasetMLPRGB, self).__init__(data_root, gma_dataset_name)
        self.crop_size = crop_size

    def __getitem__(self, idx):
        prop_img = imread(self.image_paths[idx])
        h, w, _ = prop_img.shape
        ch, cw = (h - self.crop_size) // 2, (w - self.crop_size) // 2
        prop_img = prop_img[ch : ch + self.crop_size, cw : cw + self.crop_size]
        gt_ln_prop_img = decode_prop(to_single(prop_img))

        ln_srgb = prop_to_srgb_cat02(gt_ln_prop_img)
        srgb = decode_srgb(to_single(to_uint8(encode_srgb(np.clip(ln_srgb, 0, 1)))))
        ln_prop = srgb_to_prop_cat02(srgb)

        ln_prop_tensor = torch.from_numpy(ln_prop)
        gt_ln_prop_tensor = torch.from_numpy(gt_ln_prop_img)

        input = ln_prop_tensor.view(-1, 3)
        return input, gt_ln_prop_tensor.view(-1, 3)
