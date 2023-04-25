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
