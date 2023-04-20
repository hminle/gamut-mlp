import math
import pathlib
import time
from pathlib import Path
from typing import Union

import cv2
import numpy as np
import torch

from src.utils.color import (
    decode_prop,
    decode_srgb,
    encode_srgb,
    prop_to_srgb_cat02,
    srgb_to_prop_cat02,
    to_single,
    to_uint8,
)


def crop_image(img: np.ndarray, crop_size: int = 512):
    h, w, _ = img.shape
    ch, cw = (h - crop_size) // 2, (w - crop_size) // 2
    out_img = img[ch : ch + crop_size, cw : cw + crop_size]
    return out_img


def get_wh_mgrid(width: int, height: int, dim: int = 2) -> torch.Tensor:
    w = torch.linspace(-1, 1, steps=width)
    h = torch.linspace(-1, 1, steps=height)
    mgrid = torch.stack(torch.meshgrid(w, h), dim=-1)
    return mgrid.reshape(-1, dim)


def read_image(path: Union[str, Path]) -> np.ndarray:
    img = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img


def write_image(img: np.ndarray, path: Union[str, Path]) -> None:
    path = str(path)
    if img.dtype == np.uint16:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.uint16))
    elif img.dtype == np.uint8:
        cv2.imwrite(path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR).astype(np.uint8))
    else:
        raise ValueError("unsupport image type, should be np.uint8 or np.uint16")


# channel is 0, 1, or 2
def compute_histogram(A, channel, bins, scale=-1):
    # create histogram
    hist = [0] * (bins)
    for row in range(0, 512):
        for col in range(0, 512):
            index = A[row][col][channel]
            assert index <= bins, "Tone value larger than maximum bin"
            hist[index] = hist[index] + 1
    sum = 0
    for x in hist:
        if x > 0:
            sum += 1
    print(f"total number of unique tones {sum}")
    if scale != -1:
        for i in range(0, len(hist)):
            hist[i] = hist[i] // scale
    return hist


def compute_hist_cdf(hist):
    cdf = [0] * len(hist)
    for idx, value in enumerate(hist):
        if idx - 1 >= 0:
            cdf[idx] = cdf[idx - 1] + hist[idx]
        else:
            cdf[idx] = hist[idx]

    cdf = np.array(cdf)
    cdf = cdf / cdf.max()
    return cdf


def gma_pipeline(gma_method, prop_img):
    gma_method.init_model_and_trainer()
    ln_prop_img = decode_prop(to_single(prop_img))
    ln_srgb = prop_to_srgb_cat02(ln_prop_img)
    bf_gma_mask = (
        ((ln_srgb < 0) | (ln_srgb > 1)).any(axis=2)
        & ~(ln_srgb < 0).all(axis=2)
        & ~(ln_srgb > 1).all(axis=2)
    )

    # Reduction
    start_time = time.time()
    srgb_gma = to_uint8(encode_srgb(gma_method.gamut_reduction(ln_prop_img)))
    training_time = round((time.time() - start_time) / 60, 2)
    # Expansion
    ln_srgb_gma = decode_srgb(to_single(srgb_gma))
    prop_gma = gma_method.gamut_expansion(ln_srgb_gma)
    return prop_gma


def get_pad_width(shape: tuple[int, int, int]) -> list[tuple[int, int]]:
    # use this pad_width to pad a numpy image: img_padded = np.pad(img, pad_width=get_pad_width(img.shape), mode='constant')
    max_length = math.ceil(max(shape) / 512) * 512
    pad_width = []
    delta_0 = max_length - shape[0]
    pad_width.append((int(round(delta_0 / 2)), int(delta_0 - round(delta_0 / 2))))
    delta_1 = max_length - shape[1]
    pad_width.append((int(round(delta_1 / 2)), int(delta_1 - round(delta_1 / 2))))
    pad_width.append((0, 0))  # Do not pad to channel, only pad to height and witdh
    return pad_width


def pad_img(img: np.ndarray, pad_width: list[tuple[int, int]]) -> np.ndarray:
    img_padded = np.pad(img, pad_width=pad_width, mode="constant")
    return img_padded


def unpad_img(
    padded_img: np.ndarray,
    pad_width: list[tuple[int, int]],
) -> np.ndarray:
    height_pad, width_pad, _ = pad_width
    height, width, _ = padded_img.shape
    return padded_img[
        height_pad[0] : height - height_pad[1], width_pad[0] : width - width_pad[1], :
    ]
