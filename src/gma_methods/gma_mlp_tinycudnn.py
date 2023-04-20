import os
import time
from typing import List

import numpy as np
import pytorch_lightning as pl
import tinycudann as tcnn
import torch
from hydra.utils import instantiate
from pytorch_lightning.callbacks.early_stopping import EarlyStopping
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split
from torchvision.transforms import Compose, Normalize
from tqdm import tqdm

from src.gma_methods.base_gma_method import BaseGMAMethod
from src.manual_trainers.gma_trainer import tinycudnn_training_loop
from src.utils.color import (
    decode_prop,
    decode_srgb,
    encode_srgb,
    prop_to_srgb_cat02,
    srgb_to_prop_cat02,
    to_single,
    to_uint8,
)
from src.utils.image_func import get_wh_mgrid
from src.utils.input_tensor_tinycudnn import InputTensor
from src.utils.mask import compute_masks

torch.manual_seed(202207)


class GMAMLPTinyCudnn(BaseGMAMethod):
    def __init__(
        self,
        method_name: str,
        sample: int = 10,
        og_sample: int = 1,
        gpus=[0],
        n_frequencies: int = 12,
        n_neurons: int = 64,
        n_hidden_layers: int = 2,
        activation: str = "ReLU",
        output_activation: str = "None",
        max_epochs: int = 10,
        lr: float = 1e-4,
        loss_func: str = "mse_loss",  # l1_loss, mse_loss
        checkpoint_path: str = None,
        is_trained: bool = False,
        retrain: bool = False,
        pretrained_model: str = "",
        precision: int = 32,
        n_steps: int = 10000,
        using_jit: bool = True,
        expand_og_only: bool = False,
        need_encoding: bool = True,
    ):
        super(GMAMLPTinyCudnn, self).__init__()
        self.method_name = method_name
        self.sample = sample
        self.og_sample = og_sample

        self.max_epochs = max_epochs
        self.lr = lr
        self.loss_func = loss_func
        self.is_trained = is_trained
        self.retrain = retrain
        self.pretrained_model = pretrained_model
        self.checkpoint_path = checkpoint_path
        self.precision = precision
        self.n_steps = n_steps
        self.using_jit = using_jit
        self.expand_og_only = expand_og_only
        self.need_encoding = need_encoding
        self.network_config = {
            "encoding": {"otype": "Frequency", "n_frequencies": n_frequencies},
            "network": {
                "otype": "FullyFusedMLP",
                "activation": activation,
                "output_activation": output_activation,
                "n_neurons": n_neurons,
                "n_hidden_layers": n_hidden_layers,
            },
        }
        gpu = str(gpus) if (type(gpus) == int) else str(gpus[0])
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
        self.device = torch.device("cuda")

    def init_model_and_trainer(self):
        # MLP Tiny does not have Trainer
        if self.need_encoding:
            self.logger.info("Create model with encoding")
            self.model = tcnn.NetworkWithInputEncoding(
                n_input_dims=5,
                n_output_dims=3,
                encoding_config=self.network_config["encoding"],
                network_config=self.network_config["network"],
            )
        else:
            self.logger.info("Create model without encoding")
            self.model = tcnn.Network(
                5,
                3,
                self.network_config["network"],
            )

        if self.is_trained and os.path.isfile(self.pretrained_model):
            self.logger.info(f"Load model: {self.pretrained_model}")
            self.model.load_state_dict(torch.load(self.pretrained_model))

    def gamut_reduction(self, gt_ln_prop_img):
        sample = self.sample
        og_sample = self.og_sample
        ln_srgb = prop_to_srgb_cat02(gt_ln_prop_img)

        if self.retrain:
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
                [
                    x[bf_gma_mask.reshape(-1)][::og_sample, :],
                    x[~bf_gma_mask.reshape(-1)][::sample, :],
                ],
                dim=0,
            )

            rgb = torch.cat(
                [
                    ln_prop_tensor[bf_gma_mask][::og_sample, :],
                    ln_prop_tensor[~bf_gma_mask][::sample, :],
                ],
                dim=0,
            )
            # Encoding input
            input = torch.cat(
                [
                    coords,
                    rgb,
                ],
                dim=1,
            )

            gt = torch.cat(
                [
                    gt_ln_prop_tensor[bf_gma_mask][::og_sample, :],
                    gt_ln_prop_tensor[~bf_gma_mask][::sample, :],
                ],
                dim=0,
            )

            self.logger.info(f"Target dim: {gt.shape}")

            inputTensor = InputTensor(input, gt, self.device)

            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)

            if self.checkpoint_path:
                checkpoint_full_path = (
                    f"{self.checkpoint_path}/{self.image_idx}_{self.image_name}.pt"
                )
            else:
                checkpoint_full_path = None

            self.logger.info(f"Train MLP Tiny with {self.n_steps} steps")

            self.model.train()
            start_time = time.time()
            tinycudnn_training_loop(
                n_steps=self.n_steps,
                batch_size=2 ** 16,
                device=self.device,
                inputTensor=inputTensor,
                optimizer=optimizer,
                model=self.model,
                checkpoint_path=checkpoint_full_path,
                using_jit=self.using_jit,
            )
            training_time = time.time() - start_time  # seconds

            return np.clip(ln_srgb, 0, 1), training_time
        return np.clip(ln_srgb, 0, 1), 0

    def gamut_expansion(self, ln_srgb_img, return_inference_time=False):
        srgb = ln_srgb_img
        ln_prop = srgb_to_prop_cat02(srgb)

        ln_prop_tensor = torch.from_numpy(ln_prop)
        # ln_prop_tensor = torch.from_numpy(ln_prop)
        transform = Compose([Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        ln_prop_tensor = ln_prop_tensor.permute(2, 0, 1)
        ln_prop_tensor = transform(ln_prop_tensor)
        ln_prop_tensor = ln_prop_tensor.permute(1, 2, 0)
        x = get_wh_mgrid(ln_prop_tensor.shape[0], ln_prop_tensor.shape[1])
        x = x.float()

        # Encoding input
        coords = x.view(-1, 2)
        clipped_prop = ln_prop_tensor.view(-1, 3)

        input = torch.cat(
            [
                coords,
                clipped_prop,
            ],
            dim=1,
        )
        self.model.eval()
        start_time = time.time()
        with torch.no_grad():
            ln_prop_out = (self.model(input).detach().cpu() + clipped_prop).view(
                ln_prop_tensor.shape
            )
        inference_time = time.time() - start_time  # seconds
        # un-normalized
        MEAN = torch.tensor([0.5, 0.5, 0.5])
        STD = torch.tensor([0.5, 0.5, 0.5])

        ln_prop_out = ln_prop_out * STD[None, None, :] + MEAN[None, None, :]
        ln_prop_out = ln_prop_out.numpy()
        prop_gma = np.clip(ln_prop_out, 0, 1)
        if self.expand_og_only:
            o2o_mask, m2o_mask, m_inner = compute_masks(srgb)
            prop_gma[o2o_mask] = ln_prop[o2o_mask]

        if return_inference_time:
            return prop_gma, inference_time
        return prop_gma

    def __str__(self):
        return self.method_name
