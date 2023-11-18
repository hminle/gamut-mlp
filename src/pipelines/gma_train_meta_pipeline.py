import copy
import os
import sys
import time
from datetime import timedelta
from typing import List, Optional

import hydra
import numpy as np
import tinycudann as tcnn
import torch
from imageio import imread
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from src.gma_methods.base_gma_method import BaseGMAMethod
from src.manual_trainers.gma_trainer import tinycudnn_training_loop
from src.metrics import calc_deltaE, calc_mae, calc_psnr, calc_rmse
from src.pipelines.base_pipeline import BasePipeline
from src.utils import utils
from src.utils.color import (
    decode_prop,
    decode_srgb,
    encode_srgb,
    prop_to_srgb_cat02,
    to_single,
    to_uint8,
)
from src.utils.input_tensor import InputTensor


class GMATrainMetaPipeline(BasePipeline):
    def __init__(
        self,
        pipeline_name: str,
        activation: str = "ReLU",
        output_activation: str = "None",
        n_frequencies: int = 12,
        n_neurons: int = 64,
        n_hidden_layers: int = 2,
        lr: float = 1e-3,
        meta_epoch: int = 10,
        meta_batch_size: int = 3,
        meta_inner_lr: float = 0.01,
        meta_inner_steps: int = 10000,
        checkpoint_meta_path: str = "",
        precision: int = 32,
        gpus=[0],
    ) -> None:
        super(GMATrainMetaPipeline, self).__init__(pipeline_name)

        self.activation = activation
        self.lr = lr
        self.gpus = gpus
        self.meta_epoch = meta_epoch
        self.meta_batch_size = meta_batch_size
        self.meta_inner_lr = meta_inner_lr
        self.meta_inner_steps = meta_inner_steps
        self.checkpoint_meta_path = checkpoint_meta_path
        self.precision = precision

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

    def set_dataset(self, dataset) -> None:
        self.logger.info(f"Set dataset {dataset.dataset_name} for pipeline")
        self.dataset = dataset

    def run(self, config: DictConfig) -> None:
        self.logger.info(f"Run Pipeline {self.pipeline_name}")
        self.model = tcnn.NetworkWithInputEncoding(
            n_input_dims=5,
            n_output_dims=3,
            encoding_config=self.network_config["encoding"],
            network_config=self.network_config["network"],
        )

        start_experiment = time.time()
        # Set seed for random number generators in pytorch, numpy and python.random
        if config.get("seed"):
            torch.manual_seed(config.seed)

        train_loader = DataLoader(self.dataset, batch_size=self.meta_batch_size, num_workers=8)
        self.model.to(self.device)
        meta_optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for epoch in range(self.meta_epoch):
            with tqdm(
                train_loader, desc=f"Epoch {epoch}", leave=epoch == self.meta_epoch - 1
            ) as tepoch:
                for input, gt in tepoch:
                    input, gt = input.to(self.device), gt.to(self.device)

                    meta_optim.zero_grad()
                    with torch.no_grad():
                        for meta_param in self.model.parameters():
                            meta_param.grad = 1.0 * meta_param

                    for i in range(len(input)):
                        inner_model = copy.deepcopy(self.model)
                        inner_optim = torch.optim.SGD(
                            inner_model.parameters(), lr=self.meta_inner_lr
                        )

                        # inner loop
                        inputTensor = InputTensor(input[i], gt[i], self.device)

                        tinycudnn_training_loop(
                            n_steps=self.meta_inner_steps,
                            batch_size=2 ** 16,
                            device=self.device,
                            inputTensor=inputTensor,
                            optimizer=inner_optim,
                            model=inner_model,
                            checkpoint_path=None,
                            using_jit=False,
                        )

                        with torch.no_grad():
                            for meta_param, inner_param in zip(
                                self.model.parameters(), inner_model.parameters()
                            ):
                                meta_param.grad -= 1.0 / len(input) * inner_param

                    meta_optim.step()

                    # tepoch.set_postfix(loss=loss.item())

        torch.save(self.model.state_dict(), self.checkpoint_meta_path)
        end_experiment = time.time()
        duration = end_experiment - start_experiment

        self.logger.info(
            f"---- FINISHED Pipeline {self.pipeline_name} in {timedelta(seconds=duration)} ----"
        )
        self.logger.info(f"Dataset: {self.dataset.dataset_name}")

    def stop(self):
        pass
