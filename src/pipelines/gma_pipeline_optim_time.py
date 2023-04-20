import sys
import time
from typing import List, Optional

import hydra
import numpy as np
import torch
from imageio import imread
from omegaconf import DictConfig
from torch.utils.data import Dataset
from tqdm import tqdm

from src.gma_methods.base_gma_method import BaseGMAMethod
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


class GMAPipelineOptimTime(BasePipeline):
    def __init__(self, pipeline_name: str, start_idx: int = 0, stop_idx: int = -1):
        super(GMAPipelineOptimTime, self).__init__(pipeline_name)
        self.start_idx = start_idx
        self.stop_idx = stop_idx

    def set_dataset(self, dataset) -> None:
        self.logger.info(f"Set dataset {dataset.dataset_name} for pipeline")
        self.dataset = dataset

    def set_method(self, method, config: DictConfig) -> None:
        self.logger.info(f"Set method {method.method_name} for pipeline")
        self.method = method

    def set_reporter(self, reporter) -> None:
        self.logger.info("Set reporter for pipeline")
        self.reporter = reporter

    def run(self, config: DictConfig) -> None:
        start_experiment = time.time()
        # Set seed for random number generators in pytorch
        if config.get("seed"):
            torch.manual_seed(config.seed)

        if config.get("enable_db_reporting"):
            self.logger.info(f"Reporter set method name {self.method.method_name}")
            self.reporter.set_method_name(self.method.method_name)
            self.reporter.create_new_columns()

        for idx, data in enumerate(tqdm(self.dataset)):
            if idx == self.stop_idx:
                break
            if idx >= self.start_idx:
                prop_img, img_name = data
                self.logger.info(f"Process image {img_name} index {idx}")
                self.method.init_model_and_trainer()
                ln_prop_img = decode_prop(prop_img)
                ln_srgb = prop_to_srgb_cat02(ln_prop_img)
                bf_gma_mask = (
                    ((ln_srgb < 0) | (ln_srgb > 1)).any(axis=2)
                    & ~(ln_srgb < 0).all(axis=2)
                    & ~(ln_srgb > 1).all(axis=2)
                )
                self.method.set_image_info(idx, img_name)

                # Reduction
                method_output = self.method.gamut_reduction(ln_prop_img)
                if type(method_output) == tuple:
                    srgb_gma, training_time = method_output
                else:
                    srgb_gma = method_output
                    training_time = 0

                self.logger.info(f"TRAINING TIME: {training_time}")
                srgb_gma = to_uint8(encode_srgb(srgb_gma))

                # Expansion
                if "cic_2020" in self.method.method_name:
                    prop_gma = self.method.gamut_expansion(img_name)
                else:
                    ln_srgb_gma = decode_srgb(to_single(srgb_gma))
                    prop_gma = self.method.gamut_expansion(ln_srgb_gma)

                rmse = calc_rmse(prop_gma, ln_prop_img)
                psnr = calc_psnr(prop_gma.astype(np.float32), ln_prop_img.astype(np.float32))
                mae = calc_mae(prop_gma, ln_prop_img)
                deltaE2000 = calc_deltaE(prop_gma, ln_prop_img, color_space="ProPhotoRGB")
                rmse_oog = calc_rmse(prop_gma[bf_gma_mask], ln_prop_img[bf_gma_mask])
                psnr_oog = calc_psnr(
                    prop_gma[bf_gma_mask].astype(np.float32),
                    ln_prop_img[bf_gma_mask].astype(np.float32),
                )
                if config.get("enable_db_reporting"):
                    self.reporter.report_error(
                        image_filename=img_name,
                        rmse=rmse,
                        psnr=psnr,
                        mae=mae,
                        deltaE2000=deltaE2000,
                        rmse_oog=rmse_oog,
                        psnr_oog=psnr_oog,
                        training_time=training_time,
                    )

        end_experiment = time.time()
        duration = end_experiment - start_experiment
        duration = round(duration / 3600, 2)
        self.logger.info(f"---- FINISHED in {duration} hours ----")
        self.logger.info(f"Method: {self.method.method_name}")
        self.logger.info(f"Dataset: {self.dataset.dataset_name}")

    def stop(self):
        self.reporter.stop()
