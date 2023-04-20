import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List


class BaseReporter:
    def set_method_name(self, method_name):
        pass

    def create_table(self) -> None:
        pass

    def report_error(
        self,
        image_filename: str,
        rmse: float,
        psnr: float,
        mae: float,
        deltaE2000: float,
        rmse_oog: float,
        psnr_oog: float,
        training_time: float = 0,
    ):
        pass

    def report_image(self, image_filename: str, oog_percent: float, idx: int):
        pass

    def stop(self, key, value):
        pass
