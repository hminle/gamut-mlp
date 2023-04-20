from abc import ABC, abstractmethod

from omegaconf import DictConfig
from torch.utils.data import Dataset

from src.utils import utils


class BasePipeline(ABC):
    def __init__(self, pipeline_name: str):
        self.logger = utils.get_logger(__name__)
        self.pipeline_name = pipeline_name

    @abstractmethod
    def run(self, config: DictConfig) -> None:
        raise NotImplementedError

    def set_dataset(self, dataset) -> None:
        pass

    def set_method(self, method, config: DictConfig) -> None:
        pass

    def set_reporter(self, reporter) -> None:
        pass
