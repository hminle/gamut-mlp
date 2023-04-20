import time

import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf

from src.pipelines.base_pipeline import BasePipeline

class ExperimentManager:
    def __init__(self, config: DictConfig):
        self.config = config

    def init_experiment(self):

        dataset = instantiate(self.config.get("dataset"))
        method = instantiate(self.config.get("method"))
        reporter = instantiate(self.config.get("reporter"))

        self.pipeline: BasePipeline = instantiate(self.config.get("pipeline"))
        if dataset is not None:
            self.pipeline.set_dataset(dataset)
        if method is not None:
            self.pipeline.set_method(method, self.config)
        if reporter is not None:
            self.pipeline.set_reporter(reporter)

    def start_experiment(self):
        self.pipeline.run(self.config)

    def finish_experiment(self):
        self.pipeline.stop()
