import time

import hydra
from omegaconf import DictConfig, OmegaConf

from src.experiment_manager import ExperimentManager


@hydra.main(config_path="configs/", config_name="config.yaml")
def main(config: DictConfig):

    if config.get("print_config"):
        print(OmegaConf.to_yaml(config))

    experiment_manager = ExperimentManager(config)
    experiment_manager.init_experiment()
    experiment_manager.start_experiment()
    experiment_manager.finish_experiment()


if __name__ == "__main__":
    main()
