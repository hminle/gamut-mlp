from abc import ABC, abstractmethod

from src.utils import utils


class BaseGMAMethod(ABC):
    def __init__(self):
        self.logger = utils.get_logger(__name__)

    def init_positional_encoder(self, config):
        pass

    def init_rgb_encoder(self, config):
        pass

    def set_image_info(self, idx, image_name):
        self.image_idx = idx
        self.image_name = image_name

    @abstractmethod
    def init_model_and_trainer(self):
        raise NotImplementedError

    @abstractmethod
    def gamut_reduction(self, ln_prop_img):
        raise NotImplementedError

    @abstractmethod
    def gamut_expansion(self, ln_srgb_img):
        raise NotImplementedError

    @abstractmethod
    def __str__(self):
        raise NotImplementedError
