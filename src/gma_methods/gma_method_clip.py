import numpy as np

from src.gma_methods.base_gma_method import BaseGMAMethod
from src.utils.color import prop_to_srgb_cat02, srgb_to_prop_cat02


class GMAMethodClip(BaseGMAMethod):
    def __init__(self, method_name: str):
        self.method_name = method_name

    def init_model_and_trainer(self):
        pass

    def gamut_reduction(self, ln_prop_img):
        ln_srgb = prop_to_srgb_cat02(ln_prop_img)
        ln_srgb = np.clip(ln_srgb, 0, 1)
        return ln_srgb

    def gamut_expansion(self, ln_srgb_img, return_inference_time=False):
        prop_clip = srgb_to_prop_cat02(ln_srgb_img)

        if return_inference_time:
            return prop_clip, 0
        return prop_clip

    def __str__(self):
        return self.method_name
