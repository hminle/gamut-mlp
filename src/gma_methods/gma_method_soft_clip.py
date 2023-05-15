import copy
from functools import partial

import colour
import numpy as np

from src.gma_methods.base_gma_method import BaseGMAMethod
from src.gma_methods.utils import (
    ILLUMINANT_D50,
    ILLUMINANT_D65,
    prophotorgb_colourspace,
    srgb_colourspace,
)
from src.utils.color import (
    decode_prop,
    decode_srgb,
    encode_srgb,
    prop_to_srgb_cat02,
    srgb_to_prop_cat02,
    to_single,
    to_uint8,
)


def rescale2(x, source_min=0.95, source_max=1.1, target_min=0.95, target_max=1.0):

    out = (x - source_min) / (source_max - source_min) * (target_max - target_min) + target_min
    return out


def base_expand(x, source_min=0.95, source_max=1.1, target_min=0.95, target_max=1.0):
    out = (x - target_min) / (target_max - target_min) * (source_max - source_min) + source_min
    return out


class GMAMethodSoftClip(BaseGMAMethod):
    def __init__(self, method_name: str):
        self.method_name = method_name

    def init_model_and_trainer(self):
        pass

    def _set_up(self, ln_srgb_gma):
        self.pos_upper_bound = ln_srgb_gma.max()
        self.pos_lower_bound = 0.9
        self.pos_upper_bound_255 = int(round(1.2 * 255))
        self.pos_lower_bound_255 = int(round(0.9 * 255))

        self.neg_upper_bound = 0.05
        self.neg_lower_bound = ln_srgb_gma.min()
        self.neg_upper_bound_255 = int(round(self.neg_upper_bound * 255))
        self.neg_lower_bound_255 = int(round(self.neg_lower_bound * 255))

    def gamut_reduction(self, ln_prop_img):
        ln_srgb_gma = prop_to_srgb_cat02(ln_prop_img)
        self._set_up(ln_srgb_gma)

        # Process Lower Part
        lower_mask_exl_black = (ln_srgb_gma <= self.neg_upper_bound).any(axis=2) & ~(
            ln_srgb_gma <= self.neg_upper_bound
        ).all(axis=2)
        ln_srgb_gma_lower_flatten = ln_srgb_gma[lower_mask_exl_black].flatten()
        lower_mask_1D = ln_srgb_gma_lower_flatten <= self.neg_upper_bound
        ln_srgb_gma_lower_flatten_1D = ln_srgb_gma_lower_flatten[lower_mask_1D]
        new_ln_srgb_gma_lower_flatten_1D = np.apply_along_axis(
            partial(
                rescale2,
                source_min=self.neg_lower_bound,
                source_max=self.neg_upper_bound,
                target_min=0.0,
                target_max=self.neg_upper_bound,
            ),
            0,
            ln_srgb_gma_lower_flatten_1D,
        )
        ln_srgb_gma_lower_flatten[lower_mask_1D] = new_ln_srgb_gma_lower_flatten_1D
        ln_srgb_gma[lower_mask_exl_black] = ln_srgb_gma_lower_flatten.reshape(
            ln_srgb_gma[lower_mask_exl_black].shape
        )

        # Process Upper Part
        upper_mask_exl_white = (ln_srgb_gma >= self.pos_lower_bound).any(axis=2) & ~(
            ln_srgb_gma >= self.pos_lower_bound
        ).all(axis=2)
        ln_srgb_gma_upper_flatten = ln_srgb_gma[upper_mask_exl_white].flatten()
        upper_mask_1D = ln_srgb_gma_upper_flatten >= self.pos_lower_bound
        ln_srgb_gma_upper_flatten_1D = ln_srgb_gma_upper_flatten[upper_mask_1D]
        new_ln_srgb_gma_upper_flatten_1D = np.apply_along_axis(
            partial(
                rescale2,
                source_min=self.pos_lower_bound,
                source_max=self.pos_upper_bound,
                target_min=self.pos_lower_bound,
                target_max=1.0,
            ),
            0,
            ln_srgb_gma_upper_flatten_1D,
        )
        ln_srgb_gma_upper_flatten[upper_mask_1D] = new_ln_srgb_gma_upper_flatten_1D
        ln_srgb_gma[upper_mask_exl_white] = ln_srgb_gma_upper_flatten.reshape(
            ln_srgb_gma[upper_mask_exl_white].shape
        )

        return np.clip(ln_srgb_gma, 0, 1)

    def gamut_expansion(self, ln_srgb_img):
        srgb_gma = ln_srgb_img
        prop_gma = np.zeros(srgb_gma.shape)
        prop_gma[:] = srgb_gma[:]
        # Process Lower Part
        lower_mask_exl_black = (srgb_gma <= self.neg_upper_bound).any(axis=2) & ~(
            srgb_gma <= self.neg_upper_bound
        ).all(axis=2)
        srgb_gma_lower_flatten = srgb_gma[lower_mask_exl_black].flatten()

        lower_mask_1D = srgb_gma_lower_flatten <= self.neg_upper_bound
        srgb_gma_lower_flatten_1D = srgb_gma_lower_flatten[lower_mask_1D]

        expand_srgb_gma_lower_flatten_1D = np.apply_along_axis(
            partial(
                base_expand,
                source_min=self.neg_lower_bound,
                source_max=self.neg_upper_bound,
                target_min=0.0,
                target_max=self.neg_upper_bound,
            ),
            0,
            srgb_gma_lower_flatten_1D,
        )
        srgb_gma_lower_flatten[lower_mask_1D] = expand_srgb_gma_lower_flatten_1D
        srgb_gma[lower_mask_exl_black] = srgb_gma_lower_flatten.reshape(
            srgb_gma[lower_mask_exl_black].shape
        )

        # Process Upper Part
        upper_mask_exl_white = (srgb_gma >= self.pos_lower_bound).any(axis=2) & ~(
            srgb_gma >= self.pos_lower_bound
        ).all(axis=2)
        srgb_gma_upper_flatten = srgb_gma[upper_mask_exl_white].flatten()
        upper_mask_1D = srgb_gma_upper_flatten >= self.pos_lower_bound

        srgb_gma_upper_flatten_1D = srgb_gma_upper_flatten[upper_mask_1D]
        expand_srgb_gma_upper_flatten_1D = np.apply_along_axis(
            partial(
                base_expand,
                source_min=self.pos_lower_bound,
                source_max=self.pos_upper_bound,
                target_min=self.pos_lower_bound,
                target_max=1.0,
            ),
            0,
            srgb_gma_upper_flatten_1D,
        )
        srgb_gma_upper_flatten[upper_mask_1D] = expand_srgb_gma_upper_flatten_1D
        srgb_gma[upper_mask_exl_white] = srgb_gma_upper_flatten.reshape(
            srgb_gma[upper_mask_exl_white].shape
        )
        prop_gma = srgb_to_prop_cat02(srgb_gma)
        return prop_gma

    def __str__(self):
        return self.method_name
