# -*- coding: utf-8 -*-
import numpy as np


def saturated_mask_generator(pw_mask):
    for mask_value in np.unique(pw_mask):
        if mask_value != 13 and mask_value != 0 and mask_value != 26:
            mask = pw_mask == mask_value
            yield mask, mask_value


def is_black(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return (x == 0).all(axis=2)
    elif x.dtype in (np.float16, np.float32, np.float64):
        return (x == 0.0).all(axis=2)
    else:
        raise Exception(f"Wrong numpy array type. Expect float or uint8 but got {x.dtype}")


def is_white(x: np.ndarray) -> np.ndarray:
    if x.dtype == np.uint8:
        return (x == 255).all(axis=2)
    elif x.dtype in (np.float16, np.float32, np.float64):
        return (x == 1.0).all(axis=2)
    else:
        raise Exception(f"Wrong numpy array type. Expect float or uint8 but got {x.dtype}")


def is_inner(x: np.ndarray) -> np.ndarray:
    return (x != 0) & (x != 255)


def compute_masks(input_img):
    m_black = is_black(input_img)
    m_white = is_white(input_img)
    m_inner = is_inner(input_img)
    m_inner_all: np.ndarray = m_inner.all(axis=2)  # all the R, G, and B are inner values
    o2o_mask: np.ndarray = m_black | m_white | m_inner_all  # either black, white, or inner_all
    m2o_mask: np.ndarray = ~o2o_mask  # neither black, white, nor inner_all
    return o2o_mask, m2o_mask, m_inner
