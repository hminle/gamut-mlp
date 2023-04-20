import numpy as np


def __error(a, b):
    return a.astype(np.float64) - b.astype(np.float64)


def __absolute_error(a, b):
    return np.abs(__error(a, b))


def __squared_error(a, b):
    return np.power(__error(a, b), 2.0)


def l2(a, b, axis=None):
    return np.sqrt(np.sum(__squared_error(a, b), axis=axis))


def mae(a, b, axis=None):
    return np.mean(
        __absolute_error(
            a,
            b,
        ),
        axis=axis,
    )


def mse(a, b, axis=None):
    return np.mean(__squared_error(a, b), axis=axis)


def rmse(a, b, axis=None):
    return np.sqrt(mse(a, b, axis))


def psnr(a, b, axis=None):
    if a.dtype != b.dtype:
        raise Exception(
            f"Wrong numpy array type. 2 arrays should have the same dtype: {a.dtype} vs {b.dtype}"
        )
    if a.dtype == np.uint8:
        max_value = 255
    elif a.dtype in (np.float16, np.float32, np.float64):
        max_value = 1
    else:
        raise Exception(f"Wrong numpy array type. Expect float or uint8 but got {a.dtype}")

    return 10 * np.log10(max_value ** 2 / mse(a, b, axis))


def psnr_from_rmse(rmse: float):
    return 20 * np.log10(1 / rmse)
