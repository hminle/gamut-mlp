from src.utils.metric import rmse


def calc_rmse(source, target, axis=None):
    return rmse(source, target, axis=axis)
