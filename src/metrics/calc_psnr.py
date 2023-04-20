from src.utils.metric import psnr


def calc_psnr(source, target):
    return psnr(source, target)
