import logging

def get_logger(name=__name__) -> logging.Logger:
    logger = logging.getLogger(name)

    return logger
