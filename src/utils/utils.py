import logging
import warnings
from typing import List, Sequence


def get_logger(name=__name__) -> logging.Logger:
    """Initializes multi-GPU-friendly python command line logger."""

    logger = logging.getLogger(name)

    # this ensures all logging levels get marked with the rank zero decorator
    # otherwise logs would get multiplied for each GPU process in multi-GPU setup
    for level in (
        "debug",
        "info",
        "warning",
        "error",
        "exception",
        "fatal",
        "critical",
    ):
        setattr(logger, level, rank_zero_only(getattr(logger, level)))

    return logger