"""Utilities to perform common functions that are not part of the algorithm."""
import logging
import sys
import numpy as np


def setup_log(path):
    """Set up logging.

    All later logging should be registered under imlearn, e.g., imlearn.dagger.
    Args:
        path: A string that specifies the log file path.
    Returns:
        An logger object.
    """
    logger = logging.getLogger('imlearn')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler(path)
    fh.setLevel(logging.INFO)
    formatter = logging.Formatter('[%(name)-20s] %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler(sys.stdout)
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger
