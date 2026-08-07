import numpy as np


def is_abnormal(value):
    return np.logical_or(~np.isfinite(value),value<-1000)
