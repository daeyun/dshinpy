import numpy as np
import scipy.linalg as la


def weighted_std(w, x):
    if not (len(x.shape) == 2 and x.shape[1] == 3):
        raise NotImplementedError()
    assert len(w.shape) == 1
    return np.sqrt((w * np.power(x, 2).sum(axis=1)).sum() / w.sum())


def weighted_mean(w, x):
    if not (len(x.shape) == 2 and x.shape[1] == 3):
        raise NotImplementedError()
    assert len(w.shape) == 1
    return (x * w[:, None]).sum(axis=0) / w.sum()
