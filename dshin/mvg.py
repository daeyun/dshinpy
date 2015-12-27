import numpy as np


def hom(pts):
    assert pts.shape[1] in [2, 3]
    return np.hstack((pts, np.ones((pts.shape[0], 1))))


def hom_inv(pts):
    assert pts.shape[1] in [3, 4]
    return pts[:, :-1] / pts[:, -1, None]
