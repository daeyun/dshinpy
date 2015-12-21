import numpy as np
import matplotlib.pyplot as pt
from mpl_toolkits.mplot3d import art3d
# from mpl_toolkits import mplot3d


def draw_edge_3d(lines, ax=None, colors=None):
    lines = np.array(lines, dtype=np.float)
    lc = art3d.Line3DCollection(lines, linewidths=2, colors=colors)
    if ax is None:
        fig = pt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(lc)

    bmax = (lines.max(axis=0).max(axis=0))
    bmin = (lines.min(axis=0).min(axis=0))
    padding = np.abs((bmax - bmin) / 2.0).max()

    bmin = (bmax + bmin) / 2.0 - padding
    bmax = (bmax + bmin) / 2.0 + padding

    ax.set_xlim([bmin[0], bmax[0]])
    ax.set_ylim([bmin[1], bmax[1]])
    ax.set_zlim([bmin[2], bmax[2]])
    ax.set_aspect('equal')

    return ax
