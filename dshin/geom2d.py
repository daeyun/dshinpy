import matplotlib.pyplot as pt
import numpy as np
from matplotlib.patches import Polygon
import matplotlib.cm as cm


def draw_triangles(triangles, ax=None, facecolor='blue', alpha=1):
    """
    :param triangles: (n,3,2) 2d triangles.
    :return:
    """
    assert triangles.shape[1] == 3
    assert triangles.shape[2] == 2

    if ax is None:
        fig = pt.figure()
        ax = fig.gca()

    for i in range(triangles.shape[0]):
        pts = np.array(triangles[i, :, :])
        p = Polygon(pts, closed=True, facecolor=facecolor, alpha=alpha)
        ax.add_patch(p)

    ax.set_xlim([triangles[:, :, 0].min(), triangles[:, :, 0].max()])
    ax.set_ylim([triangles[:, :, 1].min(), triangles[:, :, 1].max()])


def pts(xy, ax=None, markersize=10):
    if ax is None:
        fig = pt.figure()
        ax = fig.gca()

    ax.scatter(xy[:, 0], xy[:, 1], marker='.', s=markersize)


def draw_depth(depth: np.ma.core.MaskedArray, ax=None, clim=None, nancolor='y'):
    g = cm.get_cmap('gray')
    g.set_bad(nancolor, 1.)

    if ax is None:
        fig = pt.figure()
        ax = fig.gca()

    ii = ax.imshow(depth, cmap=g, interpolation='nearest', aspect='equal')
    from mpl_toolkits.axes_grid1 import make_axes_locatable

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    # cb = ax.figure.colorbar(ii)
    cb = pt.colorbar(ii, cax=cax)
    if clim is not None:
        cb.set_clim(clim)
