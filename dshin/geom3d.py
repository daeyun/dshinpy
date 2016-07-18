import itertools
import tempfile
import os
import numpy as np
import matplotlib.pyplot as pt
import scipy.linalg as la
from mpl_toolkits.mplot3d import art3d
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from mpl_toolkits.mplot3d import proj3d


def edge_3d(lines, ax=None, colors=None, lim=None, linewidths=2):
    lines = np.array(lines, dtype=np.float)
    lc = art3d.Line3DCollection(lines, linewidths=linewidths, colors=colors)
    if ax is None:
        fig = pt.figure()
        ax = fig.add_subplot(111, projection='3d')
    ax.add_collection3d(lc)

    if lim is None:
        bmax = (lines.max(axis=0).max(axis=0))
        bmin = (lines.min(axis=0).min(axis=0))
        padding = np.abs((bmax - bmin) / 2.0).max()

        bmin = (bmax + bmin) / 2.0 - padding
        bmax = (bmax + bmin) / 2.0 + padding

    else:
        bmin = lim.ravel()[:3]
        bmax = lim.ravel()[3, :6]

    ax.set_xlim([bmin[0], bmax[0]])
    ax.set_ylim([bmin[1], bmax[1]])
    ax.set_zlim([bmin[2], bmax[2]])
    ax.set_aspect('equal')

    return ax


def pts(pts, ax=None, color='blue', markersize=5, lim=None, reset_limits=True, cam_sph=None, colorbar=False, cmap=None, is_radians=False):
    if ax is None:
        fig = pt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        ax = fig.gca(projection='3d')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_aspect('equal')
    if cam_sph is not None:
        if is_radians:
            cam_sph = cam_sph / np.pi * 180
        ax.view_init(elev=90 - cam_sph[1], azim=cam_sph[2])

    if type(lim) == list:
        lim = np.array(lim)

    p = ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], marker='.', linewidth=0,
                   c=color, s=markersize, cmap=cmap)
    if colorbar:
        pt.gcf().colorbar(p)

    if lim is None:
        bmax = (pts.max(axis=0))
        bmin = (pts.min(axis=0))

        padding = np.abs((bmax - bmin) / 2.0).max()
        bmin = (bmax + bmin) / 2.0 - padding
        bmax = (bmax + bmin) / 2.0 + padding
    else:
        bmin = lim.ravel()[:3]
        bmax = lim.ravel()[3:6]

    if reset_limits:
        ax.set_xlim([bmin[0], bmax[0]])
        ax.set_ylim([bmin[1], bmax[1]])
        ax.set_zlim([bmin[2], bmax[2]])
    ax.set_aspect('equal')

    return ax


def sphere(center_xyz=(0, 0, 0), radius=1, ax=None, color='red', alpha=1,
           linewidth=1):
    if ax is None:
        fig = pt.figure()
        ax = fig.gca(projection='3d')

    ax.set_aspect('equal')

    u, v = np.mgrid[0:2 * np.pi:20j, 0:np.pi:10j]
    x = np.cos(u) * np.sin(v)
    y = np.sin(u) * np.sin(v)
    z = np.cos(v)

    x *= radius
    y *= radius
    z *= radius

    x += center_xyz[0]
    y += center_xyz[1]
    z += center_xyz[2]

    ax.plot_wireframe(x, y, z, color=color, linewidth=linewidth, alpha=alpha)


def cube(center_xyz=(0, 0, 0), radius=1, ax=None, color='blue', alpha=1,
         linewidth=1):
    if ax is None:
        fig = pt.figure()
        ax = fig.gca(projection='3d')

    ax.set_aspect('equal')

    r = [-radius, radius]
    pts = np.array([[s, e] for s, e in itertools.combinations(
        np.array(list(itertools.product(r, r, r))), 2) if
                    np.sum(np.abs(s - e)) == r[1] - r[0]])
    pts += center_xyz

    for s, e, in pts:
        ax.plot3D(*zip(s, e), color=color, alpha=alpha, linewidth=linewidth)


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


def draw_one_arrow(xs, ys, zs, ax, color='red', linewidth=1, tip_size=10,
                   text=None):
    a = Arrow3D(xs, ys, zs, mutation_scale=tip_size, lw=linewidth,
                arrowstyle="-|>", color=color)
    ax.add_artist(a)

    if text is not None:
        pos = np.array([[0.2, 0.8]])
        text_x = pos.dot(xs[:, None])[0][0]
        text_y = pos.dot(ys[:, None])[0][0]
        text_z = pos.dot(zs[:, None])[0][0]
        ax.text(text_x, text_y, text_z, text, color='black')


def draw_arrow_3d(start_pts, end_pts, ax=None, colors='red', texts=None):
    if ax is None:
        fig = pt.figure()
        ax = fig.gca(projection='3d')
    xs = np.hstack((start_pts[:, 0, None], end_pts[:, 0, None]))
    ys = np.hstack((start_pts[:, 1, None], end_pts[:, 1, None]))
    zs = np.hstack((start_pts[:, 2, None], end_pts[:, 2, None]))
    for i in range(xs.shape[0]):
        color = colors[i] if isinstance(colors, list) or isinstance(colors,
                                                                    np.ndarray) else colors
        text = texts[i] if isinstance(texts, list) else None
        draw_one_arrow(xs[i, :], ys[i, :], zs[i, :], ax, color=color, text=text)


def draw_camera(Rt, ax=None, scale=10):
    """
    :param Rt: (3,4)
    """
    if ax is None:
        fig = pt.figure()
        ax = fig.gca(projection='3d')
    cam_xyz = -la.inv(Rt[:, :3]).dot(Rt[:, 3])

    R = Rt[:, :3]

    arrow_start = np.tile(cam_xyz, [3, 1])
    arrow_end = scale * R + cam_xyz

    draw_arrow_3d(arrow_start, arrow_end, ax, colors=['red', 'blue', 'green'],
                  texts=['x', 'y', 'z'])

    pts(cam_xyz[None, :], ax=ax, markersize=0)
    pts(arrow_end, ax=ax, markersize=0)

    pt.draw()

    return ax


def draw_cameras(cameras, ax):
    for camera in cameras:
        draw_camera(np.hstack((camera.s * camera.R, camera.t)), ax=ax, scale=0.1)


def plot_mesh(mesh, ax=None):
    if ax is None:
        fig = pt.figure()
        # ax = fig.add_subplot(111, projection='3d')
        ax = fig.gca(projection='3d')

    verts = mesh['v']
    faces = mesh['f']

    poly3d = Poly3DCollection(verts[faces])
    poly3d.set_alpha(0.8)
    poly3d.set_facecolor('blue')
    poly3d.set_edgecolor('gray')
    poly3d.set_linewidth(0.1)

    ax.add_collection3d(poly3d)

    bmax = verts.max(axis=0)
    bmin = verts.min(axis=0)
    padding = (bmax - bmin) / 10
    bmax += padding
    bmin -= padding

    ax.set_xlim(bmin[0], bmax[0])
    ax.set_ylim(bmin[1], bmax[1])
    ax.set_zlim(bmin[2], bmax[2])
    ax.set_aspect('equal')

    return ax


def set_lims(ax, bb):
    bmax = bb.max(0)
    bmin = bb.min(0)

    ax.set_xlim(bmin[0], bmax[0])
    ax.set_ylim(bmin[1], bmax[1])
    ax.set_zlim(bmin[2], bmax[2])

    pts(np.array([bmax, bmin]), ax=ax, markersize=0)


def open_in_meshlab(verts, faces):
    with tempfile.NamedTemporaryFile(prefix='mesh_', suffix='.off',
                                     delete=False) as fp:
        fp.write('OFF\n{} {} 0\n'.format(verts.shape[0], faces.shape[0]).encode(
            'utf-8'))
        np.savetxt(fp, verts, fmt='%.5f')
        np.savetxt(fp, np.hstack((3 * np.ones((faces.shape[0], 1)), faces)),
                   fmt='%d')
        fname = fp.name
    os.system('while [ ! -f {fname} ]; do sleep 0.5; done; meshlab {fname}'.format(fname=fname))
