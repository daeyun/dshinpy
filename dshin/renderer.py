import numpy as np
import time


def render_depth(vertices, P, rasterizer_func, out_wh=(512, 512), lrtb=(0, 511, 0, 511),
                 is_perspective=True, flip_z=True, near=None,
                 far=None, raw_zbuffer=False):
    """
    Off-screen depth rendering using OSMesa. If P=K[R t] (is_perspective=True),
    output is an image of depth values d(x,y) such that [x y 1]*d(x,y) restores
    the camera coordinates (P*x).

    :param vertices: (3*n, 3) triangle vertices or dict with keys 'f' and 'v'.
    :param P: (3, 4) matrix. K*[R t] (perspective) or [R t] (orthographic).
    :param out_wh: Output image width and height in pixels.
    :param lrtb: (left, right, top, bottom) viewing frustum points.
    :param is_perspective: This should be set to true when P=K*[R t].
    :param flip_z: If false, z will be flipped. Often true if P is perspective.
    :param near: Near clipping plane.
    :param far: Far clipping plane.
    :param raw_zbuffer: If true, depth will be 32 bit integer values.
    :return: (outwh[1], outwh[0]) depth image.

    :type vertices: np.ndarray | dict
    :type P: np.ndarray
    :rtype: np.ndarray | np.ma.core.MaskedArray
    """
    if type(vertices) == dict and 'f' in vertices and 'v' in vertices:
        vertices = vertices['v'][vertices['f'].ravel()]

    assert P.shape == (3, 4)
    pts = (P[:, :3].dot(vertices.T).astype(np.float64) + P[:, 3, None]).T

    if lrtb is None:
        lrtb = (0, out_wh[1] - 1, 0, out_wh[0] - 1)

    assert len(lrtb) == 4
    left, right, top, bottom = lrtb

    # X' = [x/z y/z z]
    if is_perspective:
        pts[:, :2] /= pts[:, 2, None]

    zmax = pts[:, 2].max(axis=0)
    zmin = pts[:, 2].min(axis=0)

    off = max((zmax - zmin) / 1e5, 1e-30)

    if near is None:
        near = zmin
    if far is None:
        far = zmax + off

    if flip_z:
        pts[:, 2] *= -1
    else:
        near, far = -far, -near

    P_ortho = ortho34(left, right, bottom, top, near, far)[:3, :]

    pts = (P_ortho[:, :3].dot(pts.T) + P_ortho[:, 3, None]).T
    pts = np.array(np.ascontiguousarray(pts))

    depth = rasterizer_func(pts, out_wh[::-1])

    if raw_zbuffer:
        return np.flipud(depth)

    mask = (depth == ((1 << 32) - 1))

    depth[mask] = np.nan
    depth = depth * (far - near) / ((1 << 32) - 1) + near

    depth = np.ma.array(depth, mask=mask, dtype=np.float32)
    return np.flipud(depth)


def render_silhouette(vertices, P, out_wh=(512, 512), lrtb=(0, 511, 0, 511),
                      is_perspective=True, flip_z=True, near=None, far=None):
    """
    See render_depth.
    """
    zbuffer = render_depth(vertices, P=P, out_wh=out_wh, lrtb=lrtb,
                           is_perspective=is_perspective, flip_z=flip_z,
                           near=near, far=None, raw_zbuffer=True)
    silhouette = zbuffer != ((1 << 32) - 1)
    return silhouette


def ortho34(left, right, bottom, top, znear, zfar):
    """
    :return: (3, 4) orthographic projection matrix.
    :rtype: np.ndarray
    """
    assert (right != left)
    assert (bottom != top)
    assert (znear != zfar)

    return np.array([
        [2.0 / (right - left), 0, 0, -(right + left) / float(right - left)],
        [0, 2.0 / (top - bottom), 0, -(top + bottom) / float(top - bottom)],
        [0, 0, -2.0 / (zfar - znear), -(zfar + znear) / float(zfar - znear)]
    ], dtype=np.float64)
