import math
import numpy
import numpy as np
import scipy.linalg as la
import skimage
import scipy.ndimage as spndim
from dshin import stats
from dshin import camera
from dshin import log

def translate(M, x, y=None, z=None):
    """
    translate produces a translation by (x, y, z) .
    http://www.labri.fr/perso/nrougier/teaching/opengl/

    Parameters
    ----------
    x, y, z
        Specify the x, y, and z coordinates of a translation vector.
    """
    if y is None: y = x
    if z is None: z = x
    T = [[1, 0, 0, x],
         [0, 1, 0, y],
         [0, 0, 1, z],
         [0, 0, 0, 1]]
    T = np.array(T, dtype=np.float64).T
    M[...] = np.dot(M, T)

def scale(M, x, y=None, z=None):
    """
    scale produces a non uniform scaling along the x, y, and z axes. The three
    parameters indicate the desired scale factor along each of the three axes.
    http://www.labri.fr/perso/nrougier/teaching/opengl/

    Parameters
    ----------
    x, y, z
        Specify scale factors along the x, y, and z axes, respectively.
    """
    if y is None: y = x
    if z is None: z = x
    S = [[x, 0, 0, 0],
         [0, y, 0, 0],
         [0, 0, z, 0],
         [0, 0, 0, 1]]
    S = np.array(S, dtype=np.float64).T
    M[...] = np.dot(M, S)

def xrotate(theta, deg=True):
    """
    http://www.labri.fr/perso/nrougier/teaching/opengl/
    """
    if deg:
        theta = math.pi * theta / 180.0
    cosT = math.cos(theta)
    sinT = math.sin(theta)
    R = numpy.array(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, cosT, -sinT, 0.0],
             [0.0, sinT, cosT, 0.0],
             [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    return R

def yrotate(theta, deg=True):
    """
    http://www.labri.fr/perso/nrougier/teaching/opengl/
    """
    if deg:
        theta = math.pi * theta / 180.0
    cosT = math.cos(theta)
    sinT = math.sin(theta)
    R = numpy.array(
            [[cosT, 0.0, sinT, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [-sinT, 0.0, cosT, 0.0],
             [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    return R

def zrotate(theta, deg=True):
    """
    http://www.labri.fr/perso/nrougier/teaching/opengl/
    """
    if deg:
        theta = math.pi * theta / 180.0
    cosT = math.cos(theta)
    sinT = math.sin(theta)
    R = numpy.array(
            [[cosT, -sinT, 0.0, 0.0],
             [sinT, cosT, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]], dtype=np.float64)
    return R

def rotation_matrix(angle, direction, point=None, deg=True):
    """
    http://www.lfd.uci.edu/~gohlke/code/transformations.py.html
    """
    if deg:
        angle = math.pi * angle / 180
    sina = math.sin(angle)
    cosa = math.cos(angle)
    direction /= la.norm(direction, 2)
    # rotation matrix around unit vector
    R = numpy.diag([cosa, cosa, cosa])
    R += numpy.outer(direction, direction) * (1.0 - cosa)
    direction *= sina
    R += numpy.array([[0.0, -direction[2], direction[1]],
                      [direction[2], 0.0, -direction[0]],
                      [-direction[1], direction[0], 0.0]])
    M = numpy.identity(4)
    M[:3, :3] = R
    if point is not None:
        # rotation not around origin
        point = numpy.array(point[:3], dtype=numpy.float64, copy=False)
        M[:3, 3] = point - numpy.dot(R, point)
    return M

def angle(v1, v2, axis=0, deg=False, ref_plane=None):
    """
    Angle between two vectors.
    :param axis: 0 if column vectors, 1 if row vectors.
    :param deg: Returns angle in degrees if True, radians if False.
    :param ref_plane: If set, returns a signed angle for right-handed rotation with respect to this plane.
    """
    v1 = v1 / la.norm(v1, ord=2, axis=axis, keepdims=True)
    v2 = v2 / la.norm(v2, ord=2, axis=axis, keepdims=True)

    # More numerically stable than arccos.
    dotprod = (v1 * v2).sum(axis=axis)
    crossprod = np.cross(v1, v2, axis=axis)
    angle = np.arctan2(la.norm(crossprod, ord=2, axis=axis, keepdims=True), dotprod)
    if deg:
        angle = angle / math.pi * 180.0
    if ref_plane is not None:
        angle *= np.sign((crossprod * ref_plane).sum(axis=axis))
    return angle

def frustum(left, right, bottom, top, znear, zfar):
    """
    http://www.labri.fr/perso/nrougier/teaching/opengl/
    """
    assert (right != left)
    assert (bottom != top)
    assert (znear != zfar)

    M = np.zeros((4, 4), dtype=np.float64)
    M[0, 0] = +2.0 * znear / (right - left)
    M[2, 0] = (right + left) / (right - left)
    M[1, 1] = +2.0 * znear / (top - bottom)
    M[3, 1] = (top + bottom) / (top - bottom)
    M[2, 2] = -(zfar + znear) / (zfar - znear)
    M[3, 2] = -2.0 * znear * zfar / (zfar - znear)
    M[2, 3] = -1.0
    return M

def perspective(fovy, aspect, znear, zfar):
    """
    http://www.labri.fr/perso/nrougier/teaching/opengl/
    """
    assert (znear != zfar)
    h = np.tan(fovy / 360.0 * np.pi) * znear
    w = h * aspect
    return frustum(-w, w, -h, h, znear, zfar)

def lookat_matrix(cam_xyz, obj_xyz, up=(0, 0, 1)):
    """
    Right-handed rigid transform. -Z points to object.

    :param cam_xyz: (3,)
    :param obj_xyz: (3,)
    :return: (3,4)
    """
    cam_xyz = np.array(cam_xyz)
    obj_xyz = np.array(obj_xyz)
    F = obj_xyz - cam_xyz
    f = F / la.norm(F)

    up = np.array(up)
    u = up / la.norm(up)

    s = np.cross(f, u)
    s /= la.norm(s)

    u = np.cross(s, f)

    R = np.vstack((s, u, -f))

    M = np.hstack([R, np.zeros((3, 1))])
    T = np.eye(4)
    T[:3, 3] = -cam_xyz
    MT = M.dot(T)

    return MT

def apply_Rt(Rt, pts, inverse=False):
    """
    :param Rt: (3,4)
    :param pts: (n,3)
    :return:
    """
    if inverse:
        R = Rt[:, :3].T
        t = -R.dot(Rt[:, 3, None])
        Rtinv = np.hstack((R, t))
        return Rtinv.dot(np.vstack((pts.T, np.ones((1, pts.shape[0]))))).T
    return Rt.dot(np.vstack((pts.T, np.ones((1, pts.shape[0]))))).T

def ortho44(left, right, bottom, top, znear, zfar):
    assert (right != left)
    assert (bottom != top)
    assert (znear != zfar)

    return np.array([
        [2.0 / (right - left), 0, 0, -(right + left) / float(right - left)],
        [0, 2.0 / (top - bottom), 0, -(top + bottom) / float(top - bottom)],
        [0, 0, -2.0 / (zfar - znear), -(zfar + znear) / float(zfar - znear)],
        [0, 0, 0, 1]
    ], dtype=np.float64)

def apply44(M, pts):
    assert 2 == len(pts.shape)
    assert pts.shape[1] == 3

    if M.shape == (4, 4):
        Mpts = M.dot(np.vstack((pts.T, np.ones((1, pts.shape[0])))))
        return (Mpts[:3, :] / Mpts[3, :]).T
    else:
        raise NotImplementedError()

def normalize_mesh_vertices(mesh, up='+z'):
    # pts = mesh['v'][mesh['f']]
    # a = la.norm(pts[:, 0, :] - pts[:, 1, :], 2, axis=1)
    # b = la.norm(pts[:, 1, :] - pts[:, 2, :], 2, axis=1)
    # c = la.norm(pts[:, 2, :] - pts[:, 0, :], 2, axis=1)
    # s = (a + b + c) / 2.0
    # areas_sq = s * (s - a) * (s - b) * (s - c)
    # areas_sq = np.abs(areas_sq)
    # areas = np.sqrt(areas_sq)
    # areas = np.tile(areas, 3)

    pts = mesh['v'][mesh['f'].ravel()]
    # weighted_std = stats.weighted_std(areas, pts)
    # weighted_mean = stats.weighted_mean(areas, pts)

    t = -(mesh['v'].max(0) + mesh['v'].min(0)) / 2

    furthest = la.norm(pts + t, ord=2, axis=1).max()
    # sigma = 2 * weighted_std

    scale = 1.0 / furthest

    M = np.array([
        [1, 0, 0, t[0]],
        [0, 1, 0, t[1]],
        [0, 0, 1, t[2]],
        [0, 0, 0, 1.0 / scale]
    ])

    if up == '+z':
        R = np.eye(4)
    elif up == '-z':
        R = xrotate(180)
    elif up == '+y':
        R = xrotate(90)
    elif up == '-y':
        R = xrotate(-90)
    elif up == '+x':
        R = yrotate(90)
    elif up == '-x':
        R = yrotate(-90)
    else:
        raise RuntimeError('Unrecognized up axis: {}'.format(up))

    return R.dot(M)

def cam_pts_from_ortho_depth(depth, trbl=(1, 1, -1, -1)):
    d = depth.copy()
    if hasattr(depth, 'mask'):
        d[depth.mask] = np.nan
    im_wh = d.shape[::-1]

    newd = np.concatenate((np.indices(d.shape), d[None, :, :].data), axis=0).astype(np.float64)

    impts = np.vstack((newd[1, :, :].ravel(), newd[0, :, :].ravel(), newd[2, :, :].ravel())).T

    # important.
    impts[:, :2] += 0.5

    valid_inds = np.logical_not(np.isnan(impts[:, 2]))
    impts = impts[valid_inds, :].astype(np.float64)

    top, right, bottom, left = [1, 1, -1, -1]

    impts[:, 0] *= (right - left) / im_wh[0]
    impts[:, 1] *= -(top - bottom) / im_wh[1]
    impts[:, 0] += left
    impts[:, 1] += top
    impts[:, 2] *= -1

    return impts

    # impts3d = apply_Rt(Rt, impts, inverse=True)

def quaternion_matrix(quaternion):
    """Return homogeneous rotation matrix from quaternion.
    http://www.lfd.uci.edu/~gohlke/code/transformations.py.html
    """
    _EPS = numpy.finfo(float).eps * 4.0

    q = numpy.array(quaternion, dtype=numpy.float64, copy=True)
    n = numpy.dot(q, q)
    if n < _EPS:
        return numpy.identity(4)
    q *= math.sqrt(2.0 / n)
    q = numpy.outer(q, q)
    return numpy.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
        [0.0, 0.0, 0.0, 1.0]])

def random_rotation(is_homogeneous=False):
    v = np.random.randn(4)
    Q = quaternion_matrix(v / la.norm(v, ord=2))

    assert np.isclose(la.det(Q), 1.0)
    assert np.isclose(Q[3, 3], 1.0)

    return Q if is_homogeneous else Q[:3, :3]

def xyz_to_sph(xyz):
    """
    :return: radius, inclination, azimuth. In radians.
    0 <= inclination <= 180
    -180 < azimuth <= 180
    """
    r = la.norm(xyz, ord=2, axis=-1, keepdims=True).reshape(-1, 1)
    assert (r != 0).all()
    xyz = xyz.reshape(-1, 3)
    inclination = np.arccos(xyz[:, 2, None] / r)
    azimuth = np.arctan2(xyz[:, 1, None], xyz[:, 0, None])
    return np.hstack((r, inclination, azimuth)).reshape(xyz.shape)

def sph_to_xyz(sph):
    r, inclination, azimuth = sph[:, 0], sph[:, 1], sph[:, 2]
    x = r * np.sin(inclination) * np.cos(azimuth)
    y = r * np.sin(inclination) * np.sin(azimuth)
    z = r * np.cos(inclination)
    xyz = np.stack((x, y, z), axis=1)
    assert xyz.shape == sph.shape
    return xyz

def spherical_coord_align_rotation(v1, v2):
    a = v1.copy()
    b = v2.copy()
    a = a / la.norm(a, 2)
    b = b / la.norm(a, 2)
    a[2] = b[2] = 0
    azimuth = angle(a, b, deg=True, ref_plane=np.array([0, 0, 1]))
    assert -180 <= azimuth <= 180
    xyrot = zrotate(azimuth, deg=True)[:3, :3]
    vs = v1.dot(xyrot.T)
    elaxis = np.cross(vs, v2)
    elevation = angle(vs, v2, deg=True, ref_plane=elaxis)
    assert -90 <= elevation <= 90
    elrot = rotation_matrix(elevation, elaxis, deg=True)[:3, :3]
    Q = elrot.dot(xyrot)
    assert np.allclose(la.det(Q), 1.0)
    return Q

def pca_svd(X):
    """
    :param X: (n, d)
    :return: principal components (columns) and singular values.
    """
    _, s, V = la.svd(X - X.mean(axis=0))
    return V.T, s

def depth_normals(depth, worldpts, viewdir, window_size=5, min_near_pts=4, visualize=False, ax=None):
    """
    :param viewdir: vector from camera to object. normals direction will be less than 90 degrees from viewdir.
    :return: 3-channel images filled with inward normals and worldpts. There may be discarded points.
    """
    assert window_size % 2 == 1
    assert len(viewdir.shape) == 1
    assert np.isclose(la.norm(viewdir, 2), 1.0)
    assert len(depth.shape) == 2
    assert len(worldpts.shape) == 2
    assert depth.shape[0] > 4
    assert worldpts.shape[1] == 3
    assert (~np.isnan(depth)).sum() == worldpts.shape[0]

    imxyz = np.full((depth.shape[0], depth.shape[1], 3), np.nan, dtype=np.float32)
    imxyz[~np.isnan(depth)] = worldpts

    npts = worldpts.shape[0]
    nthsmallest = max(3, int(npts / 40))  # rough heuristic
    gap = np.partition(la.norm(worldpts[:-1, :] - worldpts[1:, :], 2, axis=1),
                       nthsmallest)[nthsmallest]
    maxdist = gap * (2 ** 0.5) * ((5 - 1) / 2) * 1.02  # rough heuristic

    padding_left, padding_right = int((window_size - 1) / 2), int((window_size - 1) / 2 + 0.5)
    windows = skimage.util.view_as_windows(
            np.pad(imxyz, [[padding_left, padding_right], [padding_left, padding_right], [0, 0]],
                   mode='constant', constant_values=np.nan), (window_size, window_size, 3), step=1)

    normals = np.full(imxyz.shape, np.nan, dtype=np.float32)

    for idx in np.ndindex(imxyz.shape[0], imxyz.shape[1], 1):
        win = windows[idx].view()
        # filter center
        current = win[padding_left, padding_left]

        if np.isnan(current[0]):
            continue
        # assuming nan if first channel has nan.
        valid = ~np.isnan(win[:, :, 0])
        if valid.sum() < min_near_pts:
            continue
        pts = win[valid] - current
        dists = la.norm(win[valid] - current, ord=2, axis=1)
        valid = dists < maxdist
        if valid.sum() < min_near_pts:
            continue
        pts = pts[valid]
        pc, s = pca_svd(pts)
        normal = pc[:, 2]
        normal /= la.norm(normal, ord=2)
        normals[idx[:2]] = normal

    flip = (normals * viewdir).sum(2) > 0
    normals[flip] *= -1

    if visualize:
        from dshin import geom3d
        pts = imxyz[~np.isnan(normals).any(axis=2)]
        vecs = normals[~np.isnan(normals).any(axis=2)]
        edges = np.stack((pts, pts + vecs * 0.05), axis=1)
        ax = geom3d.pts(pts, markersize=30, ax=ax)

        start = la.norm(worldpts - worldpts.mean(0), 2, 1).max() * 1.2 * viewdir.reshape(1, 3)
        end = worldpts.mean(0).reshape(1, 3) * 0.2 + start * 0.8
        geom3d.draw_arrow_3d(start, end, ax=ax)
        geom3d.edge_3d(edges, ax=ax, colors='green', linewidths=0.5)
        geom3d.pts(pts + vecs * 0.1, ax=ax, color='red')

    imxyz[np.isnan(normals)] = np.nan

    return normals, imxyz

def rescale_and_recenter(image, hw=(64, 64), padding=1):
    assert 2 == len(image.shape)
    # Crop margins with nan values.
    y, x = np.where(~np.isnan(image))
    try:
        h, w = y.max() - y.min() + 1, x.max() - x.min() + 1
        center = int(y.min() + h / 2), int(x.min() + w / 2)
        image = image[center[0] - h / 2:center[0] + h / 2, center[1] - w / 2:center[1] + w / 2]
        # Resize.
        longest = max(zip(image.shape, hw))
        resize = (longest[1]-padding*2) / longest[0]
        resized = spndim.zoom(image, zoom=resize, order=0, mode='constant', cval=np.nan)
        h, w = resized.shape
        output = np.full(hw, np.nan)
        hstart, wstart = int((output.shape[0] - h) / 2 + 0.5), int((output.shape[1] - w) / 2 + 0.5)
        output[hstart:hstart + h, wstart:wstart + w] = resized
    except Exception as ex:
        log.warn(str(ex))
        output = np.full(hw, np.nan)
    return output
