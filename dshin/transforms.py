import math
import numpy
import numpy as np
import numpy.linalg as la
from dshin import stats


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
    T = np.array(T, dtype=np.float32).T
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
    S = np.array(S, dtype=np.float32).T
    M[...] = np.dot(M, S)


def xrotate(degrees):
    """
    http://www.labri.fr/perso/nrougier/teaching/opengl/
    """
    t = math.pi * degrees / 180
    cosT = math.cos(t)
    sinT = math.sin(t)
    R = numpy.array(
            [[1.0, 0.0, 0.0, 0.0],
             [0.0, cosT, -sinT, 0.0],
             [0.0, sinT, cosT, 0.0],
             [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    return R


def yrotate(degrees):
    """
    http://www.labri.fr/perso/nrougier/teaching/opengl/
    """
    t = math.pi * degrees / 180
    cosT = math.cos(t)
    sinT = math.sin(t)
    R = numpy.array(
            [[cosT, 0.0, sinT, 0.0],
             [0.0, 1.0, 0.0, 0.0],
             [-sinT, 0.0, cosT, 0.0],
             [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    return R


def zrotate(degrees):
    """
    http://www.labri.fr/perso/nrougier/teaching/opengl/
    """
    t = math.pi * degrees / 180
    cosT = math.cos(t)
    sinT = math.sin(t)
    R = numpy.array(
            [[cosT, -sinT, 0.0, 0.0],
             [sinT, cosT, 0.0, 0.0],
             [0.0, 0.0, 1.0, 0.0],
             [0.0, 0.0, 0.0, 1.0]], dtype=np.float32)
    return R


def rotate(M, angle, x, y, z, point=None):
    """
    rotate produces a rotation of angle degrees around the vector (x, y, z).
    http://www.labri.fr/perso/nrougier/teaching/opengl/

    Parameters
    ----------
    M
       Current transformation as a numpy array

    angle
       Specifies the angle of rotation, in degrees.

    x, y, z
        Specify the x, y, and z coordinates of a vector, respectively.
    """
    angle = math.pi * angle / 180
    c, s = math.cos(angle), math.sin(angle)
    n = math.sqrt(x * x + y * y + z * z)
    x /= n
    y /= n
    z /= n
    cx, cy, cz = (1 - c) * x, (1 - c) * y, (1 - c) * z
    R = numpy.array([[cx * x + c, cy * x - z * s, cz * x + y * s, 0],
                     [cx * y + z * s, cy * y + c, cz * y - x * s, 0],
                     [cx * z - y * s, cy * z + x * s, cz * z + c, 0],
                     [0, 0, 0, 1]]).T
    M[...] = np.dot(M, R)


def frustum(left, right, bottom, top, znear, zfar):
    """
    http://www.labri.fr/perso/nrougier/teaching/opengl/
    """
    assert (right != left)
    assert (bottom != top)
    assert (znear != zfar)

    M = np.zeros((4, 4), dtype=np.float32)
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
    pts = mesh['v'][mesh['f']]

    a = la.norm(pts[:, 0, :] - pts[:, 1, :], 2, axis=1)
    b = la.norm(pts[:, 1, :] - pts[:, 2, :], 2, axis=1)
    c = la.norm(pts[:, 2, :] - pts[:, 0, :], 2, axis=1)
    s = (a + b + c) / 2.0
    areas = np.sqrt(s * (s - a) * (s - b) * (s - c))
    areas = np.tile(areas, 3)

    pts = mesh['v'][mesh['f'].ravel()]

    weighted_std = stats.weighted_std(areas, pts)
    weighted_mean = stats.weighted_mean(areas, pts)

    t = -weighted_mean

    furthest = la.norm(pts, ord=2, axis=1).max()
    sigma = 2.5 * weighted_std

    # min(distance to the furthest point, 2.5 standard deviation) should have length 1.
    scale = 1.0 / min(furthest, sigma)

    M = np.array([
        [1, 0, 0, t[0]],
        [0, 1, 0, t[1]],
        [0, 0, 1, t[2]],
        [0, 0, 0, scale]
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
