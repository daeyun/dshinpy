import numpy as np
import scipy.linalg as la


def ray_triangle_intersection(faces, vertices, ray):
    assert faces.shape[1] == 3
    assert vertices.shape[1] == 3

    is_valid = np.ones((faces.shape[0]), dtype=np.bool)
    intersections = np.zeros((faces.shape[0], 3))
    intersections[:] = np.nan

    triangles = vertices[faces]
    u = triangles[:, 1, :] - triangles[:, 0, :]
    v = triangles[:, 2, :] - triangles[:, 0, :]
    n = np.cross(u, v, axis=1)

    # check if degenerate
    is_valid &= (la.norm(n, 2, axis=1) > 1e-16)

    vdir = ray[1, :] - ray[0, :]
    w0 = ray[0, :] - triangles[:, 0, :]

    a = -(n * w0).sum(axis=1)
    b = (n * vdir).sum(axis=1)

    # check if parallel to plane. if a==0, ray lies in plane
    is_valid &= np.abs(b) > 1e-12

    r = a[is_valid] / b[is_valid]

    # check if ray goes away from triangle if r < 0

    if is_valid.any():
        intersections[is_valid] = ray[None, 0, :] + r[:, None] * vdir[None, :]

    uu = (u[is_valid] * u[is_valid]).sum(axis=1)
    uv = (u[is_valid] * v[is_valid]).sum(axis=1)
    vv = (v[is_valid] * v[is_valid]).sum(axis=1)
    w = intersections[is_valid] - triangles[is_valid, 0, :]
    wu = (w * u[is_valid, :]).sum(axis=1)
    wv = (w * v[is_valid, :]).sum(axis=1)
    D = uv * uv - uu * vv

    s = (uv * wv - vv * wu) / D
    t = (uv * wu - uu * wv) / D

    # check if intersection point is inside triangle
    is_valid[is_valid] &= np.logical_and(s >= 0, s <= 1) & np.logical_and(t >= 0, t + s <= 1)

    intersections[np.logical_not(is_valid), :] = np.nan

    return intersections, is_valid
