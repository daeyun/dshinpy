from os import path
import sys
import os
import numpy as np


def read_mesh(filename):
    """
    :param filename: full path to mesh file.
    :return: dict with keys v and f.
        v: ndarray of size (num_vertices, 3), type uint32. zero-indexed.
        f: ndarray of size (num_faces, 3), type float64.
    """
    if filename.endswith('.off'):
        return read_off(filename)
    if filename.endswith('.ply'):
        return read_ply(filename)


def read_off(filename):
    filename = path.expanduser(filename)
    with open(filename) as f:
        content = f.read()
    lines = content.splitlines()

    assert lines[0].upper() == 'OFF'
    num_vertices, num_faces, _ = [int(val) for val in lines[1].split()]

    vertices = np.fromstring(' '.join(lines[2:num_vertices + 2]),
                             dtype=np.float64, sep=' ').reshape((-1, 3))
    faces = np.fromstring(
            ' '.join(lines[num_vertices + 2:num_vertices + num_faces + 2]),
            dtype=np.uint32,
            sep=' ').reshape((-1, 4))

    assert len(lines) == num_vertices + num_faces + 2
    assert (faces[:, 0] == 3).all(), "all triangle faces"

    faces = faces[:, 1:]

    if faces.min() != 0:
        print('faces.min() != 0', file=sys.stderr)

    if faces.max() != vertices.shape[0] - 1:
        print('faces.max() != vertices.shape[0]-1', file=sys.stderr)
        assert faces.max() < vertices.shape[0]

    return {
        'v': vertices,
        'f': faces,
    }


def read_ply(filename):
    from plyfile import PlyData
    plydata = PlyData.read(filename)
    v = np.vstack((plydata['vertex']['x'], plydata['vertex']['y'],
                   plydata['vertex']['z'])).T
    inds = plydata['face']['vertex_indices']
    f = np.vstack(([i for i in inds]))
    return {
        'v': v,
        'f': f,
    }


def save_off(mesh, filename):
    verts = mesh['v'].astype(np.float32)
    faces = mesh['f'].astype(np.int32)
    if not path.isdir(path.dirname(filename)):
        os.makedirs(path.dirname(filename))

    with open(path.expanduser(filename), 'ab') as fp:
        fp.write('OFF\n{} {} 0\n'.format(
                verts.shape[0], faces.shape[0]).encode('utf-8'))
        np.savetxt(fp, verts, fmt='%.5f')
        np.savetxt(fp, np.hstack((3 * np.ones((
            faces.shape[0], 1)), faces)), fmt='%d')
