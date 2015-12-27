from os import path
import glog as log
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
        log.warn('faces.min() != 0')

    if faces.max() != vertices.shape[0] - 1:
        log.warn('faces.max() != vertices.shape[0]-1')
        assert faces.max() < vertices.shape[0]

    return {
        'v': vertices,
        'f': faces,
    }
