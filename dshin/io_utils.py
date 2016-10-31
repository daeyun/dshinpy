from os import path
import tempfile

import stl
from dshin import log
import sys
import os
import io
import numpy as np
import hashlib
import functools


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


def read_off_num_fv(filename: str) -> tuple:
    filename = path.expanduser(filename)
    with open(filename, 'r') as f:
        first_two_lines = [f.readline(), f.readline()]
    assert first_two_lines[0][:3] == 'OFF'

    tokens = ' '.join([item.strip() for item in [
        first_two_lines[0][3:],
        first_two_lines[1]
    ]]).split()

    num_vertices = int(tokens[0])
    num_faces = int(tokens[1])

    # TODO(daeyun): Make this a warning.
    # OK to fail.
    assert int(tokens[2]) == 0

    return num_faces, num_vertices


def read_off(filename):
    """
    Read OFF mesh files.

    File content must start with "OFF". Not always followed by a whitespace.
    """
    filename = path.expanduser(filename)
    with open(filename, 'r') as f:
        content = f.read()

    assert content[:3].upper() == 'OFF'
    content = content[4:] if content[3] == '\n' else content[3:]
    lines = content.splitlines()

    num_vertices, num_faces, _ = [int(val) for val in lines[0].split()]

    vertices = np.fromstring(' '.join(lines[1:num_vertices + 1]),
                             dtype=np.float64, sep=' ').reshape((-1, 3))
    faces = np.fromstring(
        ' '.join(lines[num_vertices + 1:num_vertices + num_faces + 1]),
        dtype=np.uint32,
        sep=' ').reshape((-1, 4))

    assert len(lines) == num_vertices + num_faces + 1
    assert (faces[:, 0] == 3).all(), "all triangle faces"

    faces = faces[:, 1:]

    if faces.min() != 0:
        print('faces.min() != 0', file=sys.stderr)

    if faces.max() != vertices.shape[0] - 1:
        print('faces.max() != vertices.shape[0]-1', file=sys.stderr)
        assert faces.max() < vertices.shape[0]

    assert vertices.shape[0] == num_vertices
    assert faces.shape[0] == num_faces

    return {
        'v': vertices,
        'f': faces,
    }


def save_stl(mesh, filename):
    faces = mesh['f']
    verts = mesh['v']
    stl_mesh = stl.mesh.Mesh(np.zeros(faces.shape[0], dtype=stl.mesh.Mesh.dtype))
    for i, f in enumerate(faces):
        for j in range(3):
            stl_mesh.vectors[i][j] = verts[f[j], :]
    stl_mesh.save(filename)


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

    filename = path.expanduser(filename)
    if not path.isdir(path.dirname(filename)):
        os.makedirs(path.dirname(filename))

    # Overwrite.
    if path.exists(filename):
        log.warn('Removing existing file %s', filename)
        os.remove(filename)

    with open(path.expanduser(filename), 'ab') as fp:
        fp.write('OFF\n{} {} 0\n'.format(
            verts.shape[0], faces.shape[0]).encode('utf-8'))
        np.savetxt(fp, verts, fmt='%.5f')
        np.savetxt(fp, np.hstack((3 * np.ones((
            faces.shape[0], 1)), faces)), fmt='%d')


def merge_meshes(*args):
    v = np.concatenate([item['v'] for item in args], axis=0)
    offsets = np.cumsum([0] + [item['v'].shape[0] for item in args])
    f = np.concatenate([item['f'] + offset for item, offset in zip(args, offsets)], axis=0)
    return {'v': v, 'f': f}


def sha1(objs):
    assert isinstance(objs, list), isinstance(objs, tuple)
    sha1 = hashlib.sha1()
    for obj in objs:
        sha1.update(str(obj).encode('utf8'))
    return sha1.hexdigest()


def sha256(objs):
    assert isinstance(objs, list), isinstance(objs, tuple)
    h = hashlib.sha256()
    for obj in objs:
        h.update(str(obj).encode('utf8'))
    return h.hexdigest()


def stringify_float_arrays(arr_list, precision=6):
    assert isinstance(arr_list, list), isinstance(arr_list, tuple)
    arr = np.hstack(arr_list).ravel().astype(np.float32)
    return np.array_str(arr, precision=precision, max_line_width=np.iinfo(np.int64).max)


def temp_filename(dirname='/tmp', prefix='', suffix=''):
    temp_name = next(tempfile._get_candidate_names())
    return path.join(dirname, prefix+temp_name+suffix)
