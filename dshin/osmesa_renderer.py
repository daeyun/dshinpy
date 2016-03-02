import ctypes
from ctypes import cdll
from os import path

import numpy as np
from numpy.ctypeslib import ndpointer

from sys import platform as _platform

if _platform == "darwin":
    libname = 'render_depth.darwin.so'
else:
    libname = 'render_depth.linux.so'

file_dir = path.dirname(path.realpath(__file__))
libpath = path.join(file_dir, 'c', libname)
lib = cdll.LoadLibrary(libpath)

lib.render_depth.restype = ctypes.c_int
lib.render_depth.argtypes = [ndpointer(ctypes.c_double, flags="C_CONTIGUOUS"),
                             ctypes.c_size_t, ctypes.c_int, ctypes.c_int,
                             ndpointer(ctypes.c_uint32, flags="C_CONTIGUOUS")]


def zbuffer_int32(triangle_pts, hw):
    depth = np.zeros(hw, dtype=np.uint32)
    ret = lib.render_depth(triangle_pts, triangle_pts.size,
                           depth.shape[1], depth.shape[0], depth)
    if ret != 0:
        raise RuntimeError("render_depth() returned an error.")
    print(ret, np.unique(depth))
    return depth
