import ctypes
import os
import sys
from ctypes import cdll
from os import path


def is_loaded(libpath):
    libpath = path.abspath(libpath)
    ret = os.system("lsof -p %d | grep %s > /dev/null" % (os.getpid(), libpath))
    return ret == 0


def reload_lib(lib):
    handle = lib._handle
    name = lib._name
    del lib
    while is_loaded(name):
        libdl = ctypes.CDLL("libdl.so")
        libdl.dlclose(handle)
    return ctypes.cdll.LoadLibrary(name)


def compile_and_reload(libpath):
    lib = cdll.LoadLibrary(libpath)

    source_mtime = path.getmtime(path.splitext(libpath)[0] + '.c')
    lib_mtime = path.getmtime(libpath)

    if source_mtime > lib_mtime:
        source_dir = path.dirname(libpath)
        ret = os.system('cd {} && make'.format(source_dir))
        if ret != 0:
            print('Error: make not successful.')
            sys.exit()
        lib = reload_lib(lib)

    return lib

