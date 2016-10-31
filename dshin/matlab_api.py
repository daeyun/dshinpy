# import matlab
# import matlab.engine
# import numpy as np
# import copy
# from dshin import log
# from os import path
#
# _matlab_engine = None
#
#
# def engine():
#     global _matlab_engine
#     if _matlab_engine is None:
#         log.info('Starting MATLAB engine.')
#         _matlab_engine = matlab.engine.start_matlab()
#     return _matlab_engine
#
#
# def set_workdir(workdir_path):
#     assert path.isdir(workdir_path)
#     eng = engine()
#     eng.cd(workdir_path)
#     eng.addpath(eng.genpath(workdir_path))
#
#
# def fit_mesh_to_voxel_space(mesh, res=50, padding=0, bmin=None, bmax=None):
#     assert mesh['v'].shape[1] == 3
#     assert mesh['f'].shape[1] == 3
#
#     mesh = copy.deepcopy(mesh)
#
#     if bmin is None:
#         bmin = mesh['v'].min(0)
#     if bmax is None:
#         bmax = mesh['v'].max(0)
#
#     if not isinstance(bmin, np.ndarray):
#         bmin = np.array(bmin)
#     if not isinstance(bmax, np.ndarray):
#         bmax = np.array(bmax)
#
#     mesh['v'] -= bmin
#     mesh['v'] *= (res - 1 - padding * 2) / (bmax - bmin).max() + 1
#
#     bbmax = (bmax - bmin) * (res - 1 - padding * 2) / (bmax - bmin).max() + 1
#     mesh['v'] += ((res - padding * 2) - bbmax) / 2.0
#
#     mesh['v'] += padding
#
#     return mesh
#
#
# def voxelize_mesh(mesh, res=50, padding=0, bmin=None, bmax=None):
#     mesh = fit_mesh_to_voxel_space(mesh, res, padding, bmin=bmin, bmax=bmax)
#
#     # Assumes vertices are 0-indexed.
#     mesh = {
#         'faces': matlab.double((mesh['f'] + 1).tolist()),
#         'vertices': matlab.double((mesh['v']).tolist()),
#     }
#
#     dims = matlab.double([res, res, res])
#
#     eng = engine()
#     volume = eng.polygon2voxel(mesh, dims, 'none', nargout=1)
#     volume = np.array(volume, dtype=np.bool)
#
#     # Example: Convert volume to pcl:
#     # y, x, z = np.where(volume)
#     # pts = np.vstack([x,y,z]).T+1
#
#     return volume
#
#
# def surface_dist(mesh1, mesh2):
#     assert mesh1['v'].shape[1] == 3
#     assert mesh1['f'].shape[1] == 3
#     assert mesh2['v'].shape[1] == 3
#     assert mesh2['f'].shape[1] == 3
#
#     # Make sure surface area is not too big. Otherwise it takes too long.
#
#     # Assumes vertices are 0-indexed.
#     mesh1 = {
#         'faces': matlab.double((mesh1['f'] + 1).tolist()),
#         'vertices': matlab.double((mesh1['v']).tolist()),
#     }
#     mesh2 = {
#         'faces': matlab.double((mesh2['f'] + 1).tolist()),
#         'vertices': matlab.double((mesh2['v']).tolist()),
#     }
#     eng = engine()
#     surf_dist = eng.surfDistMetric(mesh1, mesh2, nargout=1)
#
#     return surf_dist
#
#
# def sample_points_in_mesh(mesh):
#     assert mesh['v'].shape[1] == 3
#     assert mesh['f'].shape[1] == 3
#
#     # Assumes vertices are 0-indexed.
#     mesh = {
#         'f': matlab.double((mesh['f'].T + 1).tolist()),
#         'v': matlab.double((mesh['v'].T).tolist()),
#     }
#     eng = engine()
#     return eng.samplePointsInMesh(mesh, nargout=1)
