import numbers

import numpy as np
import typecheck as tc
import typecheck.framework

from dshin import log


class array(typecheck.framework.Checker):
    def __init__(self, *args, dtype=None):
        self._dims = []
        for i, dim in enumerate(args):
            if dim is None:
                if i != len(args) - 1:
                    raise tc.InputParameterError('`None` must be in the last dimension.')
            elif not (isinstance(dim, (list, tuple)) or isinstance(dim, numbers.Integral) or isinstance(dim, str)):
                raise tc.InputParameterError('Unexpected dimension {} of type {}.'.format(dim, type(dim)))

            elif isinstance(dim, (list, tuple)):
                if len(dim) == 0 or len(dim) > 3:
                    raise tc.InputParameterError('Dimension type should be an integer or a tuple of size <= 3.')
                for item in dim:
                    if item is not None and (not isinstance(item, numbers.Integral) or item <= 0):
                        raise tc.InputParameterError('Dimension tuple must contain positive integer values.')
                d = list(dim)
                while len(d) < 3:
                    d.append(None)
                self._dims.append(tuple(d))

            elif isinstance(dim, numbers.Integral):
                if dim <= 0:
                    raise tc.InputParameterError('Dimension must be a positive integer, a tuple, or a string.')
                self._dims.append(dim)

            else:
                if not isinstance(dim, str):
                    raise tc.InputParameterError('Dimension must be a positive integer, a tuple, or a string.')
                self._dims.append(dim)

        if dtype is None:
            self._dtype_names = None
        else:
            if isinstance(dtype, type):
                self._dtype_names = [dtype.__name__]
            elif tc.seq_of(type).check(dtype, namespace=''):
                self._dtype_names = [item.__name__ for item in dtype]
            else:
                raise tc.InputParameterError('Expected dtype to be a type. Instead got {} of type {}.'.format(
                    dtype, type(dtype).__name__))

    def check(self, value: np.ndarray, namespace) -> bool:
        if not isinstance(value, np.ndarray):
            log.error('Expected value of type np.ndarray but instead got %s of type %s.', str(value), type(value).__name__)
            return False

        if self._dtype_names is not None and value.dtype.name not in self._dtype_names:
            log.error('Expected types %s but instead got type %s.', self._dtype_names, value.dtype)
            return False

        variables = {}
        for i, dim in enumerate(self._dims):
            if isinstance(dim, numbers.Integral):
                if value.shape[i] != dim:
                    return False
            elif isinstance(dim, tuple):
                if dim[0] is not None:
                    if dim[0] <= 0:
                        raise TypeError('Invalid lower bound value {} for dimension {} in Array type annotation.'.format(dim[0], i))
                    if value.shape[i] < dim[0]:
                        return False
                if dim[1] is not None:
                    if dim[1] <= 0:
                        raise TypeError('Invalid upper bound value {} for dimension {} in Array type annotation.'.format(dim[1], i))
                    if value.shape[i] > dim[1] != 0:
                        return False
                if dim[2] is not None:
                    if dim[2] <= 0:
                        raise TypeError('Invalid multiplier value {} for dimension {} in Array type annotation.'.format(dim[2], i))
                    if value.shape[i] % dim[2] != 0:
                        return False
            elif isinstance(dim, str):
                if dim in variables:
                    if value.shape[i] != variables[dim]:
                        return False
                else:
                    print(variables)
                    variables[dim] = value.shape[i]
            elif dim is None:
                if value.ndim > i:
                    return False
                break
            else:
                raise RuntimeError

        return True
