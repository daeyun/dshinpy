import numpy as np
import pytest
import typecheck as tc

from dshin import typing_utils


def test_numpy_array_dimension():
    assert typing_utils.array(1, 2).check(np.empty((1, 2)), namespace='')
    assert typing_utils.array(1).check(np.empty((1, 2)), namespace='')
    assert typing_utils.array(2).check(np.empty((2, 2)), namespace='')
    assert typing_utils.array(2, (3,)).check(np.empty((2, 3)), namespace='')
    assert typing_utils.array(2, (3, 3)).check(np.empty((2, 3)), namespace='')
    assert typing_utils.array(2, (3, None, 1)).check(np.empty((2, 3)), namespace='')
    assert typing_utils.array(2, (None, 3, 1)).check(np.empty((2, 3)), namespace='')
    assert typing_utils.array(2, [None, 3, 1]).check(np.empty((2, 3)), namespace='')
    assert typing_utils.array(2, (3, 3, 1)).check(np.empty((2, 3)), namespace='')
    assert not typing_utils.array(2, (3, 3, 2)).check(np.empty((2, 3)), namespace='')
    assert not typing_utils.array(2, (3, 3, 2)).check(np.empty((2, 4)), namespace='')
    assert not typing_utils.array(2, (3, 3, 2)).check(np.empty((2, 2)), namespace='')
    assert typing_utils.array(2, None).check(np.empty((2, 2)), namespace='')
    assert not typing_utils.array(2, 3).check(np.empty((2, 2)), namespace='')


def test_numpy_array_dtype():
    assert typing_utils.array(dtype=np.bool).check(np.empty((2, 2), dtype=np.bool), namespace='')
    assert typing_utils.array(2, dtype=np.bool).check(np.empty((2, 2), dtype=np.bool), namespace='')
    assert typing_utils.array(2, dtype=np.float64).check(np.empty((2, 2), dtype=np.float64), namespace='')
    assert not typing_utils.array(2, dtype=np.float32).check(np.empty((2, 2), dtype=np.float64), namespace='')
    assert not typing_utils.array(2, dtype=float).check(np.empty((2, 2), dtype=np.float64), namespace='')
    assert not typing_utils.array(2, dtype=float).check(np.empty((2, 2), dtype=np.float32), namespace='')
    assert not typing_utils.array(2, dtype=float).check(np.empty((2, 2), dtype=np.int32), namespace='')
    assert not typing_utils.array(2, dtype=np.bool).check(np.empty((2, 2), dtype=np.int32), namespace='')
    assert typing_utils.array(2, dtype=(np.bool, np.int32)).check(np.empty((2, 2), dtype=np.int32), namespace='')
    assert typing_utils.array(2, dtype=(np.bool, np.int32, np.float32)).check(np.empty((2, 2), dtype=np.bool), namespace='')
    assert not typing_utils.array(2, dtype=(np.bool, np.int32, np.float32)).check(np.empty((2, 2), dtype=np.float64), namespace='')


def test_numpy_array_type_invalid_init():
    with pytest.raises(tc.InputParameterError):
        assert typing_utils.array(2, None, None)
    with pytest.raises(tc.InputParameterError):
        assert typing_utils.array(2, None, 8)
    with pytest.raises(tc.InputParameterError):
        assert typing_utils.array(0)
    with pytest.raises(tc.InputParameterError):
        assert typing_utils.array(dtype='')
    with pytest.raises(tc.InputParameterError):
        assert typing_utils.array(1, dtype=1)
    with pytest.raises(tc.InputParameterError):
        assert typing_utils.array([1, 1, 1, 1])
    with pytest.raises(tc.InputParameterError):
        assert typing_utils.array([1, 1, 0])
    with pytest.raises(tc.InputParameterError):
        assert typing_utils.array([1, '', 1])


def test_numpy_array_dimension_variable():
    assert typing_utils.array('N').check(np.empty((2, 2)), namespace='')
    assert typing_utils.array('N', 'N').check(np.empty((2, 2)), namespace='')
    assert typing_utils.array('N', 'M').check(np.empty((2, 2)), namespace='')
    assert typing_utils.array('N', 'M').check(np.empty((2, 3)), namespace='')
    assert not typing_utils.array('N', 'N').check(np.empty((2, 3)), namespace='')
    assert typing_utils.array('N', 'M', 'N', 'M').check(np.empty((2, 3, 2, 3)), namespace='')
    assert not typing_utils.array('N', 'M', 'N', 'M').check(np.empty((2, 3, 2, 4)), namespace='')
    assert not typing_utils.array('N', 'M', 'N', 'M').check(np.empty((3, 3, 2, 3)), namespace='')
    assert typing_utils.array('N', 'N', 'M').check(np.empty((3, 3, 2, 3)), namespace='')
    assert not typing_utils.array('N', 'N', 'N').check(np.empty((3, 3, 2, 3)), namespace='')


def test_numpy_array_type_checker_decorator_return_values():
    @tc.typecheck
    def return_none(arr: typing_utils.array(1, 1)) -> typing_utils.array(1, 1):
        return None

    @tc.typecheck
    def tile(arr: typing_utils.array(1, 1)) -> typing_utils.array(1, 1):
        return np.tile(arr, [2])

    @tc.typecheck
    def identity(arr: typing_utils.array(1, 1)) -> typing_utils.array(1, 1):
        return arr

    @tc.typecheck
    def cast(arr: typing_utils.array(1, 1)) -> typing_utils.array(dtype=np.float64):
        return arr.astype(np.float32)

    with pytest.raises(tc.ReturnValueError):
        return_none(np.empty((1, 1)))

    with pytest.raises(tc.ReturnValueError):
        return_none(np.empty((1, 1)))

    with pytest.raises(tc.ReturnValueError):
        tile(np.empty((1, 1)))

    identity(np.empty((1, 1)))


def test_numpy_array_type_checker_decorator_type_error():
    @tc.typecheck
    def accept_11(arr: typing_utils.array(1, 1)):
        return None

    @tc.typecheck
    def accept_11_float32(arr: typing_utils.array(1, 1, dtype=np.float32)):
        return None

    with pytest.raises(tc.TypeCheckError):
        accept_11(np.empty((1, 2)))

    with pytest.raises(tc.TypeCheckError):
        accept_11_float32(np.empty((1, 1), dtype=np.float64))

    accept_11(np.empty((1, 1), dtype=np.float32))
