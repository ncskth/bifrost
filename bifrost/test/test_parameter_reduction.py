from bifrost.extract.utils import try_reduce_param
import numpy as np


def test_random_vector():
    vector = np.random.uniform(0.0, 1.0, 10)
    reduced = try_reduce_param(vector)
    assert vector.shape == reduced.shape


def test_same_vector():
    vector = np.ones(10) * np.random.uniform(0.0, 1.0)
    reduced = try_reduce_param(vector)
    assert vector.size == 10
    assert np.isscalar(reduced)
    assert isinstance(reduced, float)


def test_dim_0_array():
    # single value arrays are 'extracted' to a float
    # they should have the _item_ method to get the float value
    value = np.array(np.random.uniform(0.0, 1.0))
    assert np.ndim(value) == 0
    assert hasattr(value, "item")

    reduced = try_reduce_param(value)
    assert np.isscalar(reduced)
    assert isinstance(reduced, float)


def test_float():
    value = np.random.uniform(0.0, 1.0)
    assert np.ndim(value) == 0
    assert not hasattr(value, "item")

    reduced = try_reduce_param(value)
    assert np.isscalar(reduced)
    assert isinstance(reduced, float)
