import numpy as np

def size_from_shape(shape):
    return int(np.prod(shape))

def layer_attr(layer, attr):
    return getattr(layer, attr)