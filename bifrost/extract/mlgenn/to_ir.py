from bifrost import ir as IR
from bifrost.ir import (NeuronLayer, Cell)
from copy import copy
import numpy as np

def get_cell_class(class_name):
    return getattr(IR, class_name)

def to_cell(cell_params):
    cell_dict = copy(cell_params)
    cell_name = cell_dict.pop('target')
    cell_class = get_cell_class(cell_name)
    return cell_class(cell_dict)

def to_neuron_layer(index, network_dictionary):
    keys = sorted(network_dictionary.keys())
    lkey = keys[index]
    ldict = network_dictionary[lkey]
    size = ldict['params']['size']
    syn_type = ldict['type'].lower()

    if 'conv2d' in syn_type:
        shape = ldict['params']['shape'][:2] # first two elements in array are height, width
        channs = ldict['params']['n_channels']
    else:
        shape = [size, 1]
        channs = 1

    assert size == int(np.prod(shape)), \
           f'Size and Shape are not compatible {size} != product({shape})'

    cell = to_cell(ldict['params']['cell'])
    return NeuronLayer(index=index, key=lkey,
                       name=ldict['name'],
                       size=size,
                       cell=cell,
                       n_channels=channs,
                       shape=shape,
                       )
