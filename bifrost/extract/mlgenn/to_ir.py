from bifrost import ir as IR
from bifrost.ir import (NeuronLayer, Cell)
from copy import copy
import numpy as np

def get_ir_class(class_name):
    return getattr(IR, class_name)

def to_synapse(layer_dict):
    syn_class_name = layer_dict['type'].lower()
    # todo: this probably should go in the extraction part?
    if 'conv2d' in syn_class_name:
        syn_class = get_ir_class('ConvolutionSynapse')
    elif 'dense' in syn_class_name:
        syn_class = get_ir_class('DenseSynapse')
    else:
        raise NotImplementedError('Synapse Class not implemented')
    syn_type = layer_dict['params']['cell'].pop('synapse_type', 'current')
    syn_shape = layer_dict['params']['cell'].pop('synapse_shape', 'delta')
    return syn_class(syn_type, syn_shape)

def to_cell(cell_params):
    cell_dict = copy(cell_params)
    cell_name = cell_dict.pop('target')
    cell_class = get_ir_class(cell_name)
    return cell_class(cell_dict)

def to_neuron_layer(index, network_dictionary):
    keys = sorted(network_dictionary.keys())
    lkey = keys[index]
    ldict = copy(network_dictionary[lkey])
    size = ldict['params']['size']
    syn_type = ldict['type'].lower()

    shape = ldict['params'].get('shape', None)
    channs = ldict['params'].get('n_channels', 1)
    if 'conv2d' in syn_type:
        shape = shape[:2] # first two elements in array are height, width
    else:
        shape = [size, 1]

    sh_size = np.prod(shape)
    assert size == int(sh_size), \
           f'Size and Shape are not compatible {size} != product({shape})'
    synapse = to_synapse(ldict)
    cell = to_cell(ldict['params']['cell'])
    return NeuronLayer(index=index, key=lkey,
                       name=ldict['name'],
                       size=size,
                       cell=cell,
                       synapse=synapse,
                       n_channels=channs,
                       shape=shape,
                       )
