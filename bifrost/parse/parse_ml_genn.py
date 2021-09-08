from bifrost import ir as IR
from bifrost.extract.ml_genn.extractor import extract_all
from bifrost.ir import (NeuronLayer, Cell, Connection)
from bifrost.ir.network import Network
from bifrost.ir.output import OutputLayer
from bifrost.ir.input import InputLayer
from bifrost.ir.constants import (SynapseTypes, SynapseShapes, NeuronTypes)
from bifrost.export.ml_genn import MLGeNNContext
from typing import List, Dict, Any
from copy import copy
import numpy as np
import ml_genn

def get_ir_class(class_name):
    return getattr(IR, class_name)

def to_synapse(layer_dict):
    syn_class_name = layer_dict['type'].lower()
    if 'conv2d' in syn_class_name:
        syn_class = get_ir_class('ConvolutionSynapse')
    elif 'dense' in syn_class_name:
        syn_class = get_ir_class('DenseSynapse')
    else:
        raise NotImplementedError('Synapse Class not implemented')
    syn_type = layer_dict['params']['cell'].pop('synapse_type',
                                                SynapseTypes.CURRENT)
    syn_shape = layer_dict['params']['cell'].pop('synapse_shape',
                                                 SynapseShapes.DELTA)
    return syn_class(syn_type, syn_shape)

def to_cell(cell_params):
    cell_class = get_ir_class(cell_params['target'])
    return cell_class()

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
    return NeuronLayer(name=ldict['name'], size=size,
                       cell=cell, synapse=synapse, channels=channs,
                       index=index, key=lkey, shape=shape,)

def to_connection(pre: NeuronLayer, post: NeuronLayer, network_dictionary):
    ldict = copy(network_dictionary[post.key])
    conn = get_ir_class(ldict['connector_type'])(post.key)
    if 'pool_shape' in ldict['params']:
        conn.pooling_key = str(post.key)
    return Connection(pre, post, conn)


def ml_genn_to_network(model: ml_genn.Model, input_layer: InputLayer,
        output_layer: OutputLayer, config: Dict[str, Any]={}) -> Network:
    runtime = config.get('runtime', -1.0)  # run forever if not specified
    timestep = config.get('timestep', model.g_model.dT)  # override timestep
    net_dict = extract_all(model)
    layers = []
    net_map = {}
    for i, k in enumerate(sorted(net_dict.keys())):
        if i == 0: # first index is the
            continue
        lyr = to_neuron_layer(i, net_dict)
        net_map[str(lyr)] = k
        layers.append(lyr)
    layers = [input_layer] + layers
    conns = [to_connection(layers[i], layers[i + 1], net_dict)
             for i in range(len(layers[:-1]))]

    network = Network(layers=layers, connections=conns, timestep=timestep,
                      runtime=runtime)

    if output_layer is not None:
        layers = network.layers + [output_layer]
        out_conn = [Connection(pre=network.layers[-1], post=output_layer,
                               connector=MatrixConnector("0"))]
        conns = network.connections + out_conn
        network = Network(layers=layers, connections=conns)
        # todo: how to add something to the network_dictionary to connect
        #       the head of the network to the live output? We probably just
        #       need the 'weights' even if they are just fake

    context = MLGeNNContext(net_map)

    return network, context, net_dict


