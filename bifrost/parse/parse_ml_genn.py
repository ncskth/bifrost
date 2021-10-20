from bifrost import ir as IR
from bifrost.extract.ml_genn.extractor import extract_all
from bifrost.export.ml_genn import MLGeNNContext
from bifrost.ir import (NeuronLayer, Cell, Connection)
from bifrost.ir.network import Network
from bifrost.ir.output import OutputLayer
from bifrost.ir.input import InputLayer, ImageDataset, PoissonImageDataset
from bifrost.ir.constants import (SynapseTypes, SynapseShapes, NeuronTypes)
from bifrost.parse.utils import adjust_runtime
from bifrost.parse.constants import DEFAULT_RUNTIME, DEFAULT_DT
from typing import List, Dict, Any
from copy import copy
import numpy as np
import ml_genn


def get_ir_class(class_name):
    return getattr(IR, class_name)


def to_synapse(layer_dict):
    syn_class_name = layer_dict["type"].lower()
    if "conv2d" in syn_class_name:
        syn_class = get_ir_class("ConvolutionSynapse")
    elif "dense" in syn_class_name:
        syn_class = get_ir_class("DenseSynapse")
    else:
        raise NotImplementedError("Synapse Class not implemented")

    if not ("params" in layer_dict):
        raise AttributeError("Layer description dictionary should contain "
                             "\"params\" key to extract "
                             "synapse type and shapes parameters")

    if not ("cell" in layer_dict["params"]):
        raise AttributeError("Layer description dictionary should contain "
                             "\"params\" and then \"cell\" keys to extract "
                             "synapse type and shapes parameters")

    syn_type = layer_dict["params"]["cell"].pop("synapse_type",
                                                SynapseTypes.CURRENT)
    syn_shape = layer_dict["params"]["cell"].pop("synapse_shape",
                                                 SynapseShapes.DELTA)
    return syn_class(syn_type, syn_shape)


def to_cell(cell_params):
    cell_target_class = cell_params["target"]
    if cell_target_class in ["IFCell", "LICell", "LIFCell"]:
        cell_class = get_ir_class(cell_target_class)
    else:
        raise NotImplementedError("Cell Class not implemented")

    return cell_class()


def to_neuron_layer(index: int, network_dictionary: Dict, network: Network) -> NeuronLayer:
    """
    :param index: numeric index for the layer to translate
    :param network_dictionary: ml_genn description from bifrost.extract.ml_genn
    :return bifrost.ir.layer :
    """
    keys = sorted(network_dictionary.keys())
    layer_key = keys[index]
    layer_dictionary = copy(network_dictionary[layer_key])
    size = layer_dictionary["params"]["size"]
    syn_type = layer_dictionary["type"].lower()

    shape = layer_dictionary["params"].get("shape", None)
    channs = layer_dictionary["params"].get("n_channels", 1)

    if "conv2d" in syn_type:
        shape = shape[:2] # first two elements in array are height, width
    else:
        shape = [1, size]

    sh_size = np.prod(shape)
    assert size == int(sh_size), \
           f"Size and Shape are not compatible {size} != product({shape})"
    synapse = to_synapse(layer_dictionary)

    reset_variables = layer_dictionary["params"]["cell"].pop("reset_variables",
                                                             None)
    cell = to_cell(layer_dictionary["params"]["cell"])

    if reset_variables is not None:
        cell.reset_variables_values = reset_variables
    name = f"{index:03d}_{layer_dictionary['name']}"
    return NeuronLayer(name=name, size=size, cell=cell, synapse=synapse,
                       channels=channs, index=index, key=layer_key, shape=shape,
                       network=network)


def to_connection(pre: NeuronLayer, post: NeuronLayer, network_dictionary: Dict,
                  network: Network) -> Connection:
    # weights are stored in the post-synaptic population
    layer_dict = copy(network_dictionary[post.key])
    conn = get_ir_class(layer_dict["connector_type"])(post.key)
    if "pool_shape" in layer_dict["params"]:
        conn.pooling_key = str(post.key)
    return Connection(pre, post, conn, network=network)


def ml_genn_to_network(model: ml_genn.Model, input_layer: InputLayer,
        output_layer: OutputLayer, config: Dict[str, Any]={}) -> Network:
    # don't run if  no time is specified
    runtime = adjust_runtime(config.get("runtime", DEFAULT_DT),
                             input_layer)
    timestep = config.get("timestep", model.g_model.dT)  # override timestep
    split_runs = config.get("split_runs", False)
    configuration = config.get('configuration', {})

    net_dict = extract_all(model)
    network = Network([input_layer], [], timestep=timestep,
                      runtime=runtime, configuration=configuration,
                      split_runs=split_runs)
    layers = network.layers
    net_map = {}
    for i, k in enumerate(sorted(net_dict.keys())):
        if i == 0: # first index is the
            continue
        lyr = to_neuron_layer(i, net_dict, network)
        net_map[str(lyr)] = k
        layers.append(lyr)

    conns = network.connections
    for i, k in enumerate(sorted(net_dict.keys())[:-1]):
        conns.append(to_connection(layers[i], layers[i + 1], net_dict, network))
        if i > 0:
            layers[i].incoming_connection = conns[i - 1]

    layers[-1].incoming_connection = conns[-1]

    if output_layer is not None:
        output_layer.source = network.layers[-1]
        network.layers.append(output_layer)

    context = MLGeNNContext(net_map)

    return network, context, net_dict


