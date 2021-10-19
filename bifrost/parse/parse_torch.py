from typing import Callable, List, Optional, Tuple, Dict, Any
from copy import copy
import numpy as np
from collections import OrderedDict

import os
os.environ["CUDA_VISIBLE_DEVICES"]=""

import torch
torch.device("cpu")
from torch import Tensor

import pytorch_lightning as pl
import norse.torch as norse

from bifrost.export.torch import TorchContext
from bifrost.ir.output import OutputLayer
from bifrost.ir.input import InputLayer
from bifrost.ir.connection import (
    AllToAllConnector, Connection, MatrixConnector, ConvolutionConnector,
    DenseConnector, Connector
)
from bifrost.ir.cell import LIFCell, LICell
from bifrost.ir.layer import NeuronLayer, Layer
from bifrost.ir.synapse import (
    Synapse, ConvolutionSynapse, DenseSynapse, StaticSynapse
)

from bifrost.ir.parameter import ParameterContext
from bifrost.ir.network import Network
from bifrost.ir.constants import SynapseTypes, SynapseShapes, DefaultLayerKeys
from bifrost.extract.utils import try_reduce_param
from bifrost.extract.torch.parameter_buffers import (set_parameter_buffers,
                                                     DONT_PARSE_THESE_MODULES)
from bifrost.parse.utils import adjust_runtime
from bifrost.parse.constants import DEFAULT_RUNTIME, DEFAULT_DT


# todo: remove all the magic constants and move them to a common file

Continuation = Callable[[Network], Network]

START_OF_POPULATION_SHAPE_INDEX = -2
CHANNEL_INDEX = 1

def trimed_named_modules(torch_net: torch.nn.Module):
    def _invalid(k, modules):
        instance_invalid = isinstance(modules[k], DONT_PARSE_THESE_MODULES)
        is_topmost = k == ''  # means top-most module
        has_children = len(list(modules[k].children()))
        return (is_topmost or has_children or instance_invalid)


    # this expands inner SequentialStates
    modules = dict(torch_net.named_modules())
    remove_these =  [k for k in modules.keys() if _invalid(k, modules)]
    for k in remove_these:
        del modules[k]

    return modules


def get_shapes(modules: Dict[str, torch.nn.Module], network: Network) -> Dict[str, torch.Size]:
    input_layer = network.layers[0] # assuming the first layer is of input type
    # Batch size, Number of Channels, Height, Width
    # this has to be in [e.g. (8, 3, 28, 28)] BCYX format
    input_shape = (1, input_layer.channels, *input_layer.source.shape)
    # input_shape = (input_layer.channels, *input_layer.source.shape)

    shapes = {}
    x = torch.randn(input_shape)
    keys = list(modules.keys())
    last_module = modules[keys[0]]
    for i, k in enumerate(keys):
        print(k)
        print(x.shape)
        print(modules[k].__class__.__name__)
        if isinstance(modules[k], torch.nn.Linear):
            x = x.view(1, -1)
            print(x.shape)

        x = modules[k](x)

        # first is output of previous layer, second is the state
        if isinstance(x, tuple):
            x = x[0]
        shapes[k] = x.data.shape
    return shapes


def torch_to_network(model: torch.nn.Module, input_layer: InputLayer,
        output_layer: OutputLayer, config: Dict[str, Any]={}) -> Network:

    __acceptable_parents = (
        norse.SequentialState, pl.LightningModule, torch.nn.Module
    )
    if not isinstance(model, __acceptable_parents):
            raise ValueError("Unknown model type", type(model))

    # default is 0, don't run if  no time is specified
    runtime = adjust_runtime(config.get('runtime', DEFAULT_RUNTIME),
                             input_layer)
    timestep = config.get("timestep", DEFAULT_DT)
    configuration = config.get('configuration', {})

    net_dict = trimed_named_modules(model)
    source_dt = 1.0
    for k in net_dict:
        class_name = net_dict[k].__class__.__name__
        if 'LIF' in class_name:
            source_dt = net_dict[k].dt
            break

    default_network = Network(layers=[input_layer], connections=[],
                              runtime=runtime, timestep=timestep,
                              source_dt=source_dt,
                              configuration=configuration)

    network = module_to_ir(modules=net_dict, network=default_network)

    if output_layer is not None:
        output_layer.source = network.layers[-1]
        network.layers.append(output_layer)

    return network


def torch_to_context(net: Network, torch_net: torch.nn.Module) -> ParameterContext[str]:
    net_dict = trimed_named_modules(torch_net)

    state_dict = torch_net.state_dict(keep_vars=True)
    weight_keys = set(['.'.join(k.split('.')[:-1]) for k in state_dict.keys()])

    keys = list(net_dict.keys())
    layer_map = {}
    last_stride_idx = 0

    for idx, layer in enumerate(net.layers):
        if isinstance(layer, InputLayer):
            continue
        map_key = str(layer)
        layer_map[map_key] = layer.key

    return TorchContext(layer_map), {'state_dict': state_dict}


def module_to_ir(modules: Dict[str, torch.nn.Module], network: Network) -> Network:
    shapes = get_shapes(modules, network)
    layer_map = {}
    layers = network.layers
    connections = network.connections
    keys = list(modules.keys())
    last_neuron_idx = -1
    for idx, k in enumerate(keys):
        mod = modules[k]
        shape = shapes[k]
        name = mod.__class__.__name__.lower()
        if name in ['conv2d', 'avgpool2d', 'linear']:
            continue

        if name in ['licell', 'lifcell']:
            _shape = shape[START_OF_POPULATION_SHAPE_INDEX:]
            size = int(np.prod(_shape))
            channels = 1 if len(shape) == 2 else shape[CHANNEL_INDEX]
            cell = __choose_cell(mod)

            # todo: get this from some configuration
            cell.reset_variables_values = [('v', 0.0)]
            # end-todo

            connector = __get_connector(last_neuron_idx + 1, idx, keys, modules)
            synapse = __get_synapse(last_neuron_idx + 1, idx, keys, modules)
            post = NeuronLayer(f"{name}_{k}", size, channels, dt=network.source_dt,
                               cell=cell, synapse=synapse, key=k, shape=_shape,
                               network=network)
            conn = Connection(layers[-1], post, connector, network=network)

            post.incoming_connection = conn
            layers[-1].outgoing_connection = conn

            layers.append(post)
            connections.append(conn)

            last_neuron_idx = idx
            continue

        # this should not be reached
        raise ValueError("Unknown torch module ", name)

    return network


def __choose_cell(module: torch.nn.Module):
    # neuron types for Norse (LI or LIF) have an attribute for parameters, this
    # also dictates which 'cell' type we use
    # todo: I think this is somewhat wrong --- need help
    parameter_class_name = module.p.__class__.__name__.lower()
    if 'liparameters' in parameter_class_name:
        return LICell()
    else:
        return LIFCell()  # default


def __choose_synapse_shape(torch_params):
    # if hasattr(torch_params, 'alpha'):
    #     return SynapseShapes.ALPHA
    # else:  # note: do we support delta?
    #     return SynapseShapes.EXPONENTIAL
    # TODO: I think Norse just does exponential synapses
    return SynapseShapes.EXPONENTIAL


def __get_synapse(start_index: int, end_index: int, keys: List[str],
                  modules: Dict[str, torch.nn.Module]) -> Synapse:
    synapse = None
    for layer_index in range(start_index, end_index):
        k = keys[layer_index]
        module_class_name = modules[k].__class__.__name__.lower()
        if 'conv2d' in module_class_name:
            synapse = ConvolutionSynapse()
            break
        elif 'dense' in module_class_name:
            synapse = DenseSynapse()
            break
        elif 'linear' in module_class_name:
            synapse = DenseSynapse()
            # synapse = StaticSynapse()
            break
        elif 'avgpool2d' in module_class_name:  # these usually come after conv2d
            continue
        # this should not be reached
        raise ValueError("Unknown connector", module_class_name)

    # end index is always the one for a Cell (LI or LIF)
    module = modules[keys[end_index]]
    synapse.synapse_shape = __choose_synapse_shape(module.p)
    # note: seems we only support current synapse types as of now (7/9/21)
    return synapse


def __get_connector(start_idx: int, end_idx: int, keys: List[str],
                    modules: Dict[str, torch.nn.Module]) -> Connector:
    pool_index = None
    connector = None
    for conn_idx in range(start_idx, end_idx):
        k = keys[conn_idx]
        module = modules[k]
        module_class_name = module.__class__.__name__.lower()
        pool_key = (
            DefaultLayerKeys.POOLING if pool_index is None else str(pool_index)
        )
        bias_key = (keys[conn_idx]
                    if hasattr(module, 'bias') else DefaultLayerKeys.BIAS)
        if 'conv2d' in module_class_name:
            connector = ConvolutionConnector(str(k), pooling_key=pool_key)
            break
        elif 'dense' in module_class_name:
            connector = DenseConnector(str(k), pooling_key=pool_key)
            break
        elif 'linear' in module_class_name:
            connector = DenseConnector(str(k), pooling_key=pool_key)
            # todo: do we really need the Matrix connector?
            # connector = MatrixConnector(str(k))
            break
        elif 'avgpool2d' in module_class_name:
            pool_index = k
            continue


        # this should not be reached
        raise ValueError("Unknown connector", module_class_name)

    connector.bias_key = bias_key

    return connector
