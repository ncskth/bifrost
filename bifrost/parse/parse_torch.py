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
from bifrost.ir.constants import SynapseTypes, SynapseShapes
from bifrost.extract.utils import try_reduce_param

# todo: remove all the magic constants and move them to a common file

Continuation = Callable[[Network], Network]

dont_parse_classes = (
    norse.SequentialState,
    torch.nn.modules.loss._Loss, # loss functions
    torch.nn.modules.batchnorm._NormBase # normalizers
)
def trimed_named_modules(torch_net: torch.nn.Module):
    def _invalid(k, modules):
        return (isinstance(modules[k], dont_parse_classes) or k == '')

    modules = dict(torch_net.named_modules()) # this expands inner SequentialStates
    remove_these =  [k for k in modules.keys() if _invalid(k, modules)]
    for k in remove_these:
        del modules[k]

    return modules


def get_shapes(modules: Dict[str, torch.nn.Module], network: Network) -> Dict[str, torch.Size]:
    input_layer = network.layers[0] # assuming the first layer is of input type
    # Batch size, Number of Channels, Height, Width
    # this has to be in [e.g. (8, 3, 28, 28)] BCYX format
    input_shape = (1, input_layer.channels, *input_layer.source.shape)

    shapes = {}
    x = torch.randn(input_shape)
    for i, k in enumerate(modules):
        print(k)
        x = modules[k](x)

        # first is output of previous layer, second is the state
        if isinstance(x, tuple):
            x = x[0]
        shapes[k] = x.data.shape
    return shapes


def torch_to_network(model: torch.nn.Module, input_layer: InputLayer,
        output_layer: OutputLayer, config: Dict[str, Any]={}) -> Network:

    if not isinstance(model, (norse.SequentialState, pl.LightningModule)):
            raise ValueError("Unknown model type", type(model))

    runtime = config.get('runtime', -1.0)  # run forever if not specified
    timestep = config.get("timestep", 1.0)

    net_dict = trimed_named_modules(model)

    default_network = Network(layers=[input_layer], connections=[],
                              runtime=runtime, timestep=timestep)

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
            size = int(np.prod(shape[-2:]))
            # size = int(np.prod(layer_info['output_size'][-2:]))
            channels = shape[1]
            cell = __choose_cell(mod)

            connector = __get_connector(last_neuron_idx + 1, idx, keys, modules)
            synapse = __get_synapse(last_neuron_idx + 1, idx, keys, modules)
            post = NeuronLayer(f"{name}_{k}", size, channels, cell=cell,
                               synapse=synapse, key=k)
            conn = Connection(layers[-1], post, connector)

            layers.append(post)
            connections.append(conn)

            last_neuron_idx = idx
            continue

        # this should not be reached
        raise ValueError("Unknown torch module ", name)

    return network


def __choose_cell(module: torch.nn.Module):
    name = module.p.__class__.__name__.lower()
    if 'liparameters' in name:
        return LICell()
    else:
        return LIFCell()  # default


def __choose_synapse_shape(torch_params):
    if hasattr(torch_params, 'alpha'):
        return SynapseShapes.ALPHA
    else:  # note: do we support delta?
        return SynapseShapes.EXPONENTIAL

def __get_synapse(start_idx: int, end_idx: int, keys: List[int],
                    modules: Dict[int, torch.nn.Module]) -> Synapse:
    syn = None
    for conn_idx in range(start_idx, end_idx):
        k = keys[conn_idx]
        name = modules[k].__class__.__name__.lower()
        if 'conv2d' in name:
            syn = ConvolutionSynapse()
            break
        elif 'dense' in name:
            syn = DenseSynapse()
            break
        elif 'linear' in name:
            syn = StaticSynapse()
            break
        elif 'avgpool2d' in name:
            continue
        # this should not be reached
        raise ValueError("Unknown connector", name)

    mod = modules[keys[end_idx]]
    syn.synapse_shape = __choose_synapse_shape(mod.p)
    # note: seems we only support current synapse types as of now (7/9/21)
    return syn


def __get_connector(start_idx: int, end_idx: int, keys: List[int],
                    modules: Dict[int, torch.nn.Module]) -> Connector:
    pool_idx = None
    ctor = None
    for conn_idx in range(start_idx, end_idx):
        k = keys[conn_idx]
        name = modules[k].__class__.__name__.lower()
        pool = '' if pool_idx is None else str(pool_idx)
        if 'conv2d' in name:
            ctor = ConvolutionConnector(str(k), pooling_key=pool)
            continue
        elif 'dense' in name:
            ctor = DenseConnector(str(k), pooling_key=pool)
            continue
        elif 'linear' in name:
            ctor = MatrixConnector(str(k))
            continue
        elif 'avgpool2d' in name:
            pool_idx = k
            continue
        # this should not be reached
        raise ValueError("Unknown connector", name)

    return ctor


def module_to_layer(
    module: torch.nn.Module, index: int, input_channels: int, input_neurons: int
) -> Layer:
    if isinstance(module, norse.LICell):
        return NeuronLayer(str(index), input_neurons, input_channels,
                           index=index, cell=LICell())
    elif isinstance(module, norse.LIFCell):
        return NeuronLayer(str(index), input_neurons, input_channels, index=index,
                           cell=LIFCell())
    else:
        raise ValueError("Unknown torch module layer", module)
