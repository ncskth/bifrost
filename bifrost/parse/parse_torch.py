from bifrost.export.torch import TorchContext
from bifrost.ir.output import OutputLayer
from bifrost.ir.input import InputLayer
from bifrost.ir.connection import AllToAllConnector, Connection, MatrixConnector
from bifrost.ir.layer import NeuronLayer, Layer
from typing import Callable, List, Optional, Tuple
from bifrost.ir.parameter import ParameterContext
from bifrost.ir.network import Network
import torch
import pytorch_lightning as pl

import norse.torch as norse

Continuation = Callable[[Network], Network]


def torch_to_network(
    model: torch.nn.Module, input_layer: InputLayer, output_layer: OutputLayer
) -> Network:
    if not isinstance(model, (norse.SequentialState, pl.LightningModule)):
            raise ValueError("Unknown model type", type(model))

    default_network = Network(layers=[input_layer], connections=[])

    network = module_to_ir(
        modules=list(model.children()), index=0, network=default_network
    )

    if output_layer is not None:
        layers = network.layers + [output_layer]
        out_conn = [Connection(pre=network.layers[-1], post=output_layer,
                               connector=MatrixConnector("0"))]
        conns = network.connections + out_conn
        network = Network(layers=layers, connections=conns)

    return network


def torch_to_context(net: Network, modules: List[torch.nn.Module]) -> ParameterContext[str]:
    layer_map = {
        str(l): l.name
        for l in net.layers
        if not (isinstance(l, InputLayer))
    }

    return TorchContext(layer_map)


def module_to_ir(
    modules: List[torch.nn.Module],
    index: int,
    network: Network,
) -> Network:
    if len(modules) == 0:
        return network

    if isinstance(modules[0], torch.nn.CrossEntropyLoss):
        return module_to_ir(modules[1:], index + 1, network)

    if isinstance(modules[0], norse.SequentialState):
        return module_to_ir(list(modules[0].children()), index, network)

    if isinstance(modules[0], torch.nn.Linear):
        assert (
            len(modules) > 0
        ), "Linear layer requires output connection, but was final layer"
        linear = modules[0]
        out = module_to_layer(modules[1], index + 1, 1, linear.out_features)
        connection = Connection(
            pre=network.layers[-1], post=out, connector=MatrixConnector(str(index))
        )
        network = Network(
            layers=network.layers + [out],
            connections=network.connections
            + [connection],
        )
        return module_to_ir(modules[2:], index + 2, network)

    if isinstance(modules[0], torch.nn.Conv2d):
        assert (
            len(modules) > 0
        ), "Linear layer requires output connection, but was final layer"
        conv2d_in_conn = modules[0]
        neuron = module_to_layer(modules[1], index + 1, 1, conv2d_in_conn.out_features)
        if netwok.layers[-1]



    raise ValueError("Unknown torch module", modules[0])


def module_to_layer(
    module: torch.nn.Module, index: int, input_channels: int, input_neurons: int
) -> Layer:
    if isinstance(module, norse.LIFCell):
        return NeuronLayer(str(index), input_channels, input_neurons, index=index)
    else:
        raise ValueError("Unknown torch module layer", module)
