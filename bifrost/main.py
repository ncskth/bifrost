from ast import literal_eval as make_tuple
from bifrost.ir.output import OutputLayer
from bifrost.ir.input import InputLayer
from bifrost.ir.parameter import ParameterContext
from typing import Tuple
from bifrost.ir.connection import MatrixConnector, Connection
from bifrost.ir.layer import LIFAlphaLayer

import torch

from bifrost.exporter import export_network
from bifrost.export.torch import TorchContext
from bifrost.ir.network import Network
from bifrost.parse.parse_torch import torch_to_network, torch_to_context

# todo: is this more like a test?

def export(model, text_shape, writer):
    shape = make_tuple(text_shape)
    data = torch.zeros(shape)
    # TODO: Parse graph
    # graph = model_to_graph(model, data)
    layers = [LIFAlphaLayer("l1", 1, 1), LIFAlphaLayer("l2", 1, 1)]
    connections = [Connection("l1", "l2", MatrixConnector())]

    context = TorchContext({})

    net = Network(layers=layers, connections=connections, runtime=100.0, timestep=1.0)
    result = export_network(net, context)
    writer.write(result)


def parse_torch(
    model: torch.nn.Module, input_layer: InputLayer, output_layer: OutputLayer
) -> Tuple[Network, ParameterContext[str]]:
    network = torch_to_network(model, input_layer, output_layer)
    context = torch_to_context(model)
    return network, context
