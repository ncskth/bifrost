from ast import literal_eval as make_tuple

import numpy as np
from bifrost.ir.output import OutputLayer
from bifrost.ir.input import InputLayer
from bifrost.ir.parameter import ParameterContext
from typing import Tuple
from bifrost.ir.connection import MatrixConnector, Connection
from bifrost.ir.layer import NeuronLayer
from bifrost.exporter import export_network

from bifrost.ir.network import Network

# todo: figure out if the input model is ml_genn or norse?

def export(model, text_shape, writer, net_dict_fname='abc.npz'):
    parser, saver = get_parser(model)
    shape = make_tuple(text_shape)
    data = torch.zeros(shape)
    # TODO: Parse graph
    # graph = model_to_graph(model, data)
    # layers = [NeuronLayer("l1", 1, 1), NeuronLayer("l2", 1, 1)]
    # connections = [Connection("l1", "l2", MatrixConnector())]
    #
    # context = TorchContext({})
    #
    # net = Network(layers=layers, connections=connections, runtime=100.0, timestep=1.0)
    net, context, net_dict = parser(model, inp, out)
    saver(net_dict, net_dict_fname)
    result = export_network(net, context)
    writer.write(result)


def get_parser_and_saver(model):
    if 'torch' in model.__class__.__name__.lower():
        from bifrost.export.torch import TorchContext
        from bifrost.parse.parse_torch import torch_to_network, torch_to_context
        import torch
        def parse_torch(model: torch.nn.Module, input_layer: InputLayer,
                        output_layer: OutputLayer) -> Tuple[Network, ParameterContext[str]]:
            network = torch_to_network(model, input_layer, output_layer)
            context = torch_to_context(network, model)
            return network, context, model

        def save_net_dict(net_dict, filename):
            pass

        return parse_torch, save_net_dict

    elif hasattr(model, 'g_model'):
        from bifrost.parse.parse_ml_genn import ml_genn_to_network
        def save_net_dict(net_dict, filename):
            np.savez_compressed(filename, **net_dict)

        return ml_genn_to_network, save_net_dict



