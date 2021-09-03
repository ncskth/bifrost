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



def export(model_import, text_shape, writer, net_dict_fname='abc.npz'):
    model = __import__(model_import) # this will do a import model_import as model
    parser, saver = get_parser(model)
    shape = make_tuple(text_shape)
    data = torch.zeros(shape)
    net, context, net_dict = parser(model, inp, out)
    saver(net_dict, net_dict_fname)
    result = export_network(net, context)
    writer.write(result)


def get_parser_and_saver(model):
    class_name = model.__class__.__name__.lower()
    # for torch/norse
    if 'sequentialstate' in class_name and hasattr(model, '_modules'):
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

    # for ml_genn
    elif 'model' in class_name and hasattr(model, 'g_model'):
        from bifrost.parse.parse_ml_genn import ml_genn_to_network
        def save_net_dict(net_dict, filename):
            np.savez_compressed(filename, **net_dict)

        return ml_genn_to_network, save_net_dict



