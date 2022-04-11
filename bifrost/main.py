from ast import literal_eval as make_tuple

import numpy as np
from typing import Tuple, Dict, List, Optional, Any

from bifrost.ir.output import OutputLayer
from bifrost.ir.input import InputLayer
from bifrost.ir.parameter import ParameterContext
from bifrost.ir.connection import MatrixConnector, Connection
from bifrost.ir.layer import NeuronLayer
from bifrost.exporter import export_network
from bifrost.ir.network import Network


def export(model_import, text_shape, writer, record: Dict[str, List[int]]):
    # todo:
    #  * what to do with the text_shape argument?
    #  * actually, the shape has to be set into input layer source
    #  * also, n_channels has to be set into the main attrs of input layer
    #  * how to pass in the input/output layers?

    model = __import__(model_import)  # this will do a import model_import as model
    parser, saver = get_parser(model)
    shape = make_tuple(text_shape)
    net, context, net_dict = parser(model, inp, out)
    saver(net_dict, net_dict_fname)
    set_recordings(net, record)
    result = export_network(net, context)
    writer.write(result)


def set_recordings(network, record: Dict[str, List[int]]) -> Network:
    for what in record:
        for which in record[what]:
            if isinstance(network.layers[which], OutputLayer):
                which -= 1

            if isinstance(network.layers[which].record, tuple):
                network.layers[which].record = [what]
            else:
                network.layers[which].record.append(what)


def get_parser_and_saver(model):
    class_name = model.__class__.__name__.lower()
    # for torch/norse
    if "sequentialstate" in class_name or hasattr(model, "_modules"):
        from bifrost.export.torch import TorchContext
        from bifrost.parse.parse_torch import torch_to_network, torch_to_context
        import torch

        def parse_torch(
            model: torch.nn.Module,
            input_layer: InputLayer,
            output_layer: Optional[OutputLayer] = None,
            config: Dict[str, Any] = {},
        ) -> Tuple[Network, ParameterContext[str]]:
            network = torch_to_network(model, input_layer, output_layer, config=config)
            context, net_dict = torch_to_context(network, model)
            return network, context, net_dict

        def save_net_dict(net_dict, filename):
            pass

        return parse_torch, save_net_dict

    # for ml_genn
    elif "model" in class_name and hasattr(model, "g_model"):
        from bifrost.parse.parse_ml_genn import ml_genn_to_network

        def save_net_dict(net_dict, filename):
            np.savez_compressed(filename, **net_dict)

        return ml_genn_to_network, save_net_dict

    else:
        raise Exception(f"Not supported model: {model}")
