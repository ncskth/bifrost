from ast import literal_eval as make_tuple
from bifrost.ir import connection
from bifrost.ir.connection import AllToAllConnector, Connection
from bifrost.ir.layer import LIFLayer

import torch

from bifrost.exporter import export_network
from bifrost.ir.network import Network


def export(model, text_shape, writer):
    shape = make_tuple(text_shape)
    data = torch.zeros(shape)
    # TODO: Parse graph
    # graph = model_to_graph(model, data)
    layers = [LIFLayer("l1", 10), LIFLayer("l2", 20)]
    connections = [Connection("l1", "l2", AllToAllConnector())]

    net = Network(layers=layers, connections=connections, runtime=100.0, timestep=1.0)
    result = export_network(net)
    writer.write(result)
