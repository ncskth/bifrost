from ast import literal_eval as make_tuple
import re
from pathlib import Path
from typing import List

from networkx import DiGraph

import torch
import torch.utils.tensorboard._pytorch_graph as pytorch_graph

from bifrost.ir import *

LAYER_REGEX = re.compile(r"^(\w*)\[(\d{1,3})\]$")


def attach_node(tree, node, inputs):
    # Only add nodes that have not been added and are not root
    if not node == node.root and not node == Path("."):
        if node not in tree:
            tree.add_node(node)
        if node.parent not in tree:
            attach_node(tree, node.parent, [])
    # Add edges
    for input in inputs:
        if input not in tree:
            tree.add_node(input)
        if not input == node:  # Avoid recurrence
            tree.add_edge(input, node)


def node_to_layer(n):
    p = Path(n)
    # name = p.parent.stem
    # match = LAYER_REGEX.match(name)
    # if match is None:
    #     layer = name
    #     index = 0
    # else:
    #     layer, index = match.groups()
    return TorchLayer(p.stem, p.parent)


def model_to_graph(model, data) -> List[TorchLayer]:
    # Extract layers into graph
    layer_graph = DiGraph()
    model_graph, _ = pytorch_graph.graph(model.to(data.device), data, False)

    for layer in model_graph.node:
        path = Path(layer.name).parent
        inputs = [Path(x).parent for x in layer.input]
        attach_node(layer_graph, path, inputs)
    # layers_and_inputs = [
    #     (node_to_layer(x.name), [node_to_layer(y) for y in x.input])
    #     for x in pytorch_graph.node
    # ]
    print(layer_graph)
    return layer_graph


def export(model, text_shape, writer):
    shape = make_tuple(text_shape)
    data = torch.zeros(shape)
    graph = model_to_graph(model, data)
    writer.write("hi")


if __name__ == "__main__":
    ms = torch.nn.Sequential(torch.nn.Linear(28, 1), torch.nn.ReLU())
    data = torch.zeros((1, 28, 28))
    layers = model_to_graph(ms, data)
    print(layers)
