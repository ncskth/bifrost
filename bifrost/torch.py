from typing import List
import torch
from bifrost.export.torch import TorchContext
from bifrost.exporter import export_network

from bifrost.ir.input import RandomPoissonSource, InputLayer
from bifrost.ir.output import EthernetOutput, OutputLayer
from bifrost.parse.parse_torch import torch_to_context, torch_to_network


def parse_poisson(model: torch.nn.Module, rates: List[int], target_address: str):
    assert ":" in target_address, "Target address must be of type <ip>:<port>"
    host, port = target_address.split(":")

    input_layer = InputLayer(
        name="Input",
        size=len(rates),
        channels=1,
        source=RandomPoissonSource(shape=(len(rates),), rates=rates),
    )
    output_layer = OutputLayer(
        name="Output", size=1, channels=1, sink=EthernetOutput(host=host, port=port)
    )

    network = torch_to_network(model, input_layer, output_layer)
    context = torch_to_context(network, model)

    return export_network(network, context)
