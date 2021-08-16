"""
Internal representations (IR) for internal use
"""
from dataclasses import dataclass
import torch

from bifrost.ir.layer import Layer


@dataclass
class Synapse:
    pass


@dataclass
class StaticSynapse(Synapse):
    pass


@dataclass
class Connector:
    pass


@dataclass
class AllToAllConnector(Connector):
    pass


@dataclass
class ConvolutionConnector:
    weights: torch.Tensor
    padding: torch.Tensor = torch.tensor([1, 1])


@dataclass
class Connection:
    pre: Layer
    post: Layer
    connector: Connector
    synapse: Synapse = StaticSynapse()
