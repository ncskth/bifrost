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
class AllToAllConnector:
    pass


@dataclass
class ConvolutionConnector:
    weights: torch.Tensor
    padding: torch.Tensor = torch.tensor([1, 1])


@dataclass
class Connection:
    pre: str
    post: str
    connector: Connector
    synapse: Synapse = StaticSynapse()
