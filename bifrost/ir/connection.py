"""
Internal representations (IR) for internal use
"""
from dataclasses import dataclass
import numpy as np
from bifrost.ir.layer import Layer


@dataclass
class Synapse:
    pass

@dataclass
class StaticSynapse(Synapse):
    pass


@dataclass
class Connector:
    weights: np.ndarray

@dataclass
class AllToAllConnector(Connector):
    """Also Known As DenseConnector"""
    pass

@dataclass
class ConvolutionConnector(Connector):
    pass


@dataclass
class Connection:
    pre: Layer
    post: Layer
    connector: Connector
    synapse: Synapse = StaticSynapse
