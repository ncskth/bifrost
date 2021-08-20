"""
Internal representations (IR) for internal use
"""
from dataclasses import dataclass
import numpy as np
from bifrost.ir.layer import Layer

@dataclass
class Connector:
    pass

@dataclass
class AllToAllConnector(Connector):
    """Also Known As DenseConnector"""
    pass

@dataclass
class DenseConnector(Connector):
    pass

@dataclass
class ConvolutionConnector(Connector):
    pass


@dataclass
class Connection:
    pre: Layer
    post: Layer
    connector: Connector
