"""
Internal representations (IR) for internal use
"""
from dataclasses import dataclass
from typing import Generic, TypeVar

from bifrost.ir.layer import Layer
from bifrost.text_utils import sanitize
from bifrost.ir.constants import DefaultLayerKeys

class Connector:
    pass

@dataclass
class AllToAllConnector(Connector):
    """Also Known As DenseConnector"""
    weights_key: str = DefaultLayerKeys.WEIGHT

@dataclass
class MatrixConnector(Connector):  # todo: is this the same as All-to-All?
    weights_key: str = DefaultLayerKeys.WEIGHT

@dataclass
class DenseConnector(Connector):
    weights_key: str = DefaultLayerKeys.WEIGHT
    pooling_key: str = DefaultLayerKeys.POOLING

@dataclass
class ConvolutionConnector(Connector):
    weights_key: str = DefaultLayerKeys.WEIGHT
    pooling_key: str = DefaultLayerKeys.POOLING


From = TypeVar("From", Layer, Layer)
To = TypeVar("To", Layer, Layer)

@dataclass
class Connection(Generic[From, To]):
    pre: From
    post: To
    connector: Connector

    def variable(self, channel_in: int, channel_out: int) -> str:
        return sanitize(f"c_{self.pre.name}_{channel_in}__to__{self.post.name}_{channel_out}")
