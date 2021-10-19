"""
Internal representations (IR) for internal use
"""
from dataclasses import dataclass
from typing import Generic, TypeVar, Optional

from bifrost.ir.layer import Layer
from bifrost.text_utils import sanitize
from bifrost.ir.constants import DefaultLayerKeys
from bifrost.ir.bases import (
    ConnectionBase, ConnectorBase, LayerBase, NetworkBase)

class Connector(ConnectorBase):
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
    bias_key: str = DefaultLayerKeys.BIAS

@dataclass
class ConvolutionConnector(Connector):
    weights_key: str = DefaultLayerKeys.WEIGHT
    pooling_key: str = DefaultLayerKeys.POOLING
    bias_key: str = DefaultLayerKeys.BIAS


From = TypeVar("From", LayerBase, LayerBase)
To = TypeVar("To", LayerBase, LayerBase)

@dataclass
class Connection(ConnectionBase, Generic[From, To]):
    pre: From
    post: To
    connector: Connector
    network: Optional[NetworkBase]

    def variable(self, channel_in: int, channel_out: int) -> str:
        return sanitize(f"c_{self.pre.name}_{channel_in}__to__{self.post.name}_{channel_out}")
