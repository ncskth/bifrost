"""
Internal representations (IR) for internal use
"""
from dataclasses import dataclass
from typing import Generic, TypeVar

from bifrost.ir.layer import Layer
from bifrost.text_utils import sanitize


class Connector:
    pass

@dataclass
class AllToAllConnector(Connector):
    """Also Known As DenseConnector"""

@dataclass
class MatrixConnector(Connector):  # todo: is this the same as All-to-All?
    weights_key: str = "layer access key for weights"

@dataclass
class DenseConnector(Connector):
    weights_key: str = "layer access key for weights"
    pooling_key: str = "layer access key for pooling"

@dataclass
class ConvolutionConnector(Connector):
    weights_key: str = "layer access key for weights"
    pooling_key: str = "layer access key for pooling"


From = TypeVar("From", Layer, Layer)
To = TypeVar("To", Layer, Layer)

@dataclass
class Connection(Generic[From, To]):
    pre: From
    post: To
    connector: Connector

    def variable(self, channel_in: int, channel_out: int) -> str:
        return sanitize(f"c_{self.pre.name}_{channel_in}__to__{self.post.name}_{channel_out}")
