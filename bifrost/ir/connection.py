"""
Internal representations (IR) for internal use
"""
from dataclasses import dataclass
from typing import Generic, TypeVar

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
    ...


@dataclass
class MatrixConnector(Connector):
    weights_key: str = "weights"


@dataclass
class ConvolutionConnector:
    weights_key: str = "weights"
    padding_key: str = "padding"


From = TypeVar("From", Layer, Layer)
To = TypeVar("To", Layer, Layer)


@dataclass
class Connection(Generic[From, To]):
    pre: From
    post: To
    connector: Connector
    synapse: Synapse = StaticSynapse()

    def variable(self, channel: int) -> str:
        return f"c_{self.pre.name}_{self.post.name}_{channel}"
