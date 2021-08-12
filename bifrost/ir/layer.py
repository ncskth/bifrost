from dataclasses import dataclass
from norse.torch import LIFParameters


@dataclass
class Layer:
    name: str


@dataclass
class Conv2dLIFLayer(Layer):
    width: int
    height: int
    channels: int
    parameters: LIFParameters = LIFParameters()


@dataclass
class LIFLayer(Layer):
    neurons: int
    parameters: LIFParameters = LIFParameters()
