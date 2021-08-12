from dataclasses import dataclass
from .layer import Layer


@dataclass
class InputSource(Layer):
    pass


@dataclass
class InputLayer(Layer):
    source: InputSource


@dataclass
class SpiNNakerSPIFInput(InputSource):
    x: int
    y: int
    x_sub: int = 32
    y_sub: int = 16
    x_shift: int = 16
    y_shift: int = 0
