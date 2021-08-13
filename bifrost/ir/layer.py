from dataclasses import dataclass
from norse.torch import Conv2dParameters
from bifrost.ir.cell import Cell

@dataclass
class Layer:
    name: str

@dataclass
class Conv2dLayer(Layer):
    cell: Cell
    parameters: Conv2dParameters

@dataclass
class DenseLayer(Layer):
    cell: Cell
    parameters: DenseParameters

@dataclass
class LIFLayer(Layer):
    size: int
    cell: LIFCell
