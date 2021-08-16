from dataclasses import dataclass
from bifrost.ir.parameters import Conv2dParameters
from bifrost.ir.cell import Cell

@dataclass
class Layer:
    index: int
    name: str
    size: int

@dataclass
class Conv2dLayer(Layer):
    cell: Cell
    parameters: Conv2dParameters

@dataclass
class PoolDenseLayer(Layer):
    cell: Cell
    parameters: DenseParameters

@dataclass
class LIFLayer(Layer):
    size: int
    cell: LIFCell
