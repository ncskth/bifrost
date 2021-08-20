from bifrost.ir.parameter import ParameterContext
from dataclasses import dataclass
from bifrost.ir.cell import Cell
from typing import Dict, List, Set

@dataclass
class Layer:
    index: int
    key: str
    name: str
    size: int

    @property
    def variable(self):
        return f"layer_{self.name}_{self.index}"


@dataclass
class NeuronLayer(Layer):
    cell: Cell
    n_channels: int = 1
    shape: List[int] = (0, 0)
