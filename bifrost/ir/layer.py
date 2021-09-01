from bifrost.ir.parameter import ParameterContext
from dataclasses import dataclass
from bifrost.ir.cell import Cell, LIFCell
from bifrost.ir.synapse import Synapse, StaticSynapse
from typing import Dict, List, Set

@dataclass
class Layer:
    name: str
    size: int
    channels: int

    def variable(self, channel):
        return f"l_{self.name}_{channel}"

    def __repr__(self):
        return f"l_{self.name}_{self.size}_{self.channels}"

    def __str__(self):
        return self.__repr__()


@dataclass
class NeuronLayer(Layer):
    cell: Cell = LIFCell()
    synapse: Synapse = StaticSynapse()
    index: int = 0
    key: str = ""
    shape: List[int] = (1, 1)

    def __repr__(self):
        return f"l_{self.name}_{self.size}_{self.channels}"

    def __str__(self):
        return self.__repr__()
