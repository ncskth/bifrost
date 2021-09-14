from bifrost.ir.parameter import ParameterContext
from dataclasses import dataclass
from bifrost.ir.cell import Cell, LIFCell
from bifrost.ir.synapse import Synapse, StaticSynapse
from typing import Dict, List, Set
from bifrost.ir.sanitize import sanitize

@dataclass
class Layer:
    name: str
    size: int
    channels: int

    def variable(self, channel):
        return sanitize(f"l_{self.name}_{self.size}_{channel}")

    def __repr__(self):
        return sanitize(f"l_{self.name}_{self.size}_{self.channels}")

    def __str__(self):
        return self.__repr__()


@dataclass
class NeuronLayer(Layer):
    cell: Cell = LIFCell()
    synapse: Synapse = StaticSynapse()
    index: int = 0  # todo: not sure we use this anymore
    # note: it's easier to store layer keys here than the map in parameter context
    key: str = ""
    shape: List[int] = (1, 1)
    record: List[str] = ()

    def __repr__(self):
        return super().__repr__()

    def __str__(self):
        return super().__str__()
