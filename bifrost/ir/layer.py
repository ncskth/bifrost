from dataclasses import dataclass
from bifrost.ir.cell import Cell
from bifrost.ir.synapse import Synapse
from typing import Dict, List, Set

@dataclass
class Layer:
    index: int
    key: str
    name: str
    size: int
    channels: int

    def variable(self, channel):
        return f"l_{self.name}_{channel}"


@dataclass
class NeuronLayer(Layer):
    cell: Cell
    synapse: Synapse
    n_channels: int = 1
    shape: List[int] = (1, 1)
