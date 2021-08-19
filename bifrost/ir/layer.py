from bifrost.ir.parameter import ParameterContext
from dataclasses import dataclass
from bifrost.ir.parameters import Conv2dParameters
from bifrost.ir.cell import Cell
from bifrost.ir.connection import Connection
from typing import Dict, List, Set

@dataclass
class Layer:
    index: int
    name: str
    size: int
    cells: List[Cell]
    in_connections: Dict[Layer, List[Connection]]
