from dataclasses import dataclass
from typing import List, Set

from bifrost.ir.connection import Connection
from bifrost.ir.layer import Layer


@dataclass
class Network:
    layers: List[Layer]
    connections: Set[Connection]
    runtime: float = -1.0  # Default to infinity
    timestep: float = 1.0  # Default to 1ms
    config: List[str] = ()
