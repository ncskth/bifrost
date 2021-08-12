from dataclasses import dataclass
from typing import Dict, List, Set

from bifrost.ir.connection import Connection
from bifrost.ir.layer import Layer


@dataclass
class Network:
    layers: List[Layer]
    connections: Set[Connection]
    runtime: float
    timestep: float = 1.0
    config: List[str] = ()
