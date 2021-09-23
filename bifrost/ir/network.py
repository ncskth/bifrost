from dataclasses import dataclass
from typing import List, Set, Dict, Any, Tuple

from bifrost.ir.connection import Connection
from bifrost.ir.layer import Layer


@dataclass
class Network:
    layers: List[Layer]
    connections: Set[Connection]
    runtime: float = -1.0  # Default to infinity
    timestep: float = 1.0  # Default to 1ms
    configuration: Dict[str, Any] = ()
    # this will be used as the recordings output filename as well
    name: str = "Bifrost Network"
    split_runs: bool = False
