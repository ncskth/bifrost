from dataclasses import dataclass
from typing import List, Set, Dict, Any, Tuple

from bifrost.ir.connection import Connection
from bifrost.ir.layer import Layer
from bifrost.ir.bases import NetworkBase


@dataclass
class Network(NetworkBase):
    layers: List[Layer]
    connections: List[Connection]
    runtime: float = -1.0  # Default to infinity
    timestep: float = 1.0  # Default to 1ms, target simulation step (fraction of ms)
    source_dt: float = (
        1.0  # source network (norse, ml_genn) simulation step (fraction of ms)
    )
    configuration: Dict[str, Any] = ()
    # this will be used as the recordings output filename as well
    name: str = "Bifrost Network"
    split_runs: bool = False
