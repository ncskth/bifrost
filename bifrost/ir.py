from dataclasses import dataclass
from typing import Dict, List

import torch

from norse.torch import LIFParameters

@dataclass
class PyNNProjection:
    pre: str
    post: str

@dataclass
class PyNNPopulation:
    label: str
    cell: str
    
@dataclass
class LIFPopulation(PyNNPopulation):
    parameters: LIFParameters

@dataclass
class PyNNProgram:
    populations: Dict[str, PyNNPopulation]
    projections: List[PyNNProjection]

