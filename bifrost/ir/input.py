from dataclasses import dataclass
from bifrost.ir.layer import Layer
from typing import Dict, List, Set, Any


@dataclass
class InputSource(Layer):
    shape: List[int]

@dataclass
class ImageDataset(InputSource):
    load_command: str
    num_samples: int

@dataclass
class PoissonImageDataset(ImageDataset):
    intensity_to_rate: float


@dataclass
class InputLayer(Layer):
    source: InputSource


@dataclass
class SpiNNakerSPIFInput(InputSource):
    x: int
    y: int
    x_sub: int = 32
    y_sub: int = 16
    x_shift: int = 16
    y_shift: int = 0
