from dataclasses import dataclass
from bifrost.ir.layer import Layer
from typing import Dict, List, Set, Any


@dataclass
class InputSource:
    shape: List[int]

@dataclass
class ImageDataset(InputSource):
    load_command: str
    num_samples: int

@dataclass
class PoissonImageDataset(ImageDataset):
    intensity_to_rate: float

@dataclass
class SpiNNakerSPIFInput(InputSource):
    x_sub: int = 32
    y_sub: int = 16
    x_shift: int = 16
    y_shift: int = 0

    @property
    def x(self):
        return self.shape[1]

    @property
    def y(self):
        return self.shape[0]

class DummyTestInputSource(InputSource):
    pass

@dataclass
class InputLayer(Layer):
    source: InputSource

