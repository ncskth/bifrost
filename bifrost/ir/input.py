from dataclasses import dataclass
from bifrost.ir.layer import Layer
from typing import Dict, List, Set, Any


@dataclass
class InputSource:
    shape: List[int]

@dataclass
class ImageDataset(InputSource):
    defines: Dict[int, str] = () # keys are order, thus ints
    # : Dict[int, str]  # keys are order, thus ints
    imports: List[str] = ()
    num_samples: int = 1


@dataclass
class PoissonImageDataset(ImageDataset):
    pass

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

@dataclass
class RandomPoissonSource(InputSource):
    rates: List[int]

class DummyTestInputSource(InputSource):
    pass



@dataclass
class InputLayer(Layer):
    source: InputSource
    record: List[str] = ()

