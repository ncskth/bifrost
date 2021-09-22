from dataclasses import dataclass
from bifrost.ir.layer import Layer
from typing import Dict, List, Set, Any


@dataclass
class InputSource:
    shape: List[int]

@dataclass
class ImageDataset(InputSource):
    defines: Dict[int, str]  # keys are order, thus ints
    imports: List[str]
    load_command_body: str
    start_sample: int
    num_samples: int
    on_time_ms: float  # milliseconds
    off_time_ms: float  # milliseconds

    @property
    def start_sample_variable(self):
        return "__start_sample"

    @property
    def num_samples_variable(self):
        return "__num_samples"

    @property
    def images_variable(self):
        return "__images_dictionary"

    @property
    def classes_variable(self):
        return "__classes"


@dataclass
class PoissonImageDataset(ImageDataset):
    pixel_to_rate_transform: str

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
    # this is just needed for testing and throwing non-known source types
    pass


@dataclass
class InputLayer(Layer):
    source: InputSource
    record: List[str] = ()

    @property
    def num_channels_variable(self):
        return "__n_channels"

