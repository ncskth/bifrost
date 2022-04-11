from abc import abstractmethod, abstractproperty
from typing import Dict, List, TypeVar, Generic

from bifrost.ir.cell import Cell

Output = TypeVar("Output")


class ParameterContext(Generic[Output]):
    @abstractproperty
    def imports(self) -> List[Output]:
        ...

    @abstractproperty
    def preamble(self) -> Output:
        ...

    # Note: I think these are the all-to-all/dense weights?
    @abstractmethod
    def linear_weights(
        self, layer: str, channel_in: int, num_in_channels: int, num_out_neurons: int
    ) -> Output:
        ...

    @abstractmethod
    def conv2d_weights(self, layer: str, channel_in: int, channel_out: int) -> Output:
        ...

    @abstractmethod
    def conv2d_strides(self, layer: str) -> Output:
        ...

    @abstractmethod
    def conv2d_padding(self, layer: str) -> Output:
        ...

    @abstractmethod
    def conv2d_pooling(self, layer: str) -> Output:
        ...

    @abstractmethod
    def neuron_parameter(self, layer: str, parameter: str) -> Output:
        ...

    @abstractmethod
    def parameter_names(self, layer: Cell) -> List[str]:
        ...
