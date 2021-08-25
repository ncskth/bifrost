from abc import abstractmethod, abstractproperty
from typing import Dict, List, TypeVar, Generic

from bifrost.ir.layer import Layer

Output = TypeVar("Output")


class ParameterContext(Generic[Output]):
    @abstractproperty
    def preamble(self) -> Output:
        ...

    @abstractmethod
    def linear_weights(self, layer: str, channel: int) -> Output:
        ...

    @abstractmethod
    def conv2d_weights(self, layer: str, channel_in: int, channel_out: int) -> Output:
        ...

    @abstractmethod
    def neuron_parameter(self, layer: str, parameter: str) -> Output:
        ...

    @abstractmethod
    def parameter_names(self, layer: Layer) -> List[str]:
        ...