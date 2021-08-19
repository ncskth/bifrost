from abc import abstractmethod, abstractproperty
from dataclasses import dataclass
from typing import TypeVar, Generic

Output = TypeVar("Output")


class ParameterContext(Generic[Output]):
    @abstractproperty
    def preamble(self) -> Output:
        ...

    @abstractmethod
    def weights(self, layer: str) -> Output:
        ...

    @abstractmethod
    def conv2d_weights(self, layer: str, channel_in: int, channel_out: int) -> Output:
        ...
