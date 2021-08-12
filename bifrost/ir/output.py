from dataclasses import dataclass
from typing import Optional

from .layer import Layer


class OutputSource:
    pass


@dataclass
class OutputLayer(Layer):
    sink: OutputSource


@dataclass
class EthernetOutput(OutputSource):
    host: Optional[str] = "localhost"
    post: Optional[int] = 3333
