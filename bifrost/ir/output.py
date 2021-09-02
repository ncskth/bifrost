from dataclasses import dataclass
from typing import Optional
from bifrost.ir.synapse import Synapse, StaticSynapse
from bifrost.ir.layer import Layer


class OutputSource:
    pass


@dataclass
class OutputLayer(Layer):
    sink: OutputSource
    synapse: Synapse = StaticSynapse()


@dataclass
class EthernetOutput(OutputSource):
    host: Optional[str] = "localhost"
    post: Optional[int] = 3333
