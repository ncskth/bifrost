from dataclasses import dataclass
from typing import Optional
from bifrost.ir.synapse import Synapse, StaticSynapse
from bifrost.ir.layer import Layer


class OutputSink:
    pass

@dataclass
class EthernetOutput(OutputSink):
    host: Optional[str] = "localhost"
    post: Optional[int] = 3333


@dataclass
class OutputLayer(Layer):
    sink: OutputSink
    synapse: Synapse = StaticSynapse()

    def __repr__(self):
        return super().__repr__()

    def __str__(self):
        return super().__str__()
