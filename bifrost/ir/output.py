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

class DummyTestOutputSink(OutputSink):
    # this is just needed for testing and throwing non-known source types
    pass


@dataclass
class OutputLayer(Layer):
    sink: OutputSink
    source: Layer = None
    key: str = ""
    synapse: Synapse = StaticSynapse()

    def __repr__(self):
        return super().__repr__()

    def __str__(self):
        return super().__str__()
