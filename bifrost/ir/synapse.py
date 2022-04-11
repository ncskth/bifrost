from dataclasses import dataclass
import sys

version = sys.version_info
if version.major >= 3 and version.minor >= 8:
    from typing import Literal
else:
    from typing_extensions import Literal
from bifrost.ir.constants import SynapseShapes, SynapseTypes


@dataclass
class Synapse:
    synapse_type: Literal[
        SynapseTypes.CURRENT, SynapseTypes.CONDUCTANCE
    ] = SynapseTypes.CURRENT
    synapse_shape: Literal[
        SynapseShapes.EXPONENTIAL, SynapseShapes.ALPHA, SynapseShapes.DELTA
    ] = SynapseShapes.EXPONENTIAL


@dataclass
class StaticSynapse(Synapse):
    pass


@dataclass
class ConvolutionSynapse(StaticSynapse):
    pass


@dataclass
class DenseSynapse(StaticSynapse):
    pass
