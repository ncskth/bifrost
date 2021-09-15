from dataclasses import dataclass
from typing import Literal
from bifrost.ir.constants import SynapseShapes, SynapseTypes

@dataclass
class Synapse:
    synapse_type: Literal[SynapseTypes.CURRENT,
                          SynapseTypes.CONDUCTANCE] = SynapseTypes.CURRENT
    synapse_shape: Literal[SynapseShapes.EXPONENTIAL,
                           SynapseShapes.ALPHA,
                           SynapseShapes.DELTA] = SynapseShapes.EXPONENTIAL

@dataclass
class StaticSynapse(Synapse):
    pass

@dataclass
class ConvolutionSynapse(StaticSynapse):
    pass

@dataclass
class DenseSynapse(StaticSynapse):
    pass

