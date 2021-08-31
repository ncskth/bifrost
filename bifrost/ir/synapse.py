from dataclasses import dataclass
from typing import Literal


@dataclass
class Synapse:
    synapse_type: Literal['current', 'conductance'] = 'current'
    synapse_shape: Literal['exponential', 'alpha', 'delta'] = 'exponential'

@dataclass
class StaticSynapse(Synapse):
    pass

@dataclass
class ConvolutionSynapse(StaticSynapse):
    pass

@dataclass
class DenseSynapse(StaticSynapse):
    pass

