from dataclasses import dataclass


@dataclass
class Synapse:
    '''
    synapse_type: ['current', 'conductance']
    synapse_shape: ['exponential', 'alpha', 'delta']
    '''
    synapse_type: str
    synapse_shape: str

@dataclass
class StaticSynapse(Synapse):
    pass

@dataclass
class ConvolutionSynapse(StaticSynapse):
    pass

@dataclass
class DenseSynapse(StaticSynapse):
    pass

