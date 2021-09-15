from enum import Enum

class SynapseTypes(Enum):  # current transfer
    CURRENT = 'current'
    CONDUCTANCE = 'conductance'

class SynapseShapes(Enum): # post-synaptic potential (kernel?) shape
    EXPONENTIAL = 'exponential'
    ALPHA = 'alpha'
    DELTA = 'delta'

class NeuronTypes(Enum):
    LI = 'LI'  # leaky-integrate neuron (DOESN'T FIRE)
    LIF = 'LIF'  # leaky-integrate and fire neuron
    NIF = 'NIF'  # (NON-leaky) integrate and fire neuron