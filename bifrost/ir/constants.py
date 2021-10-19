from enum import Enum

class SynapseTypes(Enum):  # current transfer
    CURRENT:str = "current"
    CONDUCTANCE:str = "conductance"

class SynapseShapes(Enum): # post-synaptic potential (kernel?) shape
    EXPONENTIAL:str = "exponential"
    ALPHA:str = "alpha"
    DELTA:str = "delta"

class NeuronTypes(Enum):
    LI:str = "LI"  # leaky-integrate neuron (DOESN"T FIRE)
    LIF:str = "LIF"  # leaky-integrate and fire neuron
    NIF:str = "NIF"  # (NON-leaky) integrate and fire neuron

class DefaultLayerKeys(Enum):
    POOLING:str = "parameter context layer key for pooling"
    STRIDE:str = "parameter context layer key for strides"
    WEIGHT:str = "parameter context layer key for weights"
    BIAS:str = "parameter context layer key for bias"
