from enum import Enum

SIM_NAME = 'spynn'

pynn_imports = [f"import spynnaker8 as {SIM_NAME}"]

def pynn_header(timestep=1.0):
    return f"""
{SIM_NAME}.setup({timestep})
"""


def pynn_footer(runtime):
    return f"""
run_time = {runtime}  # ms
{SIM_NAME}.run(run_time)
{SIM_NAME}.end()
"""


class PyNNSynapseTypes(Enum):  # current transfer
    CURRENT = 'curr'
    CONDUCTANCE = 'cond'

class PyNNSynapseShapes(Enum): # post-synaptic potential (kernel?) shape
    EXPONENTIAL = 'exp'
    ALPHA = 'alpha'
    DELTA = 'delta'

class PyNNNeuronTypes(Enum):
    LI = 'IF'  # leaky-integrate neuron (DOESN'T FIRE)
    LIF = 'IF'  # leaky-integrate and fire neuron
    NIF = 'NIF'  # (NON-leaky) integrate and fire neuron