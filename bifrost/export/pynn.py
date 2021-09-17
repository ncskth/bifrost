from enum import Enum
from typing import Dict, Tuple, List, Any
from bifrost.export.statement import Statement
from bifrost.ir import Layer

SIMULATOR_NAME = "spynn"

pynn_imports = [f"import spynnaker8 as {SIMULATOR_NAME}"]

def pynn_header(timestep=1.0):
    return f"{SIMULATOR_NAME}.setup({timestep})\n"

def pynn_runner(runtime):
    return (f"run_time = {runtime}  # ms\n"
            f"{SIMULATOR_NAME}.run(run_time)\n")

def pynn_footer():
    return f"{SIMULATOR_NAME}.end()\n"


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


def export_structure(layer: Layer) -> Statement:
    ratio = float(layer.shape[1]) / layer.shape[0]
    return Statement(f"Grid2D({ratio})",
                     imports=['from pyNN.space import Grid2D'])


def export_constraints(constraints: Dict[str, Any]) -> Statement:
    if 'max_neurons' in constraints:
        return export_max_neurons_per_core(constraints['max_neurons'])

    return Statement()


def export_max_neurons_per_core(constraints: List[Tuple[str, Any]]) -> Statement:
    template = f"{SIMULATOR_NAME}.set_number_of_neurons_per_core({SIMULATOR_NAME}.{{}}, {{}})"
    return Statement(
        [template.format(cell_name, limit) for cell_name, limit in constraints] + [""]
    )
