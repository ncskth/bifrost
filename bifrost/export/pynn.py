from enum import Enum
from typing import Callable
from bifrost.export.statement import Statement
from bifrost.ir.layer import Layer, NeuronLayer
from bifrost.ir.input import ImageDataset
from bifrost.ir.network import Network
from bifrost.text_utils import TAB

SIMULATOR_NAME = "spynn"

pynn_imports = [f"import spynnaker8 as {SIMULATOR_NAME}"]

def pynn_header(timestep=1.0):
    return f"{SIMULATOR_NAME}.setup({timestep})\n"

def pynn_runner(runtime: float, tab: str = "") -> str:
    return (f"{tab}run_time = {runtime}  # ms\n"
            f"{tab}{SIMULATOR_NAME}.run(run_time)\n")

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


def export_split_run(network: Network, runtime: float,
                     runner_function: Callable[[float], str] = pynn_runner) -> Statement:
    in_layer = network.layers[0]
    source = in_layer.source
    # so far, only ImageDataset inputs can benefit from splitting the run into
    # multiple chuncks (one per input image sample) and resetting neuron values
    if not isinstance(source, ImageDataset):
        return Statement(runner_function(runtime))

    reset_loop = f"{TAB}for channel in {{}}:"
    set_var = f"{TAB}{TAB}{{}}[channel].set({{}}={{}})"

    # this looks ugly, not as clean as list comprehension BUT, we could have
    # different cell types per layer, thus resetting of different state variables
    reset_each_pop = []
    for layer in network.layers[1:]:
        cell = layer.cell
        if len(cell.reset_variables_values) == 0:
            continue

        layer_name = layer.variable('')
        var_setting = []
        for variable_name, reset_value in cell.reset_variables_values:
            setting_text = set_var.format(layer_name, variable_name, reset_value)
            var_setting.append(setting_text)
        var_setting.insert(0, reset_loop.format(layer_name))

        reset_each_pop.append("\n".join(var_setting))

    resets_text = "\n\n".join(reset_each_pop)

    new_runtime = runtime / source.num_samples
    run_text = runner_function(new_runtime, TAB)

    statement_text = (
        f"for sample_id in range({source.num_samples_variable}):\n"
        f"{run_text}\n"
        f"{resets_text}\n"
    )

    return Statement(statement_text)
