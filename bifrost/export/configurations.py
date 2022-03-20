import copy
import warnings
from warnings import warn
from typing import Dict, Any, List, Tuple, Union
from enum import Enum
from bifrost.text_utils import TAB
from bifrost.export.pynn import SIMULATOR_NAME
from bifrost.export.statement import Statement
from bifrost.ir.bases import NetworkBase


class SUPPORTED_CONFIGS(Enum):
    RUNTIME: str = 'runtime'
    TIMESTEP: str = 'timestep'
    SPLIT_RUNS: str = 'split_runs'
    MAX_NEURONS_PER_COMPUTE_UNIT: str = 'max_neurons_per_compute_unit'  # e.g. core, chip
    MAX_NEURONS_PER_LAYER_TYPE: str = 'max_neurons_per_layer_type'  # dense, conv2d
    # MAX_NEURONS_PER_LAYER: str = 'max_neurons_per_layer'  # independent of type

def export_configurations(network:NetworkBase,
        configurations: Dict[str, Any]) -> Statement:
    # todo: if we support multiple output simulator front-ends, this will have
    #  to become a per-platform export
    statement = Statement()
    _cfgs = copy.deepcopy(configurations)
    if (SUPPORTED_CONFIGS.MAX_NEURONS_PER_COMPUTE_UNIT in _cfgs and
        SUPPORTED_CONFIGS.MAX_NEURONS_PER_LAYER_TYPE in _cfgs ):
        warnings.warn(
            "We do not support setting both 'global' and layer-wise "
            "maximum neurons per core: "
            f"{SUPPORTED_CONFIGS.MAX_NEURONS_PER_COMPUTE_UNIT} & "
            f"{SUPPORTED_CONFIGS.MAX_NEURONS_PER_LAYER_TYPE}. "
            "We will remove the global constraint.")
        del _cfgs[SUPPORTED_CONFIGS.MAX_NEURONS_PER_COMPUTE_UNIT]

    for config in _cfgs:
        if config == SUPPORTED_CONFIGS.MAX_NEURONS_PER_COMPUTE_UNIT:
            statement += export_max_neurons_per_core(
                            network, configurations[config])
        elif config == SUPPORTED_CONFIGS.MAX_NEURONS_PER_LAYER_TYPE:
            statement += export_max_neurons_per_layer_type(
                            network, configurations[config])
        else:
            warn(f"Configuration {config} is not supported!"
                 "Passing as is :O")
            statement += Statement(config)

    return statement


# note: this is even a sPyNNaker-only setting
#  this tells the front-end how many neurons (maximum) are allowed to be
#  simulated per ARM core (NOT per SpiNNaker chip)
def export_max_neurons_per_core(network:NetworkBase,
        configuration: List[Tuple[str, Union[int, Tuple[int]]]]) -> Statement:
    """
    :param configuration: a list of tuples containing:
        . neuron types (e.g. IF_curr_exp, IZK_curr_delta, etc.)
        . int or tuples of ints which are the shape of the population [e.g. 12,
        (3, 4), etc.]
    :return Statement: A Statement object whose value text which will contain
        the instructions required to constraint/assign the shape the required
        sub-populations (we have to split-down populations in chunks that fit
        into each SpiNNaker ARM core)
    """
    template = f"{SIMULATOR_NAME}.set_number_of_neurons_per_core({SIMULATOR_NAME}.{{}}, {{}})"
    return Statement(
        [template.format(cell_name, limit) for cell_name, limit in configuration] + [""]
    )


def export_max_neurons_per_layer_type(network:NetworkBase,
                      configuration: List[Tuple[str, Union[int, Tuple[int]]]]):
    """
    :param network: an IR version of the network which is used to iterate through
        neuron populations
    :param configuration: a list of tuples containing:
        . neuron types (e.g. IF_curr_exp, IZK_curr_delta, etc.)
        . int or tuples of ints which are the shape of the population [e.g. 12,
        (3, 4), etc.]
    :return Statement: A Statement object whose value text which will contain
        the instructions required to constraint/assign the shape the required
        sub-populations (we have to split-down populations in chunks that fit
        into each SpiNNaker ARM core)
    """
    _trans = {
        'conv2d': 'convolution',
        'dense': 'dense',
        'input': 'input',
    }
    template = ("for i in {0}:\n"
                f"{TAB}{{0}}[i].set_max_atoms_per_core({{1}})\n")

    stt = Statement()
    for constraint_layer_type, constraint_shape in configuration:
        if constraint_layer_type not in _trans:
            warnings.warn(
                f"Not supported per layer number of neurons constraint "
                f"{constraint_layer_type}.")
            continue

        ir_layer_type = _trans[constraint_layer_type]
        for lyr in network.layers:
            t = lyr.__class__.__name__.lower().replace('layer', '')
            if t == 'input' and t == ir_layer_type:
                var_name = lyr.variable('')
                stt_text = template.format(var_name, constraint_shape)
                stt += Statement(stt_text)

            elif t == 'neuron':
                in_conn = lyr.synapse.__class__.__name__.lower().replace('synapse', '')
                if in_conn != ir_layer_type:
                    continue

                var_name = lyr.variable('')
                stt_text = template.format(var_name, constraint_shape)
                stt += Statement(stt_text)

    stt += Statement("")  # add a carriage return to separate this block
    return stt
