from typing import Dict, Any, List, Tuple, Union
from enum import Enum
from bifrost.export.pynn import SIMULATOR_NAME
from bifrost.export.statement import Statement


class SUPPORTED_CONFIGS(Enum):
    RUNTIME: str = 'runtime'
    TIMESTEP: str = 'timestep'
    SPLIT_RUNS: str = 'split_runs'
    MAX_NEURONS_PER_COMPUTE_UNIT: str = 'max_neurons_per_compute_unit'  # i.e. core, chip

def export_configurations(configurations: Dict[str, Any]) -> Statement:
    # todo: if we support multiple output simulator front-ends, this will have
    #  to become a per-platform export
    statement = Statement()
    for config in configurations:
        if config in SUPPORTED_CONFIGS:
            statement += export_max_neurons_per_core(configurations[config])
        else:
            warn(f"Configuration {config} is not supported!")

    return statement


# note: this is even a sPyNNaker-only setting
#  this tells the front-end how many neurons (maximum) are allowed to be
#  simulated per ARM core (NOT per SpiNNaker chip)
def export_max_neurons_per_core(configuration: List[Tuple[str, Union[int, Tuple[int]]]]) -> Statement:
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