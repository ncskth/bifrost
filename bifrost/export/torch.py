from bifrost.ir.layer import NeuronLayer, Layer
from bifrost.ir.cell import Cell, LIFCell, LICell
from bifrost.export.statement import Statement
from typing import Any, Callable, Dict, List
from bifrost.ir.parameter import ParameterContext


class TorchContext(ParameterContext[str]):
    imports = ["import numpy as np", "import torch", "import sys"]
    preamble = """
def try_reduce_param(param):
    try:
        if np.allclose(param[:1], param):
            return np.asscalar(param[0])
    except Exception as e:
        if np.ndim(param) == 0:
            return param.item()
    else:
        return param

_checkpoint = torch.load(sys.argv[1])
_params = _checkpoint['state_dict']

_param_map = {
    "tau_mem_inv": lambda v: ("tau_m", 1.0/try_reduce_param(v)),
    "tau_syn_inv": lambda v: ("tau_syn_E", 1.0/try_reduce_param(v)),
    "v_reset": lambda v: ("v_reset", try_reduce_param(v)),
    "v_th": lambda v: ("v_thresh", try_reduce_param(v)),
}
"""

    lif_parameters = [
        "tau_mem_inv",
        "tau_syn_inv",
        "v_reset",
        "v_th",
        # "v_leak",
        # "alpha"
    ]

    li_parameters = [
        "tau_mem_inv",
        "tau_syn_inv",
        # "v_leak",
    ]

    def __init__(self, layer_map: Dict[str, str]) -> None:
        self.layer_map = layer_map

    def linear_weights(self, layer: str, channel_in: int, num_in_channels: int, num_out_neurons: int) -> str:
        return (f"_params[\"{layer}.weight\"].reshape(({num_out_neurons}, -1, {num_in_channels}))"
                f"[:, :, {channel_in}].detach().numpy()")

    def conv2d_weights(self, layer: str, channel_in: int, channel_out: int) -> str:
        return f"_params[\"{layer}.weight\"][{channel_out}, {channel_in}].detach().numpy()"

    def conv2d_strides(self, layer: str) -> str:
        return f"_params[\"{layer}.stride\"]"

    def conv2d_padding(self, layer: str) -> str:
        return f"_params[\"{layer}.padding\"]"

    def conv2d_pooling(self, layer: str) -> str:
        return (f"_params[\"{layer}.kernel_size\"]", \
                f"_params[\"{layer}.stride\"]")

    def neuron_parameter_base(self, layer:str) -> str:
        return f"_param_map[{{}}]"\
               f"(_params[\"{self.layer_map[layer]}.{{}}\"])"

    def neuron_parameter(self, layer: str, parameter_name: str) -> str:
        return f"_param_map[{parameter_name}]"\
               f"(_params[f\"{self.layer_map[layer]}.{{{parameter_name}}}\"])"

    def parameter_names(self, cell: Cell) -> List[str]:
        if isinstance(cell, LIFCell):
            return self.lif_parameters
        elif isinstance(cell, LICell):
            return self.li_parameters
        else:
            raise ValueError("Unknown neuron type ", cell)


