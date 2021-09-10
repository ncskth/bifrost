from bifrost.ir.layer import NeuronLayer, Layer
from bifrost.ir.cell import Cell, LIFCell
from bifrost.export.statement import Statement
from typing import Any, Callable, Dict, List
from bifrost.ir.parameter import ParameterContext


class TorchContext(ParameterContext[str]):
    imports = ["import numpy as np", "import torch", "import sys"]
    preamble = """
_checkpoint = torch.load(sys.argv[1])
_params = _checkpoint['state_dict']


_param_map = {
    "tau_mem_inv": lambda v: ("tau_m", 1.0/v),
    "tau_syn_inv": lambda v: ("tau_syn_E", 1.0/v),
    "v_reset": lambda v: ("v_reset", v),
    "v_th": lambda v: ("v_thresh", v),
}
"""

    lif_parameters = [
        "tau_mem_inv",
        "tau_syn_inv",
        "v_reset",
        "v_th",
    ]

    def __init__(self, layer_map: Dict[str, str]) -> None:
        self.layer_map = layer_map

    def linear_weights(self, layer: str, channel_in: int, channel_out: int) -> str:
        return f"_params[\"{layer}.weight\"][{channel_out}, {channel_in}]"

    def conv2d_weights(self, layer: str, channel_in: int, channel_out: int) -> str:
        return f"_params[\"{layer}.weight\"][{channel_out}, {channel_in}]"

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
        else:
            raise ValueError("Unknown neuron type ", cell)


