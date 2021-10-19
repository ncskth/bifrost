from bifrost.ir.layer import NeuronLayer, Layer
from bifrost.ir.cell import Cell, LIFCell, LICell
from bifrost.export.statement import Statement
from typing import Any, Callable, Dict, List
from bifrost.ir.parameter import ParameterContext


class TorchContext(ParameterContext[str]):
    imports = ["import numpy as np", "import torch", "import sys",
               "from bifrost.extract.utils import try_reduce_param"]
    preamble = """

_checkpoint = torch.load(sys.argv[1])
_params = _checkpoint['state_dict']

_param_map = {
    # taus need to be adjusted by dt thus the  * X
    "tau_m": ("tau_mem_inv", lambda v, dt: 1.0/(dt * try_reduce_param(v))),
    "tau_syn_E": ("tau_syn_inv", lambda v, dt: 1.0/(dt * try_reduce_param(v))),
    "tau_syn_I": ("tau_syn_inv", lambda v, dt: 1.0/(dt * try_reduce_param(v))),
    "v_reset": ("v_reset", lambda v, dt: try_reduce_param(v)),
    "v_rest": ("v_leak", lambda v, dt: try_reduce_param(v)),
    "v_thresh": ("v_th", lambda v, dt: try_reduce_param(v)),
    "cm": ("tau_mem_inv", lambda v, dt: 1.0),
    "ioffset": ("bias", lambda v, dt: try_reduce_param(v)),
}
"""

    lif_parameters = [
        "tau_m",
        "cm",
        "tau_syn_E",
        "tau_syn_I",
        "v_reset",
        "v_rest",
        "v_thresh",
    ]

    li_parameters = [
        "tau_m",
        "cm",
        "tau_syn_E",
        "tau_syn_I",
        "v_rest",
    ]

    # lif_parameters = [
    #     "tau_mem_inv",
    #     "tau_syn_inv",
    #     "v_reset",
    #     "v_th",
    #     # "v_leak",
    #     # "alpha"
    # ]
    #
    # li_parameters = [
    #     "tau_mem_inv",
    #     "tau_syn_inv",
    #     # "v_leak",
    # ]
    #
    def __init__(self, layer_map: Dict[str, str]) -> None:
        self.layer_map = layer_map

    def linear_weights(self, layer: str, channel_in: int, num_in_channels: int, num_out_neurons: int) -> str:
        return (f"_params[\"{layer}.weight\"].reshape(({num_out_neurons}, -1, {num_in_channels}))"
                f"[:, :, {channel_in}].detach().numpy()")

    def conv2d_weights(self, layer: str, channel_in: int, channel_out: int) -> str:
        return f"_params[\"{layer}.weight\"][{channel_out}, {channel_in}].detach().numpy()"
        # return f"np.fliplr(np.flipud(_params[\"{layer}.weight\"][{channel_out}, {channel_in}].detach().numpy()))"

    def conv2d_strides(self, layer: str) -> str:
        return f"_params[\"{layer}.stride\"]"

    def conv2d_padding(self, layer: str) -> str:
        return f"_params[\"{layer}.padding\"]"

    def conv2d_pooling(self, layer: str) -> str:
        return (f"_params[\"{layer}.kernel_size\"]", \
                f"_params[\"{layer}.stride\"]")

    def bias_conv2d(self, layer:str, channel: str) -> str:
        return f"_params[\"{layer}.bias\"][{channel}]"

    def bias_dense(self, layer:str) -> str:
        return f"_params[\"{layer}.bias\"]"

    def neuron_parameter_base(self, layer:str) -> str:
        return f"_param_map[{{}}]"\
               f"(_params[\"{self.layer_map[layer]}.{{}}\"])"

    def neuron_parameter(self, layer: str, parameter_name: str) -> str:
        return f"_params[f\"{self.layer_map[layer]}.{{{parameter_name}}}\"]"

    def parameter_map_name(self, parameter_name: str) -> str:
        return f"_param_map[{parameter_name}]"

    def parameter_names(self, cell: Cell) -> List[str]:
        if isinstance(cell, LIFCell):
            return self.lif_parameters
        elif isinstance(cell, LICell):
            return self.li_parameters
        else:
            raise ValueError("Unknown neuron type ", cell)


