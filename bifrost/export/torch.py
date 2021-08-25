from bifrost.ir.layer import LIFAlphaLayer, Layer
from bifrost.export.pynn import Statement
from typing import Any, Callable, Dict, List
from bifrost.ir.parameter import ParameterContext


class TorchContext(ParameterContext[str]):

    preamble = """
import torch
import sys

_checkpoint = torch.load(sys.argv[1])
_params = _checkpoint['state_dict']

_param_map = {
    "tau_mem_inv": (lambda v: f"tau_m=1/{v}"),
    "tau_syn_inv": (lambda v: f"tau_syn_E=1/{v}"),
    "tau_syn_inv": (lambda v: f"tau_syn_I=1/{v}"),
    "v_reset": (lambda v: f"v_reset{v}"),
    "v_th": (lambda v: f"v_thresh={v}"),
}
"""

    lif_parameters = [
        "tau_mem_inv",
        "tau_syn_inv",
        "tau_syn_inv",
        "v_reset",
        "v_th",
    ]

    def __init__(self, layer_map: Dict[str, str]) -> None:
        self.layer_map = layer_map

    def weights(self, layer: str, channel: int) -> str:
        return f"_params['{self.layer_map[layer]}'][:{channel}]"

    def conv2d_weights(self, key: str, channel_in: int, channel_out: int) -> str:
        raise NotImplementedError()

    def neuron_parameter(self, layer: Layer, parameter_name: str) -> str:
        return f"_param_map['{parameter_name}'](_params['{self.layer_map[layer.name]}'][{parameter_name}])"

    def parameter_names(self, layer: Layer) -> List[str]:
        if isinstance(layer, LIFAlphaLayer):
            return self.lif_parameters
        else:
            raise ValueError("Unknown neuron layer type ", layer)
