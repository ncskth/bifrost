from bifrost.export.statement import Statement
from typing import Any, Dict, List
from bifrost.ir.parameter import ParameterContext, Output
from bifrost.ir.cell import Cell, IFCell


class MLGeNNContext(ParameterContext[str]):

    preamble = """
import numpy as np
import sys
def to_dict(np_file):
    d = {}
    for k in np_file.keys():
        try:
            d[k] = np_file[k].item()
        except:
            d[k] = np_file[k]
    return d 

_params = to_dict( np.load(sys.argv[1], allow_pickle=True) )

_param_map = {
    "v_rest": (lambda v: "v_rest", v),
    "v_thresh": (lambda v: "v_thresh", v),
}
    """

    if_parameters = ['v_thresh', 'v_rest']

    def __init__(self, layer_map: Dict[str, Any]) -> None:
        self.layer_map = layer_map

    def weights(self, layer: str) -> Output:
        raise NotImplementedError()

    def conv2d_weights(self, layer: str, channel_in: int, channel_out: int) -> Output:
        raise NotImplementedError()

    def cell_type(self, layer: str) -> Output:
        return f"_params['{layer}']['params']['cell']['target']"

    def cell_parameter_dict(self, layer: str) -> Dict:
        return {}

    def neuron_parameter(self, layer: str, parameter_name: str) -> str:
        return f'_param_map[{parameter_name}]'\
               f'(_params["{self.layer_map[layer]}"]["params"]["cell"][{parameter_name}])'

    def parameter_names(self, cell: Cell) -> List[str]:
        if isinstance(cell, IFCell):
            return self.if_parameters
        else:
            raise ValueError("Unknown neuron type ", cell)
