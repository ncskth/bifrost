
from bifrost.export.pynn import Statement
from typing import Any, Dict
from bifrost.ir.parameter import ParameterContext, Output


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

__net_params = to_dict( np.load(sys.argv[1], allow_pickle=True) )

    """

    def __init__(self, layer_map: Dict[str, Any]) -> None:
        self.layer_map = layer_map

    def weights(self, layer: str) -> Output:
        raise NotImplementedError()

    def conv2d_weights(self, layer: str, channel_in: int, channel_out: int) -> Output:
        raise NotImplementedError()

    def cell_type(self, layer: str) -> Output:
        return f"__net_params['{layer}']['params']['cell']['target']"

    def cell_parameter_dict(self, layer: str) -> Dict:
        d = {

        }

        return d
