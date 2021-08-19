
from bifrost.export.pynn import Statement
from typing import Any, Dict
from bifrost.ir.parameter import ParameterContext


class MLGeNNContext(ParameterContext[str]):

    preamble = """
import numpy as np
import sys

__network_params = np.load(sys.argv[1], allow_pickle=True)
__all_populations = {{}}
__all_connections = {{}}
__all_projections = {{}}

    """

    def __init__(self, layer_map: Dict[str, str]) -> None:
        self.layer_map = layer_map

    def weights(self, layer: str) -> Output:
        raise NotImplementedError()

    def conv2d_weights(self, layer: str, channel_in: int, channel_out: int) -> Output:
        raise NotImplementedError()

    def cell(self, layer: str) -> Output:
        raise NotImplementedError()