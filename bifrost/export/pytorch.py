
from bifrost.export.pynn import Statement
from typing import Any, Dict
from bifrost.ir.parameter import ParameterContext


class PytorchLightningContext(ParameterContext[str]):

    preamble = """
import torch
import sys

_checkpoint = torch.load(sys.argv[1])
_params = _checkpoint['state_dict']
    """

    def __init__(self, layer_map: Dict[str, str]) -> None:
        self.layer_map = layer_map

    def weights(self, key: str) -> str:
        raise NotImplementedError()

    def conv2d_weights(self, key: str, channel_in: int, channel_out: int) -> str:
        raise NotImplementedError()
