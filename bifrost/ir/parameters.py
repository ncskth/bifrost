from dataclasses import dataclass
from typing import Dict, List, Set, Any, Tuple

@dataclass
class Parameters:
    pass

@dataclass
class Conv2dParameters(Parameters):
    stride: Tuple[int, int] = (1, 1)
    pool_shape: Tuple[int, int] = (1, 1)
    pool_stride: Tuple[int, int] = (1, 1)
    shape: Tuple[int, int]
    channels: int

    @property
    def size(self):
        return int(np.prod(self.shape))

@dataclass
class DenseParameters(Parameters):
    size: int

@dataclass
class LIFParameters(Parameters):
    """These might as well be the target  (PyNN) parameters"""
    tau_m: float = 1.0
    tau_syn_E: float = 1.0
    tau_syn_I: float = 1.0
    v_reset: float = 0.0
    v_rest: float = 0.0
    v_thresh: float = 1.0
