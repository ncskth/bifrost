from dataclasses import dataclass
from typing import Dict, List, Set, Any, TypeVar, Generic



@dataclass
class Parameters:
    pass

@dataclass
class Conv2dParameters(Parameters):
    stride: List[int, int] = (1, 1)
    pool_shape: List[int, int] = (1, 1)
    pool_stride: List[int, int] = (1, 1)
    shape: List[int, int]
    channels: int

@dataclass
class DenseParameters(Parameters):
    pool_shape: List[int, int] = (1, 1)
    pool_stride: List[int, int] = (1, 1)

@dataclass
class LIFParameters(Parameters):
    """These might as well be the target  (PyNN) parameters"""
    tau_m: float = 1.0
    tau_syn_E: float = 1.0
    tau_syn_I: float = 1.0
    v_reset: float = 0.0
    v_rest: float = 0.0
    v_thresh: float = 1.0

@dataclass
class IFParameters(Parameters):
    """These might as well be the target  (PyNN) parameters"""
    v_reset: float = 0.0
    v_rest: float = 0.0
    v_thresh: float = 1.0

