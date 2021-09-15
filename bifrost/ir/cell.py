from dataclasses import dataclass
from typing import TypeVar, Generic, Dict, Any


@dataclass
class Cell:
    pass

@dataclass
class LIFCell(Cell):
    """Leaky integrate-and-fire neuron type"""
    pass

@dataclass
class LICell(Cell):
    """Leaky integrate neuron type (not-firing)"""
    pass

@dataclass
class IFCell(Cell):
    """Integrate and fire neuron type (non-leaky)"""
    pass


