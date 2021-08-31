from dataclasses import dataclass
from typing import TypeVar, Generic, Dict, Any


@dataclass
class Cell:
    parameters: Dict[str, Any]

@dataclass
class LIFCell(Cell):
    pass

@dataclass
class LICell(Cell):
    pass

@dataclass
class IFCell(Cell):
    pass


