from dataclasses import dataclass
from typing import TypeVar, Generic, Dict, Any


@dataclass
class Cell:
    # parameter_names: List[str] = ()
    # function_name = ""
    pass

@dataclass
class LIFCell(Cell):
    pass

@dataclass
class LICell(LIFCell):
    pass

@dataclass
class IFCell(Cell):
    pass


