from dataclasses import dataclass
from bifrost.ir.parameters import (Parameters)


@dataclass
class Cell:
    parameters: Parameters

@dataclass
class LIFCell(Cell):
    pass

@dataclass
class LICell(Cell):
    pass

@dataclass
class IFCell(Cell):
    pass


