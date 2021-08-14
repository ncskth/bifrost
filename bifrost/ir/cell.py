from dataclasses import dataclass
from bifrost.ir.parameters import (IFParameters, LIFParameters)


@dataclass
class Cell:
    pass

@dataclass
class LIFCell(Cell):
    parameters: LIFParameters

@dataclass
class IFCell(Cell):
    parameters: IFParameters


