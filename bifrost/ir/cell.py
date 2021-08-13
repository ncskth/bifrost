from dataclasses import dataclass
from bifrost.ir.parameters import LIFParameters


@dataclass
class Cell:
    pass

@dataclass
class LIFCell(Cell):
    parameters: LIFParameters