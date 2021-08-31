from dataclasses import dataclass
from typing import List, Union


@dataclass
class Statement:
    value: str
    # todo: this should be a Set so we don't get repeated imports
    #       or deal with these repeated imports later
    imports: List[str] = ()

    def __init__(self, value: Union[str, List[str]] = "", imports: List[str] = ()):
        if isinstance(value, list):
            self.value = "\n".join(value)
        else:
            self.value = value
        self.imports = imports

    def __add__(self, other):
        if isinstance(other, Statement):
            stmt = (self.value + "\n") if len(self.value) > 0 else ""
            return Statement(
                f"{stmt}{other.value}", imports=self.imports + other.imports
            )
        else:
            raise ValueError("Expected Statement for addition, but found ", other)

    def __repr__(self) -> str:
        imports = "\n".join(self.imports)
        return self.value + (f"\n{imports}" if len(imports) > 0 else "")


def pynn_header(timestep=1.0):
    return f"""
import spynnaker8 as p

p.setup({timestep})
"""


def pynn_footer(runtime):
    return f"""
p.run({runtime})
p.end()
"""
