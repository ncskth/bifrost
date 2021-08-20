from bifrost.ir.parameter import ParameterContext
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class Statement:
    value: str
    imports: List[str] = ()
    preamble: str = ""


def pynn_header(timestep=1.0):
    return f"""
import spynnaker8 as p

__all_populations = {{}}
__all_connections = {{}}
__all_projections = {{}}

p.setup({timestep})
"""


def pynn_footer(runtime):
    return f"""
p.run({runtime})
p.end()
"""

