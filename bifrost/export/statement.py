from dataclasses import dataclass
from typing import List, Union, Optional, Tuple


@dataclass
class Statement:
    value: str
    # todo: these should be a Set so we don't get repeated imports
    #       or deal with these repeated imports later
    imports: List[str] = ()
    preambles: List[str] = ()

    def __init__(self, value: Union[str, List[str]] = "", imports: List[str] = (),
                 preambles: List[str] = ()):
        if isinstance(value, list):
            self.value = "\n".join(value)
        else:
            self.value = value

        self.imports = imports
        self.preambles = preambles

    def __add__(self, other):
        if isinstance(other, Statement):
            stmt = (f"{self.value}\n") if len(self.value) > 0 else ""
            impos = list(self.imports) + list(other.imports)
            prems = list(self.preambles) + list(other.preambles)
            return Statement(
                f"{stmt}{other.value}",
                imports=impos, preambles=prems
            )
        else:
            raise ValueError("Expected Statement for addition, but found ", other)

    def __repr__(self) -> str:
        imports = "\n".join(self.imports)
        sepi = "\n\n" if len(imports) > 0 else ""
        preambles = "\n".join(self.preambles)
        sepp = "\n\n" if len(preambles) > 0 else ""
        return f"{imports}{sepi}{preambles}{sepp}{self.value}"


@dataclass
class ConnectionStatement(Statement):
    configuration: Optional[str] = ""

    def __add__(self, other):
        return ConnectionStatement(
            f"{self.value}\n{other.value}",
            imports=self.imports + other.imports,
            configuration=f"{self.configuration}\n{other.configuration}",
        )
