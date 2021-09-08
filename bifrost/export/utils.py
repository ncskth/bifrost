from typing import Dict, Any, List
from bifrost.ir.layer import Layer
from bifrost.export.statement import Statement


def export_dict(d: Dict[Any, Any], join_str=",\n", n_spaces=0) -> Statement:
    def _export_dict_key(key: Any) -> str:
        if not isinstance(key, str):
            raise ValueError("Parameter key must be a string", key)
        return str(key)

    def _export_dict_value(value: Any) -> str:
        if isinstance(value, str):
            return f"'{str(value)}'"
        else:
            return str(value)

    pynn_dict = [f"{_export_dict_key(key)}={_export_dict_value(value)}"
                 for key, value in d.items()]
    spaces = " " * n_spaces
    return Statement((f"{join_str}{spaces}").join(pynn_dict), [])


def export_list(var: str, l: List[str], join_str=", ", n_spaces=0):
    spaces = " " * n_spaces
    lst = (f"{join_str}{spaces}").join([f"\"{v}\"" for v in l])

    return f"{var} = [{lst}]"


def export_structure(layer: Layer) -> Statement:
    ratio = float(layer.shape[1]) / layer.shape[0]
    return Statement(f"Grid2D({ratio})",
                     imports=['from pyNN.space import Grid2D'])