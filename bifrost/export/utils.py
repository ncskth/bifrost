from typing import Dict, Any, List
from bifrost.export.statement import Statement
from bifrost.ir import Layer, InputLayer, NeuronLayer


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

    pynn_dict = [
        f"{_export_dict_key(key)}={_export_dict_value(value)}"
        for key, value in d.items()
    ]
    spaces = " " * n_spaces
    return Statement((f"{join_str}{spaces}").join(pynn_dict), [])


def export_list_var(var: str, l: List[str], join_str=", ", n_spaces=0):
    lst = export_list(l, join_str, n_spaces)
    return f"{var} = [{lst}]"


def export_list(l: List[str], join_str=", ", n_spaces=0, q='"'):
    spaces = " " * n_spaces
    return (f"{join_str}{spaces}").join([f"{q}{v}{q}" for v in l])


def export_layer_shape(layer: Layer) -> Statement:
    if isinstance(layer, InputLayer):
        return Statement(f"{tuple(layer.source.shape)}")
    elif isinstance(layer, NeuronLayer):
        return Statement(f"{tuple(layer.shape)}")
