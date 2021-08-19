from typing import Any, Dict
from norse.torch.functional.lif import LIFParameters
from torch._C import Value
from bifrost.ir.layer import LIFLayer, Layer, Conv2dLIFLayer
from bifrost.ir.input import SpiNNakerSPIFInput
from bifrost.ir.parameter import ParameterContext
from .pynn import Statement


def export_layer(layer: Layer, context: ParameterContext[str]) -> Statement:
    if isinstance(layer, SpiNNakerSPIFInput):
        return export_layer_spif(layer, context)
    elif isinstance(layer, Conv2dLIFLayer):
        return export_layer_conv2d(layer, context)
    elif isinstance(layer, LIFLayer):
        return export_layer_lif(layer, context)
    else:
        raise ValueError("Unknown layer type", layer)


def export_layer_lif(layer: LIFLayer, context: ParameterContext[str]) -> Statement:
    lif_p = export_lif_neuron_type(layer.parameters)
    return Statement(
        f"{layer.variable} = p.Population({layer.neurons}, {lif_p.value})",
        imports=lif_p.imports,
    )


def export_layer_spif(layer: SpiNNakerSPIFInput) -> Statement:
    return Statement(
        f"""{layer.variable} = p.Population(None, p.external_devices.SPIFRetinaDevice(\
base_key=0, width={layer.x}, height={layer.y}, sub_width={layer.x_sub}, sub_height={layer.y_sub},\
input_x_shift={layer.x_shift}, input_y_shift={layer.y_shift}))"""
    )


def export_layer_conv2d(
    layer: Conv2dLIFLayer, context: ParameterContext[str]
) -> Statement:
    lif = export_lif_neuron_type(layer.parameters)
    return Statement(
        f"{layer.variable} = p.Population({layer.width * layer.height}, {lif.value}, structure=Grid2D({layer.width / layer.height}))",
        ["from pyNN.space import Grid2D"] + list(lif.imports),
    )


def export_lif_neuron_type(p: LIFParameters) -> Statement:
    pynn_parameters = {
        "tau_m": 1 / float(p.tau_mem_inv),
        "tau_syn_E": 1 / float(p.tau_syn_inv),
        "tau_syn_I": 1 / float(p.tau_syn_inv),
        "v_reset": float(p.v_reset),
        "v_thresh": float(p.v_th),
    }
    pynn_parameter_statement = export_dict(pynn_parameters)
    return Statement(
        f"p.IF_curr_exp({pynn_parameter_statement.value})",
        pynn_parameter_statement.imports,
    )


def export_dict(d: Dict[Any, Any]) -> Statement:
    def _export_dict_key(key: Any) -> str:
        if not isinstance(key, str):
            raise ValueError("Parameter key must be a string", key)
        return str(key)

    def _export_dict_value(value: Any) -> str:
        if isinstance(value, str):
            return f"'{str(value)}'"
        else:
            return str(value)

    pynn_dict = ""
    for key, value in d.items():
        pynn_dict = pynn_dict + f"{_export_dict_key(key)}={_export_dict_value(value)},"
    return Statement(pynn_dict[:-1])


# def output_ethernet(layer: )
