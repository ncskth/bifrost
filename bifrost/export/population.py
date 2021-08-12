from typing import Any, Dict
from norse.torch.functional.lif import LIFParameters
from torch._C import Value
from bifrost.ir.layer import LIFLayer, Layer, Conv2dLIFLayer
from bifrost.ir.input import SpiNNakerSPIFInput
from .pynn import Statement


def export_layer(layer: Layer) -> Statement:
    if isinstance(layer, SpiNNakerSPIFInput):
        return export_layer_spif(layer)
    elif isinstance(layer, Conv2dLIFLayer):
        return export_layer_conv2d(layer)
    elif isinstance(layer, LIFLayer):
        return export_layer_lif(layer)
    else:
        raise ValueError("Unknown layer type", layer)


def export_layer_lif(layer: LIFLayer) -> Statement:
    lif_p = export_lif_neuron_type(layer.parameters)
    return Statement(
        f"{layer.name} = p.Population({layer.neurons}, {lif_p.value})",
        imports=lif_p.imports,
    )


def export_layer_spif(layer: SpiNNakerSPIFInput) -> Statement:
    return Statement(
        f"""{layer.name} = p.Population(None, p.external_devices.SPIFRetinaDevice(\
base_key=0, width={layer.x}, height={layer.y}, sub_width={layer.x_sub}, sub_height={layer.y_sub},\
input_x_shift={layer.x_shift}, input_y_shift={layer.y_shift}))"""
    )


def export_layer_conv2d(layer: Conv2dLIFLayer) -> Statement:
    lif = export_lif_neuron_type(layer.parameters)
    return Statement(
        f"{layer.name} = p.Population({layer.width * layer.height}, {lif.value}, structure=Grid2D({layer.width / layer.height}))",
        ["from pyNN.space import Grid2D"] + list(lif.imports),
    )


def export_lif_neuron_type(p: LIFParameters) -> Statement:
    pynn_parameters = {
        "tau_m": 1 / p.tau_mem_inv,
        "tau_syn_E": 1 / p.tau_syn_inv,
        "tau_syn_I": 1 / p.tau_syn_inv,
        "v_leak": p.v_leak,
        "v_reset": p.v_reset,
        "v_th": p.v_th,
    }
    pynn_parameter_statement = export_dict(pynn_parameters)
    return Statement(
        f"p.IF_curr_exp({pynn_parameter_statement.value})",
        pynn_parameter_statement.imports,
    )


def export_dict(d: Dict[Any, Any]) -> Statement:
    def _export_dict_string(value: Any) -> str:
        if isinstance(value, str):
            return f"'{str(value)}'"
        else:
            return str(value)

    pynn_dict = "{"
    for key, value in d.items():
        pynn_dict = (
            pynn_dict + f"{_export_dict_string(key)}: {_export_dict_string(value)},"
        )
    return Statement(pynn_dict + "}")


# def output_ethernet(layer: )
