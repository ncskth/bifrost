from typing import Any, Dict
from bifrost.ir.layer import LIFAlphaLayer, Layer
from bifrost.ir.input import InputLayer, SpiNNakerSPIFInput
from bifrost.ir.output import OutputLayer
from bifrost.ir.parameter import ParameterContext
from .pynn import Statement


def export_layer(layer: Layer, context: ParameterContext[str]) -> Statement:
    if isinstance(layer, SpiNNakerSPIFInput):
        return export_layer_spif(layer, context)
    elif isinstance(layer, LIFAlphaLayer):
        return export_layer_neuron(layer, context)
    elif isinstance(layer, InputLayer):
        pass
    elif isinstance(layer, OutputLayer):
        pass
    else:
        raise ValueError("Unknown layer type", layer)


def export_layer_neuron(layer: Layer, context: ParameterContext[str]) -> Statement:
    lif_p = export_lif_neuron_type(layer, context)
    statement = Statement()
    for channel in range(layer.channels):
        statement += Statement(
            f"{layer.variable(channel)} = p.Population({layer.neurons}, {lif_p.value})",
            imports=lif_p.imports,
        )
    return statement


def export_layer_input(layer: InputLayer) -> Statement:
    if isinstance(layer.source, SpiNNakerSPIFInput):
        spif_layer = layer.source
        return Statement(
            [
                f"""{layer.variable(channel)} = p.Population(None,p.external_devices.SPIFRetinaDevice(\
base_key={channel},width={spif_layer.x},height={spif_layer.y},sub_width={spif_layer.x_sub},sub_height={spif_layer.y_sub},\
input_x_shift={spif_layer.x_shift},input_y_shift={spif_layer.y_shift}))"""
                for channel in range(layer.channels)
            ]
        )
    else:
        raise ValueError("Unknown input source", layer.source)


def export_lif_neuron_type(layer: Layer, context: ParameterContext[str]) -> Statement:
    parameters = []
    for name in context.parameter_names(layer):
        parameters.append(context.neuron_parameter(layer, name))
    parameter_values = ",".join(parameters)
    return Statement(f"p.IF_curr_exp({parameter_values})")


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
