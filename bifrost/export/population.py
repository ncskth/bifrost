from typing import Any, Dict
from norse.torch.functional.lif import LIFParameters
from torch._C import Value
from bifrost.ir.layer import NeuronLayer, Layer
from bifrost.ir.input import SpiNNakerSPIFInput
from bifrost.ir.parameter import ParameterContext
from bifrost.ir.cell import (LIFCell, LICell, IFCell)
from .pynn import Statement



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

    pynn_dict = [f"{_export_dict_key(key)}={_export_dict_value(value)}"
                 for key, value in d.items()]

    return Statement(",\n".join(pynn_dict))


def export_layer(layer: Layer, context: ParameterContext[str]) -> Statement:
    if isinstance(layer, SpiNNakerSPIFInput):
        return export_layer_spif(layer, context)
    elif isinstance(layer, NeuronLayer):
        return export_neuron_layer(layer, context)
    else:
        raise ValueError("Unknown layer type", layer)


def export_neuron_layer(layer: NeuronLayer, context: ParameterContext[str]) -> Statement:
    neuron = export_neuron_type(layer, context)
    structure = export_structure(layer)
    return Statement(
        f"{layer.variable} = p.Population({layer.size}, {neuron.value},\n"
        f"                      structure={structure})",
        imports=neuron.imports,
    )


def export_layer_spif(layer: SpiNNakerSPIFInput) -> Statement:
    return Statement(
        f"""{layer.variable} = p.Population(None, p.external_devices.SPIFRetinaDevice(\
base_key=0, width={layer.x}, height={layer.y}, sub_width={layer.x_sub}, sub_height={layer.y_sub},\
input_x_shift={layer.x_shift}, input_y_shift={layer.y_shift}))"""
    )


def export_neuron_type(layer: NeuronLayer, ctx: ParameterContext[str]) -> Statement:
    pynn_parameter_statement = export_dict(layer.cell.parameters)
    return Statement(
        f"p.{layer.type}({pynn_parameter_statement.value})",
        pynn_parameter_statement.imports,
    )

def export_structure(layer):
    return Statement("", imports=['from pyNN.space import Grid2D'])

def get_pynn_cell_type(cell, synapse):
    # TODO: this needs to take into account what synapse shape and mechanism
    #       we have (e.g. current or conductanc; exponential, delta or alpha)
    if isinstance(cell, (LICell, LIFCell)):
        cell_type = 'IF'  # in PyNN this is missing the L for some #$%@ reason
    elif isinstance(cell, IFCell):
        cell_type = 'NIF' #  as in Non-leaky Integrate and Fire
    else:
        raise NotImplementedError("Neuron type not yet available")

    if

# def output_ethernet(layer: )
