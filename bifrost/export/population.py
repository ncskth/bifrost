from typing import Any, Dict
# from norse.torch.functional.lif import LIFParameters
# from torch._C import Value
from bifrost.ir.layer import NeuronLayer, Layer
from bifrost.ir.input import SpiNNakerSPIFInput
from bifrost.ir.parameter import ParameterContext
from bifrost.ir.cell import (LIFCell, LICell, IFCell)
from .pynn import Statement



def export_dict(d: Dict[Any, Any], join_str=",\n", n_post_spaces=0) -> Statement:
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
    spaces = "".join([" "] * n_post_spaces)
    return Statement((f"{join_str}{spaces}").join(pynn_dict), [])


def export_layer(layer: Layer, context: ParameterContext[str]) -> Statement:
    if isinstance(layer, SpiNNakerSPIFInput):
        return export_layer_spif(layer, context)
    elif isinstance(layer, NeuronLayer):
        return export_neuron_layer(layer, context)
    else:
        raise ValueError("Unknown layer type", layer)


def export_neuron_layer(layer: NeuronLayer, context: ParameterContext[str],
                        param_join_str=", ", pop_join_str=",\n") -> Statement:
    neuron = export_neuron_type(layer, context, join_str=", ", spaces=0)
    structure = export_structure(layer)
    param_template = param_join_str.join([
            f"{layer.size}", f"{neuron.value}", f"structure={structure.value}",
            f"label='{layer.variable}_{{}}'"])
    list_of_pops = [
        (f"p.Population({param_template})").format(channel_index)
        for channel_index in range(layer.n_channels)
    ]
    lop = pop_join_str.join(list_of_pops)
    return Statement(
        f"{layer.variable} = [{lop}]",
        imports=neuron.imports+structure.imports,
    )


def export_layer_spif(layer: SpiNNakerSPIFInput) -> Statement:
    return Statement(
        f"""{layer.variable} = p.Population(None, p.external_devices.SPIFRetinaDevice(\
base_key=0, width={layer.x}, height={layer.y}, sub_width={layer.x_sub}, sub_height={layer.y_sub},\
input_x_shift={layer.x_shift}, input_y_shift={layer.y_shift}))"""
    )


def export_neuron_type(layer: NeuronLayer, ctx: ParameterContext[str],
                       join_str:str = ",\n", spaces:int = 0) -> Statement:
    pynn_parameter_statement = export_dict(layer.cell.parameters,
                                           join_str, spaces)
    cell_type = get_pynn_cell_type(layer.cell, layer.synapse)
    return Statement(
        f"p.{cell_type}({pynn_parameter_statement.value})",
        pynn_parameter_statement.imports,
    )

def export_structure(layer):
    ratio = float(layer.shape[1]) / layer.shape[0]
    return Statement(f"Grid2D({ratio})",
                     imports=['from pyNN.space import Grid2D'])

def get_pynn_cell_type(cell, synapse):
    if isinstance(cell, (LICell, LIFCell)):
        cell_type = 'IF'  # in PyNN this is missing the L for some #$%@ reason
    elif isinstance(cell, IFCell):
        cell_type = 'NIF' #  as in Non-leaky Integrate and Fire
    else:
        raise NotImplementedError("Neuron type not yet available")

    if synapse.synapse_type == 'current':
        syn_type = 'curr'
    elif synapse.synapse_type == 'conductance':
        syn_type = 'cond'
    else:
        raise NotImplementedError(
                f"Synapse type not yet available {synapse.synapse_type}")

    if synapse.synapse_shape == 'exponential':
        syn_shape = 'exp'
    elif synapse.synapse_shape == 'alpha':
        syn_shape = 'alpha'
    elif synapse.synapse_shape == 'delta':
        syn_shape = 'delta'
    else:
        raise NotImplementedError(
                f"Synapse 'shape' not yet available {synapse.synapse_shape}")

    return "{}_{}_{}".format(cell_type, syn_type, syn_shape)


# def output_ethernet(layer: )
