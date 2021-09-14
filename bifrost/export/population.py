from bifrost.export.output import export_layer_output
from bifrost.export.utils import export_list, export_list_var
from bifrost.ir.layer import NeuronLayer, Layer
from bifrost.ir.input import InputLayer, SpiNNakerSPIFInput
from bifrost.ir.output import OutputLayer
from bifrost.ir.parameter import ParameterContext
from bifrost.ir.cell import (LIFCell, LICell, IFCell)
from bifrost.ir.constants import (SynapseShapes, SynapseTypes)
from bifrost.export.statement import Statement
from bifrost.export.pynn import (SIMULATOR_NAME, PyNNSynapseShapes,
                                 PyNNSynapseTypes, PyNNNeuronTypes, export_structure)
from bifrost.export.input import export_layer_input
from bifrost.export.record import export_record


def export_cell_params(layer: Layer, context: ParameterContext[str],
                       join_str:str = ",\n", spaces:int = 8) -> Statement:
    # todo: review if this is the best approach
    parameter_variable_name = "_par_name"
    cell_name = layer.cell.__class__.__name__
    spaces_text = " " * spaces
    layer_name = str(layer)
    generator_function_name = f"__nrn_params_{layer_name}_f"

    parameter_list_name = "__parameter_names"
    temporary_dictionary_name = "__parameter_dict"
    function_call_for_parameters = context.neuron_parameter(
                                    layer_name, parameter_variable_name)
    parameter_names = export_list_var(parameter_list_name,
                            context.parameter_names(layer.cell))

    f = (f"def {generator_function_name}():\n"
         f"{spaces_text}{parameter_names}\n"
         f"{spaces_text}{temporary_dictionary_name} = dict()\n"
         f"{spaces_text}for {parameter_variable_name} in {parameter_list_name}:\n"
         f"{spaces_text * 2}k, v = {function_call_for_parameters}\n"
         f"{spaces_text * 2}{temporary_dictionary_name}[k] = v\n"
         f"{spaces_text}return {temporary_dictionary_name}\n"
    )
    return Statement(f"**({generator_function_name}())", preambles=[f])


def export_layer(layer: Layer, context: ParameterContext[str]) -> Statement:
    if isinstance(layer, SpiNNakerSPIFInput):
        return export_layer_spif(layer, context)
    elif isinstance(layer, NeuronLayer):
        return export_layer_neuron(layer, context)
    elif isinstance(layer, InputLayer):
        return export_layer_input(layer, context)
    elif isinstance(layer, OutputLayer):
        return export_layer_output(layer, context)
    else:
        raise ValueError("Unknown layer type", layer)


def export_layer_neuron(layer: NeuronLayer, context: ParameterContext[str],
                        param_join_str=", \n", pop_join_str=",\n") -> Statement:
    layer_variable_name = layer.variable('')
    var_name_spaces = " " * (len(layer_variable_name) + 4)
    tab = " " * 4
    neuron = export_neuron_type(layer, context, join_str=", ", spaces=4)
    structure = export_structure(layer)
    label_template = f"{layer.variable('')}{{channel}}"

    param_template = f"{param_join_str}{var_name_spaces}{tab}".join([
            f"{layer.size}", f"{neuron.value}", f"structure={structure.value}",
            f"label=f\"{label_template}\""])

    population_text = (
        f"{layer_variable_name} = {{channel: {SIMULATOR_NAME}.Population({param_template})\n"
        f"{var_name_spaces}for channel in range({layer.channels})}}"
    )

    statement = Statement(population_text,
                          imports=neuron.imports,
                          preambles=neuron.preambles)

    if len(layer.record):
        statement += export_record(layer)

    # just add a break to separate populations for each layer
    statement += Statement("")

    if isinstance(statement.imports, tuple):
        statement.imports = structure.imports
    else:
        statement.imports += structure.imports

    if isinstance(statement.preambles, tuple):
        statement.preambles = structure.preambles
    else:
        statement.preambles += structure.preambles


    return statement


def export_neuron_type(layer: NeuronLayer, ctx: ParameterContext[str],
                       join_str:str = ",\n", spaces:int = 0) -> Statement:
    pynn_parameter_statement = export_cell_params(layer, ctx, join_str, spaces)
    cell_type = get_pynn_cell_type(layer.cell, layer.synapse)
    return Statement(
        f"{SIMULATOR_NAME}.{cell_type}({pynn_parameter_statement.value})",
        imports=pynn_parameter_statement.imports,
        preambles=pynn_parameter_statement.preambles
    )


# todo: this is PyNN, I guess we should move it somewhere else
def get_pynn_cell_type(cell, synapse):
    if isinstance(cell, LICell):
        cell_type = PyNNNeuronTypes.LI.value
    elif isinstance(cell, LIFCell):
        cell_type = PyNNNeuronTypes.LIF.value
    elif isinstance(cell, IFCell):
        cell_type = PyNNNeuronTypes.NIF.value
    else:
        raise NotImplementedError("Neuron type not yet available")

    if synapse.synapse_type == SynapseTypes.CURRENT:
        syn_type = PyNNSynapseTypes.CURRENT.value
    elif synapse.synapse_type == SynapseTypes.CONDUCTANCE:
        syn_type = SynapseTypes.CONDUCTANCE.value
    else:
        raise NotImplementedError(
                f"Synapse type not yet available {synapse.synapse_type}")

    if synapse.synapse_shape == SynapseShapes.EXPONENTIAL:
        syn_shape = PyNNSynapseShapes.EXPONENTIAL.value
    elif synapse.synapse_shape == SynapseShapes.ALPHA:
        syn_shape = PyNNSynapseShapes.ALPHA.value
    elif synapse.synapse_shape == SynapseShapes.DELTA:
        syn_shape = PyNNSynapseShapes.DELTA.value
    else:
        raise NotImplementedError(
                f"Synapse 'shape' not yet available {synapse.synapse_shape}")

    return "{}_{}_{}".format(cell_type, syn_type, syn_shape)




