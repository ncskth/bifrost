from bifrost.export.output import export_layer_output
from bifrost.export.utils import export_list, export_structure
from bifrost.ir.layer import NeuronLayer, Layer
from bifrost.ir.input import InputLayer, SpiNNakerSPIFInput
from bifrost.ir.output import OutputLayer
from bifrost.ir.parameter import ParameterContext
from bifrost.ir.cell import (LIFCell, LICell, IFCell)
from bifrost.ir.constants import (SynapseShapes, SynapseTypes)
from bifrost.export.statement import Statement
from bifrost.export.pynn import (SIM_NAME, PyNNSynapseShapes,
                                 PyNNSynapseTypes, PyNNNeuronTypes)
from bifrost.export.input import export_layer_input
from bifrost.export.record import export_record


def export_cell_params(layer: Layer, context: ParameterContext[str],
                       join_str:str = ",\n", spaces:int = 8) -> Statement:
    # todo: take 'locations/addresses' from context and express as a function
    #       which returns a dictionary so that once it's called, we can use the
    #       ** operator to pass key-value pairs as parameters to cell class
    #       constructor
    par_var = "_par_name"
    cell_name = layer.cell.__class__.__name__
    sp = " " * spaces
    layer_name = str(layer)
    func_name = f"__nrn_params_{layer_name}_f"

    list_name = f"__parameter_names"
    dict_name = "__parameter_dict"
    fcall = context.neuron_parameter(layer_name, par_var)
    names = export_list(list_name, context.parameter_names(layer.cell))
    params = []

    f = f"""
def {func_name}():
    {names}
    {dict_name} = dict()
    for {par_var} in {list_name}:
        k, v = {fcall}
        {dict_name}[k] = v
    return {dict_name}
    """
    return Statement(f"**({func_name}())", preambles=[f])


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
                        param_join_str=", ", pop_join_str=",\n") -> Statement:
    neuron = export_neuron_type(layer, context, join_str=", ", spaces=0)
    structure = export_structure(layer)
    param_template = param_join_str.join([
            f"{layer.size}", f"{neuron.value}", f"structure={structure.value}",
            "label=\"{}\""])

    statement = Statement()
    for channel in range(layer.channels):
        var = f"{layer.variable(channel)}"
        par = param_template.format(var)
        pop = f"{var} = {SIM_NAME}.Population({par})"
        recs = export_record(layer, channel)
        statement += Statement(pop,
                               imports=neuron.imports,
                               preambles=neuron.preambles)
        if len(recs.value):
            statement += recs

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
        f"{SIM_NAME}.{cell_type}({pynn_parameter_statement.value})",
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




