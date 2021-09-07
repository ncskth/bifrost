from typing import Any, Dict, List
from bifrost.ir.layer import NeuronLayer, Layer
from bifrost.ir.input import InputLayer, SpiNNakerSPIFInput, DummyTestInputSource
from bifrost.ir.output import OutputLayer, EthernetOutput, DummyTestOutputSink
from bifrost.ir.parameter import ParameterContext
from bifrost.ir.cell import (LIFCell, LICell, IFCell)
from bifrost.export.statement import Statement
from bifrost.export.pynn import SIM_NAME

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
        pop = [f"{var} = {SIM_NAME}.Population({par})"]
        recs = [export_record(layer, channel)]

        statement += Statement("\n".join(pop+recs),
                               imports=neuron.imports,
                               preambles=neuron.preambles)

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


def export_layer_input(layer: InputLayer, ctx: ParameterContext[str]) -> Statement:
    source = layer.source
    if isinstance(source, SpiNNakerSPIFInput):
        spif_layer = source
        statement = Statement(
            [
                f"""{layer.variable(channel)} = {SIM_NAME}.Population(None,{SIM_NAME}.external_devices.SPIFRetinaDevice(\
base_key={channel},width={source.x},height={source.y},sub_width={source.x_sub},sub_height={source.y_sub},\
input_x_shift={source.x_shift},input_y_shift={source.y_shift}))"""
                for channel in range(layer.channels)
            ]
        )
    elif isinstance(source, DummyTestInputSource):
        pop = [f"{layer.variable(0)} = DummyInputSource()"]
        recs = [export_record(layer, 0)]
        statement = Statement("\n".join(pop + recs))
    else:
        raise ValueError("Unknown input source", source)

    statement += Statement("")  # add carriage return
    return statement

def export_record(layer: Layer, channel: int):
    if len(layer.record) == 0:
        return ""
    elif len(layer.record) == 1:
        rs = f"\"{layer.record[0]}\""
    else:
        rs = ", ".join(f"\"{r}\"" for r in layer.record)
        rs = f"[{rs}]"

    return f"{layer.variable(channel)}.record({rs})"

def export_layer_output(layer: OutputLayer, ctx: ParameterContext[str]) -> Statement:
    sink = layer.sink
    if isinstance(sink, EthernetOutput):
        statement = Statement(f"""{layer.variable(0)} = EthernetOutputSink()""")
    elif isinstance(sink, DummyTestOutputSink):
        statement = Statement(f"""{layer.variable(0)} = DummyOutputSink()""")
    else:
        raise ValueError("Unknown input source", layer.source)

    statement += Statement("")  # add carriage return
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

def export_structure(layer):
    ratio = float(layer.shape[1]) / layer.shape[0]
    return Statement(f"Grid2D({ratio})",
                     imports=['from pyNN.space import Grid2D'])

# todo: this is PyNN, I guess we should move it somewhere else
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
# create python injector
# 2  def send_spike(label, sender):
# 3 sender.send_spike(label, 0, send_full_keys=True)
# 4
# 5 # set up python injector connection
# 6  live_spikes_connection =
# 7      sim.external_devices.SpynnakerLiveSpikesConnection(
# 8          send_labels=[“spike_sender”])
# 9
# 10 # register python injector with injector connection
# 11 live_spikes_connection.add_start_callback(“spike_sender”, send_spike)
# activate_live_output_for(receiver, database_notify_port_num=19996)