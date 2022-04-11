from bifrost.export.statement import Statement
from bifrost.ir import (
    OutputLayer,
    ParameterContext,
    EthernetOutput,
    DummyTestOutputSink,
)
from bifrost.export.pynn import SIMULATOR_NAME


def export_layer_output(
    layer: OutputLayer, context: ParameterContext[str]
) -> Statement:
    sink = layer.sink
    if isinstance(sink, EthernetOutput):
        statement = export_ethernet_output(layer, context)
    elif isinstance(sink, DummyTestOutputSink):
        statement = export_dummy_output(layer, context)
    else:
        raise ValueError("Unknown input source", layer.source)

    statement += Statement("")  # add carriage return
    return statement


def export_ethernet_output(
    layer: OutputLayer, context: ParameterContext[str]
) -> Statement:
    source = layer.source
    variable_name = f"out_labels_{source.variable('')}"
    # these are supposed to be the populations 'emiting' spikes which will be
    # captured by the ethernet population
    population_labels = (
        f"[f\"{source.variable('')}{{channel}}\" "
        f"for channel in range({source.channels})]"
    )
    receiver = (
        f"{variable_name} = {population_labels}\n"
        f"live_spikes_connection_receive = "
        f"{SIMULATOR_NAME}.external_devices.SpynnakerLiveSpikesConnection(\n"
        f"    receive_labels={variable_name},\n"
        f"    local_port={layer.sink.port}, send_labels=None)\n"
    )
    ethernet_output = (
        f"for channel in range({source.channels}):\n"
        f"    {SIMULATOR_NAME}.external_devices.activate_live_output_for(\n"
        f"       {source.variable('')}[channel],\n"
        f"       database_notify_port_num=live_spikes_connection_receive.local_port)\n"
    )

    return Statement([receiver, ethernet_output])


def export_dummy_output(
    layer: OutputLayer, context: ParameterContext[str]
) -> Statement:
    Statement(f"""{layer.variable()} = {{0: DummyOutputSink()}}""")
