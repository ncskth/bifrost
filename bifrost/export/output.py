from bifrost.export.statement import Statement
from bifrost.ir import OutputLayer, ParameterContext, EthernetOutput, DummyTestOutputSink
from bifrost.export.pynn import SIM_NAME

def export_layer_output(layer: OutputLayer, ctx: ParameterContext[str]) -> Statement:
    sink = layer.sink
    if isinstance(sink, EthernetOutput):
        statement = export_ethernet_output(layer, ctx)
    elif isinstance(sink, DummyTestOutputSink):
        statement = export_dummy_output(layer, ctx)
    else:
        raise ValueError("Unknown input source", layer.source)

    statement += Statement("")  # add carriage return
    return statement


def export_ethernet_output(layer: OutputLayer, ctx: ParameterContext[str]) -> Statement:
    source = layer.source
    labels_var = f"out_labels_{source.variable('')}"
    vars = f"[f\"{source.variable('')}{{channel}}\" for channel in range({source.channels})]"
    live_rec = (
        f"{labels_var} = {vars}\n" 
        f"live_spikes_connection_receive = " 
        f"{SIM_NAME}.external_devices.SpynnakerLiveSpikesConnection(\n" 
        f"    receive_labels={labels_var},\n" 
        f"    local_port=None, send_labels=None)\n"
    )
    live_out = (
            f"for channel in range({source.channels}):\n"
            f"    {SIM_NAME}.external_devices.activate_live_output_for(\n"
            f"       {source.variable('')}[channel],\n" 
            f"       database_notify_port_num=live_spikes_connection_receive.local_port)\n"
        )

    return Statement([live_rec, live_out])

def export_dummy_output(layer: OutputLayer, ctx: ParameterContext[str]) -> Statement:
    Statement(f"""{layer.variable(0)} = DummyOutputSink()""")