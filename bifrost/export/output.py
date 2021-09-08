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
    vars = ", ".join(
        [f'"{source.variable(channel)}"' for channel in range(source.channels)])
    live_rec = f"live_spikes_connection_receive = " \
               f"{SIM_NAME}.external_devices.SpynnakerLiveSpikesConnection(" \
               f"receive_labels=[{vars}]," \
               f"local_port=None, send_labels=None)"
    live_out = [f"sim.external_devices.activate_live_output_for("\
                f"{source.variable(channel)}," \
                f"database_notify_port_num=live_spikes_connection_receive.local_port)"
                for channel in range(source.channels)]

    return Statement([live_rec] + live_out)

def export_dummy_output(layer: OutputLayer, ctx: ParameterContext[str]) -> Statement:
    Statement(f"""{layer.variable(0)} = DummyOutputSink()""")