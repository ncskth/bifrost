from bifrost.export.statement import Statement
from bifrost.ir import OutputLayer, ParameterContext, EthernetOutput, DummyTestOutputSink


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
    return Statement(f"""{layer.variable(0)} = EthernetOutputSink()""")
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
# sim.external_devices.activate_live_output_for(
#     pop_forward,
#     database_notify_port_num=live_spikes_connection_receive.local_port)

def export_dummy_output(layer: OutputLayer, ctx: ParameterContext[str]) -> Statement:
    Statement(f"""{layer.variable(0)} = DummyOutputSink()""")