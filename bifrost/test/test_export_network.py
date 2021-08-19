from bifrost.ir.parameter import ParameterContext
from bifrost.ir.layer import LIFLayer
from bifrost.ir.network import Network
from bifrost.export.population import export_layer_lif
from bifrost.exporter import export_network
from bifrost.export import pynn

class MockContext(ParameterContext):
    preamble = "Dubi\ndubi\ndubi\ndubdubdub"


def test_export_empty():
    c = MockContext()
    n = Network([], [], 100.1)
    out = export_network(n, c)
    assert out == f"{pynn.pynn_header(1.0)}\n{c.preamble}\n\n{pynn.pynn_footer(100.1)}"


def test_export_neurons_per_core():
    c = MockContext()
    n = Network([], [], 100.1, config=["some", "config"])
    out = export_network(n, c)
    assert out == f"{pynn.pynn_header(1.0)}\n{c.preamble}\nsome\nconfig\n\n{pynn.pynn_footer(100.1)}"


def test_export_single():
    l = LIFLayer("l", 10)
    c = MockContext()
    net = Network([l], set(), 100.1)
    out = export_network(net, c)
    lif = export_layer_lif(l, c)
    assert out == f"{pynn.pynn_header(1.0)}\n{c.preamble}\n{lif.value}\n{pynn.pynn_footer(100.1)}"
