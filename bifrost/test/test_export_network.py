from bifrost.export.population import export_layer_lif
from bifrost.ir.layer import LIFLayer
from bifrost.ir.network import Network
from bifrost.exporter import export_network
from bifrost.export import pynn


def test_export_empty():
    n = Network([], [], 100.1)
    out = export_network(n)
    assert out == f"{pynn.pynn_header(1.0)}\n\n{pynn.pynn_footer(100.1)}"


def test_export_neurons_per_core():
    n = Network([], [], 100.1, config=["some", "config"])
    out = export_network(n)
    assert (
        out
        == f"{pynn.pynn_header(1.0)}\nsome\nconfig\n\n{pynn.pynn_footer(100.1)}"
    )


def test_export_single():
    l = LIFLayer("l", 10)
    net = Network([l], set(), 100.1)
    out = export_network(net)
    lif = export_layer_lif(l)
    assert out == f"{pynn.pynn_header(1.0)}\n{lif.value}\n{pynn.pynn_footer(100.1)}"
