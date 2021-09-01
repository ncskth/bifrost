from bifrost.ir.parameter import ParameterContext
from bifrost.ir.layer import NeuronLayer
from bifrost.ir.network import Network
from bifrost.export.population import export_layer_neuron
from bifrost.exporter import export_network
from bifrost.export import pynn
from bifrost.export.torch import TorchContext


class MockContext(ParameterContext):
    preamble = "Dubi\ndubi\ndubi\ndubdubdub"


torch_context = TorchContext({"l": "0"})


def test_export_empty():
    c = MockContext()
    n = Network([], [], 100.1)
    out = export_network(n, c)
    assert out == f"{pynn.pynn_header(1.0)}\n{c.preamble}\n\n{pynn.pynn_footer(100.1)}"


def test_export_neurons_per_core():
    c = MockContext()
    n = Network([], [], 100.1, config=["some", "config"])
    out = export_network(n, c)
    assert (
        out
        == f"{pynn.pynn_header(1.0)}\n{c.preamble}\nsome\nconfig\n\n{pynn.pynn_footer(100.1)}"
    )


def test_export_single():
    l = NeuronLayer("l", 1, 10)
    net = Network([l], set(), 100.1)
    out = export_network(net, torch_context)
    lif = export_layer_neuron(l, torch_context)
    assert (
        out
        == f"{pynn.pynn_header(1.0)}\n{torch_context.preamble}\n{lif.value}\n{pynn.pynn_footer(100.1)}"
    )
