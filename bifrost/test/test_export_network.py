from bifrost.ir.parameter import ParameterContext
from bifrost.ir.layer import NeuronLayer
from bifrost.ir.network import Network
from bifrost.export.population import export_layer_neuron
from bifrost.exporter import export_network
from bifrost.export import pynn
from bifrost.export.torch import TorchContext
from bifrost.text_utils import remove_blank

class MockContext(ParameterContext):
    preamble = "Dubi\ndubi\ndubi\ndubdubdub"
    imports = []


torch_context = TorchContext({"l": "0"})


def test_export_empty():
    run_time = 100.1
    c = MockContext()
    n = Network([], [], run_time)
    out = export_network(n, c)
    imports = "\n".join(sorted(set(pynn.pynn_imports + c.imports)))
    test = (
        f"{imports}\n" 
        f"{pynn.pynn_header(1.0)}\n"
        f"{c.preamble}\n"
        f"{pynn.pynn_runner(run_time)}"
        f"{pynn.pynn_footer()}"
    )
    eout = remove_blank(out)
    etest = remove_blank(test)
    assert eout == etest

def test_export_neurons_per_core():
    c = MockContext()
    n = Network([], [], 100.1, config=["some", "config"])
    out = export_network(n, c)
    imports = "\n".join(sorted(set(pynn.pynn_imports + c.imports)))
    assert (
        out
        == f"{imports}\n{pynn.pynn_header(1.0)}\n{c.preamble}\nsome\nconfig\n\n{pynn.pynn_footer(100.1)}"
    )


def test_export_single():
    torch_context = TorchContext({"l_l_1_10": "0"})
    # name, size, channels
    l = NeuronLayer("l", 1, 10)
    net = Network([l], set(), 100.1)
    out = export_network(net, torch_context)
    lif = export_layer_neuron(l, torch_context)
    imports = "\n".join(sorted(set(torch_context.imports + pynn.pynn_imports + lif.imports)))
    lif_defs = "\n".join(list(set(lif.preambles)))
    expected = f"{imports}\n{pynn.pynn_header(1.0)}\n{torch_context.preamble}\n" \
               f"{lif_defs}\n{lif.value}\n{pynn.pynn_footer(100.1)}"
    assert (out == expected)
