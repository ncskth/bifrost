from bifrost.ir.parameter import ParameterContext
from bifrost.ir.layer import NeuronLayer
from bifrost.ir.network import Network
from bifrost.export.population import export_layer_neuron
from bifrost.exporter import export_network
from bifrost.export import pynn
from bifrost.export.torch import TorchContext
from bifrost.text_utils import remove_blank
from bifrost.export.constants import SAVE_VARIABLE_NAME


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
    run_time = 100.1
    n = Network([], [], run_time, configuration=["some", "config"])
    out = export_network(n, c)
    imports = "\n".join(sorted(set(pynn.pynn_imports + c.imports)))
    expected = (
        f"{imports}\n"
        f"{pynn.pynn_header(1.0)}\n"
        f"{c.preamble}\n"
        f"some\n"
        f"config\n\n"
        f"{pynn.pynn_runner(run_time)}"
        f"{pynn.pynn_footer()}"
    )

    assert ( remove_blank(out) == remove_blank(expected))


def test_export_single():
    run_time = 100.1
    layer_name = "l_l_1_"
    n_neurons = 1
    n_chan = 10
    torch_context = TorchContext({f"{layer_name}{n_chan}": "0"})
    # name, size, channels
    l = NeuronLayer("l", n_neurons, n_chan)
    net = Network([l], set(), run_time)
    out = export_network(net, torch_context)
    lif = export_layer_neuron(l, torch_context)
    imports = "\n".join(sorted(set(torch_context.imports + pynn.pynn_imports + lif.imports)))
    lif_defs = "\n".join(list(set(lif.preambles)))
    expected = (
        f"{imports}\n"
        f"{pynn.pynn_header(1.0)}\n"
        f"{torch_context.preamble}\n"
        f"{lif_defs}\n"
        f"{lif.value}\n"
        f"{pynn.pynn_runner(run_time)}\n"
        f"{pynn.pynn_footer()}\n"
    )

    assert (remove_blank(out) == remove_blank(expected))
