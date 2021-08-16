import pytest

from norse.torch import LIFParameters

from bifrost.ir.layer import Conv2dLIFLayer, LIFLayer
from bifrost.ir.input import SpiNNakerSPIFInput
from bifrost.ir.parameter import ParameterContext

from bifrost.export import population, pynn

class MockContext(ParameterContext):
    pass

def test_input_to_pynn():
    spif_layer = SpiNNakerSPIFInput("i", 1, 2)
    variable = population.export_layer_spif(spif_layer)
    assert variable == pynn.Statement(
        """l_i = p.Population(None, p.external_devices.SPIFRetinaDevice(\
base_key=0, width=1, height=2, sub_width=32, sub_height=16,\
input_x_shift=16, input_y_shift=0))"""
    )


def test_cnn2d_to_pynn():
    cnn_layer = Conv2dLIFLayer("c", 640, 480, 2)
    c = MockContext()
    cnn_pynn = population.export_layer_conv2d(cnn_layer, c)
    lif_p = population.export_lif_neuron_type(cnn_layer.parameters)
    assert cnn_pynn == pynn.Statement(
        f"l_c = p.Population({640 * 480}, {lif_p.value}, structure=Grid2D({640 / 480}))",
        ["from pyNN.space import Grid2D"],
    )


def test_lif_to_pynn():
    l = LIFLayer("l", 10)
    c = MockContext()
    lif_p = population.export_lif_neuron_type(l.parameters)
    actual = population.export_layer_lif(l, c)
    assert actual == pynn.Statement(f"l_l = p.Population(10, {lif_p.value})")


def test_lif_neuron_to_pynn():
    p = LIFParameters()
    p_dict = population.export_dict(
        {
            "tau_m": 1 / float(p.tau_mem_inv),
            "tau_syn_E": 1 / float(p.tau_syn_inv),
            "tau_syn_I": 1 / float(p.tau_syn_inv),
            "v_reset": float(p.v_reset),
            "v_thresh": float(p.v_th),
        }
    )
    s = population.export_lif_neuron_type(p)
    assert s == pynn.Statement(f"p.IF_curr_exp({p_dict.value})", imports=p_dict.imports)


def test_dict_to_pynn_parameters():
    d = {"test": 12, "tau_m": "value"}
    actual = population.export_dict(d)
    assert actual == pynn.Statement("test=12,tau_m='value'")


def test_dict_to_pynn_parameters_fail():
    d = {"test": 12, 2: "value"}
    with pytest.raises(ValueError):
        actual = population.export_dict(d)


# def test_ann_to_graph():
#     ms = torch.nn.Sequential(torch.nn.Linear(28, 1), torch.nn.ReLU())
#     data = torch.zeros((1, 28, 28))
#     layers = model_to_graph(ms, data)
#     assert layers[0] == TorchLayer("linear", "Linear", 1)
#     assert layers[1] == TorchLayer("ReLU", "ReLU", 1)


# def test_export_snn():
# ms = torch.nn.Sequential(torch.nn.Linear(28, 1), norse.LICell())
