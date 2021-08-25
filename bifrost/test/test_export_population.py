from bifrost.export.torch import TorchContext
from typing import List
import pytest

from norse.torch import LIFParameters

from bifrost.ir.layer import LIFAlphaLayer, Layer
from bifrost.ir.input import InputLayer, SpiNNakerSPIFInput
from bifrost.ir.parameter import ParameterContext

from bifrost.export import population, pynn


torch_context = TorchContext({"l": "0"})


def test_input_to_pynn():
    spif_layer = InputLayer("i", 1, SpiNNakerSPIFInput(x=1, y=2))
    variable = population.export_layer_input(spif_layer)
    assert variable == pynn.Statement(
        "l_i_0 = p.Population(None,p.external_devices.SPIFRetinaDevice(base_key=0,width=1,height=2,sub_width=32,sub_height=16,input_x_shift=16,input_y_shift=0))"
    )


def test_lif_to_pynn():
    l = LIFAlphaLayer("l", channels=1, neurons=10)
    lif_p = population.export_lif_neuron_type(l, torch_context)
    actual = population.export_layer_neuron(l, torch_context)
    assert actual == pynn.Statement(f"l_l_0 = p.Population(10, {lif_p.value})")


def test_lif_neuron_to_pynn():
    p = LIFAlphaLayer("l", 1, 10)
    s = population.export_lif_neuron_type(p, torch_context)
    expected = ""
    for p in torch_context.lif_parameters:
        expected = "_param_map['tau_mem_inv'](_params['0'][tau_mem_inv]),"
    expected = "_param_map['tau_mem_inv'](_params['0'][tau_mem_inv]),_param_map['tau_syn_inv'](_params['0'][tau_syn_inv]),_param_map['tau_syn_inv'](_params['0'][tau_syn_inv]),_param_map['v_reset'](_params['0'][v_reset]),_param_map['v_th'](_params['0'][v_th])"
    assert s == pynn.Statement(f"p.IF_curr_exp({expected})")


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
