import bifrost.export.utils
from bifrost.export.torch import TorchContext
from typing import List
import pytest

from norse.torch import LIFParameters

from bifrost.ir.layer import NeuronLayer, Layer
from bifrost.ir.input import InputLayer, SpiNNakerSPIFInput
from bifrost.ir.parameter import ParameterContext

from bifrost.export import population, statement, input
from bifrost.export.pynn import SIM_NAME

torch_context = TorchContext({"l": "0"})


def test_input_to_pynn():
    spif_layer = InputLayer("i", 1, 1,
                            source=SpiNNakerSPIFInput(shape=[2, 1]))
    actual = input.export_layer_input(spif_layer, torch_context)
    expected = (f"l_i_1_0 = {SIM_NAME}.Population(None,{SIM_NAME}.external_devices." 
                f"SPIFRetinaDevice(base_key=0,width=1,height=2,sub_width=32," 
                f"sub_height=16,input_x_shift=16,input_y_shift=0))\n")

    assert expected == actual.value


def test_lif_to_pynn():
    l = NeuronLayer(name="l", channels=1, size=10)
    lkey = "l_l_10_1" # l_{name}_{size}_{channels}
    var = l.variable(0)
    torch_context = TorchContext({lkey: "0"})
    lif_p = population.export_neuron_type(l, torch_context)
    struct = bifrost.export.utils.export_structure(l)
    actual = population.export_layer_neuron(l, torch_context)
    # population blocks end in a line break
    expected = f'{var} = {SIM_NAME}.Population(10, {lif_p.value}, structure={struct.value}, label="{var}")\n'
    assert actual.value == expected


def test_lif_neuron_to_pynn():
    l = NeuronLayer("l", 1, 10)
    lkey = "l_l_1_10" # l_{name}_{size}_{channels}
    torch_context = TorchContext({lkey: "0"})
    s = population.export_neuron_type(l, torch_context)
    expected = f"**(__nrn_params_{lkey}_f())"
    assert s.value == f"{SIM_NAME}.IF_curr_exp({expected})"


def test_dict_to_pynn_parameters():
    d = {"test": 12, "tau_m": "value"}
    actual = bifrost.export.utils.export_dict(d, join_str=',', n_spaces=0)
    expected = "test=12,tau_m='value'"
    assert actual.value == expected


def test_dict_to_pynn_parameters_fail():
    d = {"test": 12, 2: "value"}
    with pytest.raises(ValueError):
        actual = bifrost.export.utils.export_dict(d)


# def test_ann_to_graph():
#     ms = torch.nn.Sequential(torch.nn.Linear(28, 1), torch.nn.ReLU())
#     data = torch.zeros((1, 28, 28))
#     layers = model_to_graph(ms, data)
#     assert layers[0] == TorchLayer("linear", "Linear", 1)
#     assert layers[1] == TorchLayer("ReLU", "ReLU", 1)


# def test_export_snn():
# ms = torch.nn.Sequential(torch.nn.Linear(28, 1), norse.LICell())
