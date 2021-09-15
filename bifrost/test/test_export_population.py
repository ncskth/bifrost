import bifrost.export.pynn
import bifrost.export.utils
from bifrost.export.torch import TorchContext
from typing import List
import pytest

from norse.torch import LIFParameters

from bifrost.ir.layer import NeuronLayer, Layer
from bifrost.ir.input import InputLayer, InputSource, SpiNNakerSPIFInput
from bifrost.ir.output import OutputLayer, OutputSink
from bifrost.ir.parameter import ParameterContext

from bifrost.export import population, statement, input, output
from bifrost.export.pynn import SIMULATOR_NAME
from bifrost.text_utils import remove_blank as rb

torch_context = TorchContext({"l": "0"})


def test_input_to_pynn():
    spif_layer = InputLayer("i", 1, 1,
                            source=SpiNNakerSPIFInput(shape=[2, 1]))
    actual = input.export_layer_input(spif_layer, torch_context)
    expected = (f"l_i_1_ = {{channel: {SIMULATOR_NAME}.Population(None,{SIMULATOR_NAME}.external_devices." 
                f"SPIFRetinaDevice(base_key=channel,width=1,height=2,sub_width=32," 
                f"sub_height=16,input_x_shift=16,input_y_shift=0))\n"
                f"for channel in range(1)}}")

    assert rb(expected) == rb(actual.value)


def test_not_supported_input_source_to_pynn():
    class NotSupportedInputSource(InputSource):
        pass

    input_layer = InputLayer("i", 1, 1,
                             source=NotSupportedInputSource(shape=[2, 1]))
    with pytest.raises(ValueError) as e_info:
        actual = input.export_layer_input(input_layer, torch_context)


def test_not_supported_output_sink_to_pynn():
    class NotSupportedOutputSink(OutputSink):
        pass

    output_layer = OutputLayer("i", 1, 1,
                               sink=NotSupportedOutputSink())
    with pytest.raises(ValueError) as e_info:
        actual = output.export_layer_output(output_layer, torch_context)


def test_lif_to_pynn():
    l = NeuronLayer(name="l", channels=1, size=10)
    lkey = "l_l_10_1" # l_{name}_{size}_{channels}
    var = l.variable("")
    torch_context = TorchContext({lkey: "0"})
    lif_p = population.export_neuron_type(l, torch_context)
    struct = bifrost.export.pynn.export_structure(l)
    actual = population.export_layer_neuron(l, torch_context)
    # population blocks end in a line break
    expected = (f'{var} = {{channel: {SIMULATOR_NAME}.Population(10, {lif_p.value}, structure={struct.value}, '
                f'label=f"{var}{{channel}}")\n for channel in range(1)}}')
    assert rb(actual.value) == rb(expected)


def test_lif_neuron_to_pynn():
    l = NeuronLayer("l", 1, 10)
    lkey = "l_l_1_10" # l_{name}_{size}_{channels}
    torch_context = TorchContext({lkey: "0"})
    s = population.export_neuron_type(l, torch_context)
    expected = f"**(__nrn_params_{lkey}_f())"
    assert s.value == f"{SIMULATOR_NAME}.IF_curr_exp({expected})"


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
