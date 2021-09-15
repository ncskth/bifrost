from bifrost.export.torch import TorchContext
from bifrost.ir.connection import *
from bifrost.export.connection import *
from bifrost.export.pynn import SIMULATOR_NAME
from bifrost.text_utils import remove_blank as rb


class MockContext(ParameterContext):
    def weights(self, layer: str, channel: int) -> str:
        return "some_weights" + str(channel)

torch_context = TorchContext({"layer.weights": "lw"})

# def test_conv2d_connection_to_pynn():
#     k2 = torch.tensor([[1, 2], [3, 4]])
#     cc = ConvolutionConnector(k2)
#     c = MockContext()
#     assert (
#         export_connector(cc, "x", c)[0].value
#         == "p.ConvolutionConnector([[1,2],[3,4]],padding=[1,1])"
#     )
#     assert len(export_connector(cc, "x", c)[0].imports) == 0
#     assert export_connector(cc, "x", c)[1] is None

# weights are specified in torch way, need to add test for ml_genn
def test_matrix_connection_to_pynn():
    l1 = NeuronLayer("x", 1, 1)
    l2 = NeuronLayer("y", 1, 1)
    torch_context = TorchContext({"l_x_1_1": "0", "l_y_1_1": "lw"})
    c = Connection(l1, l2, MatrixConnector("layer.weights"))
    var = 'c_x___to__y_'
    actual = export_connection(c, torch_context, join_str=", ", spaces=0)
    # projections end in a line break
    expected = (
        f"{var} = {{channel_in:{{channel_out:{SIMULATOR_NAME}.Projection("
        f"l_x_1_[channel_in], l_y_1_[channel_out], " 
        f"{SIMULATOR_NAME}.AllToAllConnector(), {SIMULATOR_NAME}.StaticSynapse())" 
        f"for channel_out in range(1)}} for channel_in in range(1)}}" 
        f"for channel_in in range(1):"
        f"  for channel_out in range(1):"
        f"     {var}[channel_in][channel_out].set(" 
        f"     weight=_params[\"layer.weights.weight\"]"
        f"            [channel_out, channel_in].detach().numpy())"
                      # seems like torch uses inverted channels

    )

    assert rb(str(actual)) == rb(expected)
    assert len(actual.imports) == 0
