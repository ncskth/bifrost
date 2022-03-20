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
    net = NetworkBase()
    l1 = NeuronLayer("x", 1, 1)
    l2 = NeuronLayer("y", 1, 1)
    l1_n_ch = 1
    l2_n_ch = 1
    l1_name = "l_x_1_"
    l2_name = "l_y_1_"
    torch_context = TorchContext({f"{l1_name}{l1_n_ch}": "0",
                                  f"{l2_name}{l2_n_ch}": "lw"})
    c = Connection(l1, l2, MatrixConnector("layer.weights"), network=net)
    var = 'c_x___to__y_'
    actual = export_connection(c, torch_context, join_str=", ", spaces=0)
    # projections end in a line break
    expected = (
        f"{var} = {{channel_in:{{channel_out:{SIMULATOR_NAME}.Projection("
        f"l_x_1_[channel_in], l_y_1_[channel_out], " 
        f"{SIMULATOR_NAME}.AllToAllConnector(), {SIMULATOR_NAME}.StaticSynapse())" 
        f"for channel_out in {l2_name}}} for channel_in in {l1_name}}}" 
        f"for channel_in in {l1_name}:"
        f"  for channel_out in {l2_name}:"
        f"     {var}[channel_in][channel_out].set(" 
        f"     weight=_params[\"layer.weights.weight\"]"
        f"            .reshape((1, 1, -1))[:, channel_in, :].detach().numpy().T)"
                      # seems like torch uses inverted channels

    )

    assert rb(str(actual)) == rb(expected)
    assert len(actual.imports) == 0
