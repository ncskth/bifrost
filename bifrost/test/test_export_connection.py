from norse.torch.functional.lif import LIFParameters
from bifrost.export.torch import TorchContext
from bifrost.ir.connection import *
from bifrost.export.connection import *


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


def test_matrix_connection_to_pynn():
    l1 = LIFAlphaLayer("x", 1, 1)
    l2 = LIFAlphaLayer("y", 1, 1)
    c = Connection(l1, l2, MatrixConnector("layer.weights"), StaticSynapse())
    actual = export_connection(c, torch_context)
    assert str(actual) == "c_x_y_0 = p.Projection(l_x_0, l_y_0, p.AllToAllConnector(), p.StaticSynapse())\nc_x_y_0.set(weight=_params['lw'][:0])"
    assert len(actual.imports) == 0
