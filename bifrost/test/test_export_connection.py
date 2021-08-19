from norse.torch.functional.lif import LIFParameters
from bifrost.export.pytorch import PytorchLightningContext
from bifrost.ir.connection import *
from bifrost.export.connection import *


class MockContext(ParameterContext):
    pass


def test_a2a_connector_to_pynn():
    cc = AllToAllConnector()
    c = MockContext()
    assert export_connector(cc, c).value == "p.AllToAllConnector()"


def test_conv2d_connection_to_pynn():
    k2 = torch.tensor([[1, 2], [3, 4]])
    cc = ConvolutionConnector(k2)
    c = MockContext()
    assert (
        export_connector(cc, c).value
        == "p.ConvolutionConnector([[1,2],[3,4]],padding=[1,1])"
    )
    assert len(export_connector(cc, c).imports) == 0


def test_lif2lif_connection_to_pynn():
    l1 = LIFLayer("x", 1, LIFParameters())
    l2 = LIFLayer("y", 2, LIFParameters())
    cc = Connection("l_x", "l_y", AllToAllConnector(), StaticSynapse())
    c = MockContext()
    actual = export_connection(cc, c)
    assert (
        actual.value == "p.Projection(l_x, l_y, p.AllToAllConnector(), p.StaticSynapse())"
    )
    assert len(actual.imports) == 0
