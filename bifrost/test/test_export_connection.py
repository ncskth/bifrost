from bifrost.ir.connection import *
from bifrost.export.connection import *


def test_a2a_connector_to_pynn():
    c = AllToAllConnector()
    assert export_connector(c).value == "p.AllToAllConnector()"


def test_conv2d_connector_to_pynn():
    k2 = torch.tensor([[1, 2], [3, 4]])
    c = ConvolutionConnector(k2)
    assert (
        export_connector(c).value
        == "p.ConvolutionConnector([[1,2],[3,4]],padding=[1,1])"
    )
    assert len(export_connector(c).imports) == 0


def test_connection_to_pynn():
    c = Connection("x", "y", AllToAllConnector(), StaticSynapse())
    actual = export_connection(c)
    assert (
        actual.value == "p.Projection(x, y, p.AllToAllConnector(), p.StaticSynapse())"
    )
    assert len(actual.imports) == 0
