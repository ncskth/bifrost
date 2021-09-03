from bifrost.ir.output import EthernetOutput, OutputLayer, OutputSink
from bifrost.ir.input import InputLayer, DummyTestInputSource
from bifrost.exporter import export_network
from bifrost.main import get_parser_and_saver

import torch

from bifrost.test.generate_torch import generate_linear


def test_pytorch_convert_network():
    net, weights = generate_linear()
    parser, saver = get_parser_and_saver(net)
    i = InputLayer("in", 1, 1, DummyTestInputSource([1, 1]))
    o = OutputLayer("out", 1, 1, sink=EthernetOutput())
    network, context, net_dict = parser(net, i, o)
    out = export_network(network, context)

    print(f"\n\n{out}")
