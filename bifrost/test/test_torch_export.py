from bifrost.ir.output import EthernetOutput, OutputLayer, OutputSource
from bifrost.ir.input import InputLayer, InputSource
from bifrost.exporter import export_network
from bifrost.main import parse_torch

import torch

from bifrost.test.generate_torch import generate_linear


def test_pytorch_convert_network():
    net, weights = generate_linear()
    i = InputLayer("in", 1, 1, InputSource([1, 1]))
    o = OutputLayer("out", 1, 1, sink=EthernetOutput())
    network, context = parse_torch(net, i, o)
    out = export_network(network, context)
    print(out)
