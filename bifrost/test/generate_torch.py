import sys
from tempfile import TemporaryFile
from typing import Tuple

import torch
import pytorch_lightning as pl
import norse.torch as norse

from bifrost.main import export


def generate_linear() -> Tuple[torch.nn.Module, TemporaryFile]:
    net = norse.SequentialState(torch.nn.Linear(2, 3), norse.LIFCell())
    tmp = TemporaryFile()
    torch.save(net.state_dict(), tmp)
    tmp.seek(0)
    return net, tmp


if __name__ == "__main__":
    weights_file = sys.argv[1]
    net, tmp = generate_linear()
    with open(weights_file, "wb") as fp:
        fp.write(tmp.read())
    #export(net, "(1, 2)", sys.stdout)
