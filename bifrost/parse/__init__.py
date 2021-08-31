from typing import Callable, TypeVar

from bifrost.ir import network

# 'template'-like behaviour
T = TypeVar("T")

Parser = Callable[[T], network.Network]
