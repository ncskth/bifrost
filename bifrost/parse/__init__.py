from typing import Callable, TypeVar

from bifrost.ir import network

T = TypeVar("T")

Parser = Callable[[T], network.Network]
