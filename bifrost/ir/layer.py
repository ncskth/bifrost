from dataclasses import dataclass


@dataclass
class Layer:
    name: str
    channels: int

    def variable(self, channel):
        return f"l_{self.name}_{channel}"


@dataclass
class LIFAlphaLayer(Layer):
    neurons: int
