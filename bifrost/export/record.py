from bifrost.export.statement import Statement
from bifrost.ir.layer import NeuronLayer, Layer

def export_record(layer: Layer, channel: int) -> Statement:
    if len(layer.record) == 0:
        return Statement()
    elif len(layer.record) == 1:
        rs = f"\"{layer.record[0]}\""
    else:
        rs = ", ".join(f"\"{r}\"" for r in layer.record)
        rs = f"[{rs}]"

    return Statement(f"{layer.variable(channel)}.record({rs})")
