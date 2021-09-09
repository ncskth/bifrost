from bifrost.export.statement import Statement
from bifrost.ir.layer import NeuronLayer, Layer

def export_record(layer: Layer) -> Statement:
    var = layer.variable('')

    if len(layer.record) == 0:
        return Statement()
    elif len(layer.record) == 1:
        rs = f"\"{layer.record[0]}\""
    else:
        rs = ", ".join(f"\"{r}\"" for r in layer.record)
        rs = f"[{rs}]"
    tab = " " * 4
    txt = (f"for channel in range({layer.channels}):\n"
           f"{tab}{var}[channel].record({rs})")

    return Statement(txt)
