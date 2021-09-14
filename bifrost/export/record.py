from bifrost.export.statement import Statement
from bifrost.ir.layer import NeuronLayer, Layer

def export_record(layer: Layer) -> Statement:
    variable_name = layer.variable('')

    if len(layer.record) == 0:
        return Statement()
    elif len(layer.record) == 1:
        recordings = f"\"{layer.record[0]}\""
    else:
        recordings = ", ".join(f"\"{r}\"" for r in layer.record)
        recordings = f"[{recordings}]"
    tab = " " * 4
    txt = (f"for channel in range({layer.channels}):\n"
           f"{tab}{variable_name}[channel].record({recordings})")

    return Statement(txt)
