from bifrost.export.statement import Statement
from bifrost.ir import Network
from bifrost.ir.layer import NeuronLayer, Layer
from bifrost.text_utils import sanitize

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

def export_grab_recordings_back(layer: Layer) -> Statement:
    if len(layer.record) == 0:
        return Statement()
    variable_name = layer.variable('')
    tab = " " * 4
    txt = (f"{{channel: {variable_name}[channel].get_data()"
           f" for channel in {variable_name}}}")

    return Statement(txt)


def export_save_recordings(network: Network):
    tab = " " * 4
    recordings_list = [f"\"{lyr.variable('')}\": {export_grab_recordings_back(lyr)}"
                       for lyr in network.layers if len(lyr.record)]
    if len(recordings_list):
        variable_name = "__recordings"
        variable_init_text = f",\n{tab}".join(recordings_list)
        save_filename = f"{sanitize(network.name)}_recordings.npz"
        save_output_text = f"np.savez_compressed(\"{save_filename}\", **{variable_name})"
        statement_text = (
            f"{variable_name} = {{{variable_init_text}\n}}\n"
            f"{save_output_text}\n"
        )
        return Statement(statement_text, imports=["import numpy as np"])
    else:
        return Statement()