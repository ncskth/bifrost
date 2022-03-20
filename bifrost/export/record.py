from bifrost.export.statement import Statement
from bifrost.export.utils import export_layer_shape
from bifrost.ir import Network
from bifrost.ir.layer import Layer
from bifrost.ir.input import ImageDataset
from bifrost.export.constants import SAVE_VARIABLE_NAME
from bifrost.export.input import export_input_configuration

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
    txt = (f"for channel in {variable_name}:\n"
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


def export_save_recordings(network: Network) -> Statement:
    tab = " " * 4
    recordings_list = [f"\"{lyr.variable('')}\": {export_grab_recordings_back(lyr)}"
                       for lyr in network.layers if hasattr(lyr, "record") and len(lyr.record)]
    if len(recordings_list):
        input_layer = network.layers[0]
        input_config = export_input_configuration(input_layer)

        record_variable_name = "__recordings"
        record_init_text = f",\n{tab}".join(recordings_list)

        shapes_variable_name = "__network_shapes"
        shapes_list = [f"\"{lyr.variable('')}\": {export_layer_shape(lyr)}"
                       for lyr in network.layers if len(lyr.record)]
        shapes_init_text = f",\n{tab}".join(shapes_list)

        statement_text = (
            f"{record_variable_name} = {{\n{tab}{record_init_text}\n}}\n"
            f"{shapes_variable_name} = {{\n{tab}{shapes_init_text}\n}}\n"
            f"{SAVE_VARIABLE_NAME}[\"recordings\"] = {record_variable_name}\n"
            f"{SAVE_VARIABLE_NAME}[\"shapes\"] = {shapes_variable_name}\n"
            f"{SAVE_VARIABLE_NAME}[\"input_configuration\"] = {input_config}"
        )
        return Statement(statement_text, imports=["import numpy as np"])
    else:
        return Statement()