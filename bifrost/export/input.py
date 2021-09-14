from bifrost.ir.layer import NeuronLayer, Layer
from bifrost.ir.input import (InputLayer, SpiNNakerSPIFInput,
                              DummyTestInputSource, PoissonImageDataset,
                              RandomPoissonSource)
from bifrost.ir.parameter import ParameterContext
from bifrost.export.statement import Statement
from bifrost.export.record import export_record
from bifrost.export.pynn import (SIMULATOR_NAME, PyNNSynapseShapes,
                                 PyNNSynapseTypes, PyNNNeuronTypes, export_structure)
from bifrost.export.utils import export_list
import numpy as np

def export_layer_input(layer: InputLayer, ctx: ParameterContext[str]) -> Statement:
    source = layer.source
    if isinstance(source, SpiNNakerSPIFInput):
        statement = export_spif_input(layer, ctx)
    elif isinstance(source, DummyTestInputSource):
        statement = export_dummy_test_input(layer, ctx)
    elif isinstance(source, PoissonImageDataset):
        statement = export_poisson_image_dataset_input(layer, ctx)
    elif isinstance(source, RandomPoissonSource):
        statement = export_random_poisson_input(layer, ctx)
    else:
        raise ValueError("Unknown input source", source)

    statement += Statement("")  # add carriage return
    return statement


def export_spif_input(layer: InputLayer, ctx: ParameterContext[str]) -> Statement:
    source = layer.source
    variable_name = f"{layer.variable('')}"
    variable_spaces = " " * (len(variable_name) + 4)
    tab = " " * 4
    texts = (f"{variable_name} = {{channel: {SIMULATOR_NAME}.Population(None, \n"  
         f"{variable_spaces}{tab}{SIMULATOR_NAME}.external_devices.SPIFRetinaDevice(\n"  
         f"{variable_spaces}{tab}{tab}base_key=channel,width={source.x},height={source.y},\n"  
         f"{variable_spaces}{tab}{tab}sub_width={source.x_sub},sub_height={source.y_sub},\n"
         f"{variable_spaces}{tab}{tab}input_x_shift={source.x_shift},input_y_shift={source.y_shift}))\n"
         f"{variable_spaces}for channel in range({layer.channels})}}"
    )

    statement = Statement(texts)
    return statement


def export_random_poisson_input(layer: InputLayer, ctx: ParameterContext[str]) -> Statement:
    source = layer.source
    variable_name = f"{layer.variable('')}"
    variable_spaces = " " * (len(variable_name) + 4)
    tab = " " * 4
    size = int(np.prod(source.shape))
    if len(source.rates) > 1:
        rl = export_list(source.rates, q="")
        rates = f"[{rl}]"
    else:
        rates = source.rates[0]

    structure = export_structure(source)
    texts = (f"{variable_name} = {{channel: {SIMULATOR_NAME}.Population({size}, \n"  
         f"{variable_spaces}{tab}{SIMULATOR_NAME}.SpikeSourcePoisson(\n"
         f"{variable_spaces}{tab}rate={rates}),\n"
         f"{variable_spaces}{tab}structure={structure.value})\n"
         f"{variable_spaces}for channel in range({layer.channels})}}"
    )

    statement = Statement(texts,
                          imports=structure.imports)
    return statement


def export_dummy_test_input(layer: InputLayer, ctx: ParameterContext[str]) -> Statement:
    population = [f"{layer.variable('')} = {{0: DummyInputSource()}}"]
    recording = export_record(layer)
    statement = Statement(population)
    if len(recording.value):
        statement += recording
    return statement


def export_poisson_image_dataset_input(layer: InputLayer, ctx: ParameterContext[str]) -> Statement:
    source = layer.source
    parameter_function_name = "__nrn_params_"
    parameter_defines = [
        f"def {parameter_function_name}{layer.variable(channel)}():\n"
        f"    return dict()\n"
        for channel in range(layer.channels)
    ]
    structure = export_structure(layer.source) # in this case the struct has the shape
    statement = Statement([
            (f"{layer.variable(channel)} = {SIMULATOR_NAME}.Population({layer.size}, \n"
             f"    {SIMULATOR_NAME}.extra_models.SpikeSourcePoissonVariable( \n"
             f"        **{parameter_function_name}{layer.variable(channel)}()), \n"
             f"    structure={structure.value}, \n"
             f"    label=\"{layer.variable(channel)}\") \n")
            for channel in range(layer.channels)
        ],
        imports=structure.imports,
        preambles=parameter_defines + list(structure.preambles)
    )

    return statement

