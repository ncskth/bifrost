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

    statement = Statement(texts, imports=structure.imports)
    return statement


def export_dummy_test_input(layer: InputLayer, ctx: ParameterContext[str]) -> Statement:
    population = [f"{layer.variable('')} = {{0: DummyInputSource()}}"]
    recording = export_record(layer)
    statement = Statement(population)
    if len(recording.value):
        statement += recording
    return statement


def export_poisson_image_dataset_input(layer: InputLayer, ctx: ParameterContext[str]) -> Statement:
    tab = " " * 4
    source = layer.source
    start_sample = source.start_sample
    n_samples = source.num_samples
    variable_name = layer.variable("")
    parameter_function_name = f"__poisson_params_{variable_name}"
    load_function_name = f"__load_images_{variable_name}"
    load_function_text = (
        f"def {load_function_name}(start_sample, num_samples, num_channels):\n"
        f"{source.load_command_body}\n"
        "__images_dictionary, __classes = "
        f"{load_function_name}({start_sample}, {n_samples}, {layer.channels})\n"
    )

    transform_function_name = f"__images_to_rate_{variable_name}"
    transform_function_text = (
        f"def {transform_function_name}(images_dictionary):\n"
        f"{source.pixel_to_rate_transform}\n\n"
        f"__rates_dictionary = {transform_function_name}(__images_dictionary)\n"
    )
    on_time_ms = source.on_time_ms
    period_ms = on_time_ms + source.off_time_ms
    parameter_defines = ["\n".join(
        [f"def {parameter_function_name}(channel, rates_dictionary):",
         f"{tab}durations = np.ones(({layer.size}, {n_samples})) * {on_time_ms}",
         f"{tab}starts = np.repeat([np.arange({n_samples}) * {period_ms}], {layer.size}, axis=0)",
         f"{tab}return {{\"rates\": rates_dictionary[channel],",
         f"{tab * 3}\"durations\": durations, \"starts\": starts}}\n",]
    )]

    structure = export_structure(layer.source) # in this case the struct has the shape
    statement_text = (
        f"{variable_name} = {{channel: {SIMULATOR_NAME}.Population({layer.size}, \n"
        f"{tab}{SIMULATOR_NAME}.extra_models.SpikeSourcePoissonVariable( \n"
        f"{tab * 2}**{parameter_function_name}(channel, __rates_dictionary)), \n"
        f"{tab}structure={structure.value}, \n"
        f"{tab}label=f\"{variable_name}{{channel}}\") \n"
        f"{tab}for channel in range({layer.channels})}}\n"
    )
    source_defines_keys = sorted(source.defines.keys())
    sorted_source_defines = [source.defines[i] for i in source_defines_keys]
    load_transform = [load_function_text, transform_function_text]
    preambles = ["\n".join(load_transform + sorted_source_defines)]
    statement = Statement(statement_text,
        imports=structure.imports + source.imports,
        preambles=parameter_defines + list(structure.preambles) + preambles
    )


    return statement

