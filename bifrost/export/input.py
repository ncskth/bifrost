from bifrost.ir.layer import NeuronLayer, Layer
from bifrost.ir.input import (
    InputLayer,
    SpiNNakerSPIFInput,
    DummyTestInputSource,
    PoissonImageDataset,
    RandomPoissonSource,
    ImageDataset,
)
from bifrost.ir.parameter import ParameterContext
from bifrost.export.statement import Statement
from bifrost.export.pynn import (
    SIMULATOR_NAME,
    PyNNSynapseShapes,
    PyNNSynapseTypes,
    PyNNNeuronTypes,
    export_structure,
)
from bifrost.export.utils import export_list
from bifrost.text_utils import TAB
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
    texts = (
        f"{variable_name} = {{channel: {SIMULATOR_NAME}.Population(None, \n"
        f"{variable_spaces}{TAB}{SIMULATOR_NAME}.external_devices.SPIFRetinaDevice(\n"
        f"{variable_spaces}{TAB}{TAB}base_key=channel,width={source.x},height={source.y},\n"
        f"{variable_spaces}{TAB}{TAB}sub_width={source.x_sub},sub_height={source.y_sub},\n"
        f"{variable_spaces}{TAB}{TAB}input_x_shift={source.x_shift},input_y_shift={source.y_shift}))\n"
        f"{variable_spaces}for channel in range({layer.channels})}}"
    )

    statement = Statement(texts)
    return statement


def export_random_poisson_input(
    layer: InputLayer, ctx: ParameterContext[str]
) -> Statement:
    source = layer.source
    variable_name = f"{layer.variable('')}"
    variable_spaces = " " * (len(variable_name) + 4)
    size = int(np.prod(source.shape))
    if len(source.rates) > 1:
        rl = export_list(source.rates, q="")
        rates = f"[{rl}]"
    else:
        rates = source.rates[0]

    structure = export_structure(source)
    texts = (
        f"{variable_name} = {{channel: {SIMULATOR_NAME}.Population({size}, \n"
        f"{variable_spaces}{TAB}{SIMULATOR_NAME}.SpikeSourcePoisson(\n"
        f"{variable_spaces}{TAB}rate={rates}),\n"
        f"{variable_spaces}{TAB}structure={structure.value})\n"
        f"{variable_spaces}for channel in range({layer.channels})}}"
    )

    statement = Statement(texts, imports=structure.imports)
    return statement


def export_dummy_test_input(layer: InputLayer, ctx: ParameterContext[str]) -> Statement:
    population = [f"{layer.variable('')} = {{0: DummyInputSource()}}"]
    statement = Statement(population)
    return statement


def export_poisson_image_dataset_input(
    layer: InputLayer, ctx: ParameterContext[str]
) -> Statement:
    source = layer.source
    start_sample = source.start_sample
    n_samples = source.num_samples
    on_time = source.on_time_ms
    off_time = source.off_time_ms
    n_channels = layer.channels
    variable_name = layer.variable("")
    rates_dict_name = "__rates_dictionary"
    start_var = source.start_sample_variable
    n_samp_var = source.num_samples_variable
    n_chan_var = layer.num_channels_variable
    on_t_var = source.on_time_variable
    off_t_var = source.off_time_variable
    preamble = (
        f"{start_var} = {start_sample}\n"
        f"{n_samp_var} = {n_samples}\n"
        f"{n_chan_var} = {n_channels}\n"
        f"{on_t_var} = {on_time}\n"
        f"{off_t_var} = {off_time}\n"
    )
    parameter_function_name = f"__poisson_params_{variable_name}"
    parameter_function_args = (
        f"channel, {rates_dict_name}, {n_samp_var}, {on_t_var}, {off_t_var}"
    )
    load_function_name = f"__load_images_{variable_name}"
    images_variable_name = source.images_variable
    classes_variable_name = source.classes_variable
    load_function_text = (
        f"{preamble}\n"
        f"def {load_function_name}(start_sample, num_samples, num_channels):\n"
        f"{source.load_command_body}\n"
        f"{images_variable_name}, {classes_variable_name} = "
        f"{load_function_name}({start_var}, {n_samp_var}, {n_chan_var})\n"
    )

    transform_function_name = f"__images_to_rate_{variable_name}"
    transform_function_text = (
        f"def {transform_function_name}(images_dictionary):\n"
        f"{source.pixel_to_rate_transform}\n\n"
        f"{rates_dict_name} = {transform_function_name}({images_variable_name})\n"
    )
    on_time_ms = source.on_time_ms
    off_time_ms = source.off_time_ms
    period_ms = on_time_ms + source.off_time_ms
    parameter_defines = [
        f"\n{TAB}".join(
            [
                f"def {parameter_function_name}(channel, rates_dictionary, n_samples, on_time_ms, off_time_ms):",
                f"period_ms = on_time_ms + off_time_ms",
                f"durations = np.ones(({layer.size}, n_samples)) * on_time_ms",
                f"starts = np.repeat([np.arange(n_samples) * period_ms], {layer.size}, axis=0)",
                f'return {{"rates": rates_dictionary[channel],',
                f'{TAB * 2}"durations": durations, "starts": starts}}\n',
            ]
        )
    ]

    structure = export_structure(layer.source)  # in this case the struct has the shape
    statement_text = (
        f"{variable_name} = {{channel: {SIMULATOR_NAME}.Population({layer.size}, \n"
        f"{TAB}{SIMULATOR_NAME}.extra_models.SpikeSourcePoissonVariable( \n"
        f"{TAB * 2}**{parameter_function_name}({parameter_function_args})), \n"
        f"{TAB}structure={structure.value}, \n"
        f'{TAB}label=f"{variable_name}{{channel}}") \n'
        f"{TAB}for channel in {rates_dict_name}}}"
    )
    source_defines_keys = sorted(source.defines.keys())
    sorted_source_defines = [source.defines[i] for i in source_defines_keys]
    load_transform = [load_function_text, transform_function_text]
    preambles = ["\n".join(load_transform + sorted_source_defines)]
    statement = Statement(
        statement_text,
        imports=structure.imports + source.imports,
        preambles=parameter_defines + list(structure.preambles) + preambles,
    )

    return statement


def export_input_configuration(layer: InputLayer) -> Statement:
    source = layer.source
    source_type_name = source.__class__.__name__
    config_data = [f'"source_type": "{source_type_name}"']
    if isinstance(source, ImageDataset):
        config_data.extend(
            [
                f'"start_sample": {source.start_sample_variable}',
                f'"num_samples": {source.num_samples_variable}',
                f'"on_time_ms": {source.on_time_variable}',
                f'"off_time_ms": {source.off_time_variable}',
                f'"target_classes": {source.classes_variable}',
            ]
        )

    config_text = f",\n{TAB}".join(config_data)
    statement = f"{{\n{TAB}{config_text}\n}}"
    return Statement(f"{statement}")
