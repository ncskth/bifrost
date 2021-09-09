from bifrost.ir.layer import NeuronLayer, Layer
from bifrost.ir.input import (InputLayer, SpiNNakerSPIFInput,
                              DummyTestInputSource, PoissonImageDataset)
from bifrost.ir.parameter import ParameterContext
from bifrost.export.statement import Statement
from bifrost.export.record import export_record
from bifrost.export.pynn import (SIM_NAME, PyNNSynapseShapes,
                                 PyNNSynapseTypes, PyNNNeuronTypes)
from bifrost.export.utils import export_structure

def export_layer_input(layer: InputLayer, ctx: ParameterContext[str]) -> Statement:
    source = layer.source
    if isinstance(source, SpiNNakerSPIFInput):
        statement = export_spif_input(layer, ctx)
    elif isinstance(source, DummyTestInputSource):
        statement = export_dummy_test_input(layer, ctx)
    elif isinstance(source, PoissonImageDataset):
        statement = export_poisson_image_dataset_input(layer, ctx)
    else:
        raise ValueError("Unknown input source", source)

    statement += Statement("")  # add carriage return
    return statement


def export_spif_input(layer: InputLayer, ctx: ParameterContext[str]) -> Statement:
    source = layer.source
    var = f"{layer.variable('')}"
    var_sp = " " * (len(var) + 4)
    tab = " " * 4
    texts = (f"{var} = {{channel: {SIM_NAME}.Population(None, \n"  
         f"{var_sp}{tab}{SIM_NAME}.external_devices.SPIFRetinaDevice(\n"  
         f"{var_sp}{tab}{tab}base_key=channel,width={source.x},height={source.y},\n"  
         f"{var_sp}{tab}{tab}sub_width={source.x_sub},sub_height={source.y_sub},\n"
         f"{var_sp}{tab}{tab}input_x_shift={source.x_shift},input_y_shift={source.y_shift}))\n"
         f"{var_sp}for channel in range({layer.channels})}}"
    )

    statement = Statement(texts)
    return statement


def export_dummy_test_input(layer: InputLayer, ctx: ParameterContext[str]) -> Statement:
    pop = [f"{layer.variable('')} = {{0: DummyInputSource()}}"]
    recs = export_record(layer)
    stt = Statement(pop)
    if len(recs.value):
        stt += recs
    return stt


def export_poisson_image_dataset_input(layer: InputLayer, ctx: ParameterContext[str]) -> Statement:
    source = layer.source
    param_def_name = "__nrn_params_"
    parameter_defines = [
        f"def {param_def_name}{layer.variable(channel)}():\n"
        f"    return dict()\n"
        for channel in range(layer.channels)
    ]
    struct = export_structure(layer.source) # in this case the struct has the shape
    statement = Statement([
            (f"{layer.variable(channel)} = {SIM_NAME}.Population({layer.size}, \n"
             f"    {SIM_NAME}.extra_models.SpikeSourcePoissonVariable( \n"
             f"        **{param_def_name}{layer.variable(channel)}()), \n"
             f"    structure={struct.value}, \n"
             f"    label=\"{layer.variable(channel)}\") \n")
            for channel in range(layer.channels)
        ],
        imports=struct.imports,
        preambles=parameter_defines + list(struct.preambles)
    )

    return statement

