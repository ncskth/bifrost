from bifrost.ir.input import InputLayer, ImageDataset, PoissonImageDataset

def adjust_runtime(runtime: float, input_layer: InputLayer) -> float:
    if isinstance(input_layer.source, ImageDataset):
        source = input_layer.source
        total_runtime = (
                (source.on_time_ms + source.off_time_ms) * source.num_samples
        )
        if total_runtime > runtime:
            return total_runtime

    return runtime