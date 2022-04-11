import numpy as np
from ml_genn.layers.enum import PadMode


def kernel_weight_transform(weights):
    return np.copy(weights)


def decode_conv_padding(synapse):
    kshape = np.asarray(synapse.conv_size, dtype="int")
    padding = synapse.conv_padding
    return decode_padding(kshape, padding)


def decode_pool_padding(synapse):
    kshape = np.asarray(synapse.pool_size, dtype="int")
    padding = synapse.pool_padding
    return decode_padding(kshape, padding)


def decode_padding(kshape, padding):
    if padding == PadMode.VALID:
        return np.asarray((0, 0), dtype="int")
    elif padding == PadMode.SAME:
        return (kshape - 1) // 2


def to_channels_first(synapse):
    n_in_channels = synapse.pool_output_shape[-1]
    post_pool_shape = synapse.pool_output_shape[:2]
    n_in_neurons = int(np.prod(post_pool_shape))
    n_out_neurons = synapse.units
    out_shape = (n_in_neurons, n_out_neurons)
    pre_rows = np.repeat(np.arange(post_pool_shape[0]), post_pool_shape[1])
    pre_cols = np.tile(np.arange(post_pool_shape[1]), post_pool_shape[0])
    weights = synapse.weights
    out_weights = np.zeros((n_in_channels, 1, n_in_neurons, n_out_neurons))
    weights_rows_base = n_in_channels * (pre_rows * post_pool_shape[1] + pre_cols)

    for channel in range(n_in_channels):
        rows = weights_rows_base + channel
        out_weights[channel, 0] = weights[rows, :].reshape(out_shape)

    return out_weights


def dense_weight_transform(synapse):
    if hasattr(synapse, "pool_output_shape"):
        # NOTE: if the previous layer was convolutional we need to decode in the
        #       to move the channels first and index appropriatelly
        return to_channels_first(synapse)
    else:
        # NOTE: if the previous layer was NOT convolutional, then we do not
        #       have a poolling operation, so we can assume 1 channel in both
        #       pre and post. Last two dimensions are the pre and post sizes.
        weights = synapse.weights
        return weights.reshape((1, 1, weights.shape[0], weights.shape[1]))
