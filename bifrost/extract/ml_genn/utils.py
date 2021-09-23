import numpy as np
from ml_genn.layers.enum import PadMode

def kernel_weight_transform(weights):
    return np.copy(weights)

def decode_conv_padding(synapse):
    kshape = np.asarray(synapse.conv_size, dtype='int')
    padding = synapse.conv_padding
    return decode_padding(kshape, padding)

def decode_pool_padding(synapse):
    kshape = np.asarray(synapse.pool_size, dtype='int')
    padding = synapse.pool_padding
    return decode_padding(kshape, padding)

def decode_padding(kshape, padding):
    if padding == PadMode.VALID:
        return np.asarray((0, 0), dtype='int')
    elif padding == PadMode.SAME:
        return ((kshape - 1) // 2)

def to_channels_first(synapse):
    n_in_channels = synapse.pool_output_shape[-1]
    post_pool_shape = synapse.pool_output_shape[:2]
    n_in_neurons = int(np.prod(post_pool_shape))
    n_out_neurons = synapse.units
    weights = synapse.weights
    pre = synapse.source()
    pre_shape = pre.shape[:2]
    pre_size = int(np.prod(pre_shape))
    out_weights = np.zeros((n_in_channels, 1, n_in_neurons, n_out_neurons))

    for channel in range(n_in_channels):
        pre_rows = np.repeat(np.arange(post_pool_shape[0]), post_pool_shape[1])
        pre_cols = np.tile(np.arange(post_pool_shape[1]), post_pool_shape[0])
        weight_matrix_rows = (pre_rows * post_pool_shape[1] * n_in_channels +
                              pre_cols * n_in_channels + channel)
        n_rows = len(weight_matrix_rows)
        out_weights[channel, 0] = weights[weight_matrix_rows, :].reshape((n_rows, n_out_neurons))

    return out_weights

def dense_weight_transform(synapse):
    if hasattr(synapse, 'pool_output_shape'):
        # NOTE: if the previous layer was convolutional we need to decode in the
        #       to move the channels first and index appropriatelly
        return to_channels_first(synapse)
    else:
        # NOTE: if the previous layer was NOT convolutional, then we do not
        #       have a poolling operation, so we can assume 1 channel in both
        #       pre and post. Last two dimensions are the pre and post sizes.
        weights = synapse.weights
        return weights.reshape((1, 1, weights.shape[0], synapse.units))



