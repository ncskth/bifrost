import numpy as np
def dense_weight_transform(synapse):
    weights = synapse.weights

    if hasattr(synapse, 'pool_output_shape'):
        # :NOTE: if pre was convolutional, then we must have N channels at
        #        the end of shape
        n_in_channs = synapse.pool_output_shape[-1]
        n_in = int(np.prod(synapse.pool_output_shape[:2]))
    else:
        # :NOTE: if pre was dense layers, we have a single channel
        n_in_channs = 1
        n_in = weights.shape[0]

    # :NOTE: since post is always dense, we have a single output channel
    n_out_channs = 1
    n_out = synapse.units
    reshaped = weights.reshape((n_in_channs, n_out_channs, n_in, n_out))
    return reshaped