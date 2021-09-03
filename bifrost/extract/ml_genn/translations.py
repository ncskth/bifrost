import numpy as np
from bifrost.extract.utils import (size_from_shape)
from bifrost.extract.ml_genn.utils import dense_weight_transform

cells = {
    'IFNeurons': {
        'target': 'IFCell',
        'synapse_type': 'current',
        'synapse_shape': 'delta',
        'check': ('neurons.__class__.__name__', ),
        'v_thresh': ('neurons.nrn.extra_global_params[Vthr].view', np.copy),
        'v_rest': ('neurons.nrn.vars[Vmem].view', np.copy)
    },
}

layers = {
    # ML GeNN name
    'InputLayer': {
        'target': 'InputLayer',
        'check': ('__class__.__name__', ),
        'cell_type': ('neurons.__class__.__name__', ),
        'size': ('neurons.nrn.size', ),
        'shape': ('shape', )
    },
    'Layer': {
        'Conv2DSynapses': {
            'target': 'Conv2DLayer',
            'check': ('upstream_synapses[0].__class__.__name__', ),
            'cell_type': ('neurons.__class__.__name__', ),
            'size': ('shape[:2]', size_from_shape),
            'shape': ('shape', ),
            'n_channels': ('upstream_synapses[0].filters', ),
            'weights': ('upstream_synapses[0].weights', np.copy),
            'strides': ('upstream_synapses[0].conv_strides', ),
            'padding': ('upstream_synapses[0].conv_padding', ),
        },
        'AvePool2DConv2DSynapses': {
            'target': 'Conv2DLayer',
            'check': ('upstream_synapses[0].__class__.__name__', ),
            'cell_type': ('neurons.__class__.__name__', ),
            'size': ('shape[:2]', size_from_shape),
            'shape': ('shape', ),
            'n_channels': ('upstream_synapses[0].filters', ),
            'weights': ('upstream_synapses[0].weights', np.copy),
            'strides': ('upstream_synapses[0].conv_strides', ),
            'pool_shape': ('upstream_synapses[0].pool_size', ),
            'pool_stride': ('upstream_synapses[0].pool_strides', ),
            'padding': ('upstream_synapses[0].conv_padding', ),
        },
        'AvePool2DDenseSynapses': {
            'target': 'PoolDenseLayer',
            'check': ('upstream_synapses[0].__class__.__name__', ),
            'pool_shape': ('upstream_synapses[0].pool_size', ),
            'pool_stride': ('upstream_synapses[0].pool_strides', ),
            'size': ('shape[0]', ),
            'weights': ('upstream_synapses[0]', dense_weight_transform),
        },
        'DenseSynapses': {
            'target': 'PoolDenseLayer',
            'check': ('upstream_synapses[0].__class__.__name__', ),
            'size': ('shape[0]', ),
            'weights': ('upstream_synapses[0]', dense_weight_transform),
        },
    }
}

connectors = {
    'AvePool2DConv2DSynapses': 'ConvolutionConnector',
    'Conv2DSynapses': 'ConvolutionConnector',
    'AvePool2DDenseSynapses': 'DenseConnector',
    'DenseSynapses': 'DenseConnector',
}

input = {
    'image_dataset': {
    }
}

CELL_TYPE_PARAM = ('neurons.__class__.__name__', )
SYNAPSE_TYPE_PARAM = ('upstream_synapses[0].__class__.__name__', )
LAYER_TYPE_PARAM = ('__class__.__name__', )
