from bifrost.translate.utils import (size_from_shape, layer_attr)

cells = {
    'IFNeurons': {
        'target': 'IFCell',
        'parameters': {
            'v_thresh': ('.extra_global_params.Vthr',),
            'v_rest': ('.neruons.nrn.vars["Vmem"].view',)
        },
    }
}

layers = {
    # ML GeNN name
    'InputLayer': {
        'target': 'InputLayer',
        'check_layer': ('.__class__.__name__',),
        'cell_type': ('.neurons.__class__.__name__', ),
        'size': ('.neurons.nrn.size',),
        'shape': ('.shape',)
    },
    'Layer': {
        'Conv2DSynapses': {
            'target': 'Conv2DLayer',
            'check_synapse': ('.upstream_synapses[0].__class__.__name__', ),
            'cell_type': ('.neurons.__class__.__name__',),
            'size': ('.shape[:2]', size_from_shape),
            'shape': ('.shape',),
            'n_channels': ('.upstream_synapses[0].filters',),
            'weights': ('.upstream_synapses[0].weights',),
            'parameters': {
                'strides': ('.upstream_synapses[0].conv_strides',),
                'padding': ('.upstream_synapses[0].conv_padding',),
            }
        },
        'AvePool2DConv2DSynapses': {
            'target': 'Conv2DLayer',
            'check_synapse': ('.upstream_synapses[0].__class__.__name__',),
            'cell_type': ('.neurons.__class__.__name__',),
            'size': ('.shape[:2]', size_from_shape),
            'shape': ('.shape',),
            'n_channels': ('.upstream_synapses[0].filters',),
            'weights': ('.upstream_synapses[0].weights',),
            'parameters': {
                'strides': ('.upstream_synapses[0].conv_strides',),
                'pool_shape': ('.upstream_synapses[0].pool_size',),
                'pool_stride': ('.upstream_synapses[0].pool_strides',),
                'padding': ('.upstream_synapses[0].conv_padding',),
            }
        },
        'AvePool2DDenseSynapses': {
            'target': 'PoolDenseLayer',
            'check_synapse': ('.upstream_synapses[0].__class__.__name__', ),
            'parameters': {
                'pool_shape': ('.upstream_synapses[0].pool_size',),
                'pool_stride': ('.upstream_synapses[0].pool_strides',),
            },
            'size': ('.shape',),
            'weights': ('.upstream_synapses[0].weights',),
        },
        'DenseSynapses': {
            'target': 'PoolDenseLayer',
            'check_synapse': ('.upstream_synapses[0].__class__.__name__', ),
            'size': ('.shape',),
            'weights': ('.upstream_synapses[0].weights',),
        },
    }
}

input = {
    'image_dataset': {
    }
}