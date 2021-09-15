from bifrost.parse.parse_ml_genn import (to_cell, to_synapse, to_connection,
                                         to_neuron_layer)
from bifrost.ir.constants import (SynapseTypes, SynapseShapes, NeuronTypes)
from bifrost.ir.cell import (Cell, IFCell, LICell, LIFCell)
from bifrost.ir.synapse import (Synapse, StaticSynapse, DenseSynapse, ConvolutionSynapse)

import pytest


def test_to_cell():
    translations = {'IFCell': IFCell, 'LICell': LICell, 'LIFCell': LIFCell}
    # currently we just support non-leaking integrate and fire neurons but this
    # should still work
    for cell_name in translations:
        actual = to_cell({'target': cell_name})
        assert isinstance(actual, Cell)
        assert isinstance(actual, translations[cell_name])

    # bifrost.ir has no cell called ABC
    with pytest.raises(NotImplementedError) as e_info:
        x = to_cell({'target': 'ABC'})



def test_to_synapse():
    # currently only supports static (non-trainable) synapses
    synapse_classes = {'conv2d': ConvolutionSynapse, 'dense': DenseSynapse}
    for sc in synapse_classes:
        for st in SynapseTypes:
            for ss in SynapseShapes:
                layer_dict = {
                    'type': sc,
                    'params': {
                        'cell': {
                            'synapse_type': st,
                            'synapse_shape': ss
                        }
                    }
                }

                synapse = to_synapse(layer_dict)
                assert isinstance(synapse, Synapse)
                assert isinstance(synapse, StaticSynapse)
                assert isinstance(synapse, synapse_classes[sc])

    # default synapse type, shape
    synapse = to_synapse({'type': 'dense', 'params': {'cell':{}}})

    assert synapse.synapse_shape == SynapseShapes.DELTA
    assert synapse.synapse_type == SynapseTypes.CURRENT

    # every other synapse type is not yet implemented
    with pytest.raises(NotImplementedError) as e_info:
        x = to_synapse({'type': 'xyz'})

    # description dictionary should contain 'params' key
    with pytest.raises(AttributeError) as e_info:
        x = to_synapse({'type': 'conv2d'})

    # description dictionary should contain 'params' -> 'cell' key
    with pytest.raises(AttributeError) as e_info:
        x = to_synapse({'type': 'conv2d', 'params': {}})


def test_to_neuron_layer_size_vs_shape():
    index = 0
    net_dict = {
        '000': {
            'params': {
                'size': 10,
                'shape': [10, 10]
            },
            'type': 'xyz'
        }
    }

    # this should throw an error due to size, shape mismatch
    with pytest.raises(Exception) as e_info:
        x = to_neuron_layer(index, net_dict)

