from bifrost.ir.synapse import Synapse
from bifrost.ir.constants import SynapseShapes, SynapseTypes
import pytest


def test_synapse_type():
    for st in SynapseTypes:
        # this shouldn't throw errors
        synapse = Synapse(synapse_type=st)

    # with pytest.raises(TypeError) as e_info:
    #     # everything else should throw TypeError (standard dataclass doesn't validate)
    #     synapse = Synapse(synapse_type='abc')


def test_synapse_shape():
    for ss in SynapseShapes:
        # this shouldn't throw errors
        synapse = Synapse(synapse_shape=ss)

    # with pytest.raises(TypeError) as e_info:
    #     # everything else should throw TypeError (standard dataclass doesn't validate)
    #     synapse = Synapse(synapse_shape='abc')
