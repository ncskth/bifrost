from bifrost.extract.ml_genn.translations import (
    cells as cell_translations,
    layers as layer_translations,
    connectors as connector_translations,
    CELL_TYPE_PARAM,
    SYNAPSE_TYPE_PARAM,
    LAYER_TYPE_PARAM,
)
from bifrost.extract.utils import get_param

from copy import deepcopy


def get_params_from_dict(source, source_type: str, pdict: dict,
                         ignore_keys=('target',), try_reduce=False):
    ignore_dict = {k: pdict.pop(k, None) for k in ignore_keys}
    check_type = pdict.pop('check')
    assert source_type == get_param(source, check_type), \
            'not the same layer type as initially specified'

    params = {tgt_param: get_param(source, access, try_reduce)
              for tgt_param, access in pdict.items()}
    params.update(ignore_dict)

    return params


def extract_cell_params(layer, cell_trans):
    layer_type = get_param(layer, LAYER_TYPE_PARAM)

    if layer_type == 'InputLayer':
        return {}

    cell_type = get_param(layer,  CELL_TYPE_PARAM)
    pdict = deepcopy(cell_trans[cell_type])
    ignore_keys = ['target', 'synapse_type', 'synapse_shape']
    return get_params_from_dict(layer, cell_type, pdict, ignore_keys,
                                try_reduce=True)


def extract_layer_params(layer, layer_trans, cell_trans):
    layer_type = get_param(layer, LAYER_TYPE_PARAM)
    lparams = deepcopy(layer_trans[layer_type])

    if layer_type == 'Layer':
        layer_type = get_param(layer,
                               SYNAPSE_TYPE_PARAM)
        lparams = lparams[layer_type]

    cell_params = extract_cell_params(layer, cell_trans)
    params = get_params_from_dict(layer, layer_type, lparams)

    params['cell'] = cell_params
    return params, layer_type


def extract_layer(layer, index):
    layer_params, layer_type = extract_layer_params(
                                    layer, layer_translations,
                                    cell_translations)

    conn_type = connector_translations.get(layer_type, None)

    return {
        'pre': index - 1,
        'post': index,
        'name': layer.name,
        'type': layer_type,
        'params': layer_params,
        'connector_type': conn_type,
    }

def extract_all(mlgenn_network):
    # NOTE: here layers is a list and we assume comes from a 'sequential'
    #       construction so they are 'in order' and indices are correct
    # TODO: how do we get the input dataset name/structure?
    #       should we leave this problem for a command line parameter?
    return {f"{layer_idx:03d}": extract_layer(layer, layer_idx)
            for layer_idx, layer in enumerate(mlgenn_network.layers)}

