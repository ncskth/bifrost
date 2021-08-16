from bifrost.translate.mlgenn.translations import (
    cells as cell_translations,
    layers as layer_translations,
)
from copy import deepcopy

def get_param(obj, trans):
    o = obj
    param_chain = trans[0].split('.')
    for p in param_chain:
        indices = None
        if '[' in p:
            l = p.split('[')
            ind = l[1][:-1].split(':')
            if len(ind) == 1:
                try:
                    indices = int(ind[0])
                except:
                    print("Indices not a slice nor integer {}".format(ind[0]))
                    indices = ind[0]
            else:
                indices = slice(*[None if ii =='' else int(ii)for ii in ind])
            p = l[0]

        if p == '':
            raise Exception('Parameter chain split encountered an empty element\n'
                            '{}\n{}'.format(param_chain, obj))
        o = getattr(o, p)
        if indices is not None:
            o = o[indices]

    if len(trans) == 2:
        return trans[1](o)
    else:
        return o


def get_params_from_dict(source, source_type: str, pdict: dict):
    target = pdict.pop('target')
    check_type = pdict.pop('check')
    assert source_type == get_param(source, check_type), \
            'not the same layer type as initially specified'

    params = {tgt_param: get_param(source, access)
              for tgt_param, access in pdict.items()}
    params['target'] = target

    return params


def get_cell_params(layer, cell_trans):
    layer_type = layer.__class__.__name__

    if layer_type == 'InputLayer':
        return {}

    cell_type = get_param(layer,  ('neurons.__class__.__name__', ))
    pdict = deepcopy(cell_trans[cell_type])
    return get_params_from_dict(layer, cell_type, pdict)


def get_layer_params(layer, layer_trans, cell_trans):
    layer_type = layer.__class__.__name__
    lparams = deepcopy(layer_trans[layer_type])

    if layer_type == 'Layer':
        layer_type = get_param(layer,
                               ('upstream_synapses[0].__class__.__name__', ))
        lparams = lparams[layer_type]

    cell_params = get_cell_params(layer, cell_trans)
    params = get_params_from_dict(layer, layer_type, lparams)

    params['cell'] = cell_params
    return params, layer_type


def extract(mlgenn_network):
    # NOTE: here layers is a list and we assume comes from a 'sequential'
    #       construction so they are 'in order' and indices are correct
    params = {}
    for layer_idx, layer in enumerate(mlgenn_network.layers):
        # todo: how do we get the input dataset name/structure?
        #       should we leave this problem for a command line parameter?
        name = layer.name
        layer_params, layer_type = get_layer_params(layer, layer_translations,
                                                    cell_translations)

        params[layer_idx] = {
            'pre': layer_idx - 1,
            'post': layer_idx,
            'name': name,
            'type': layer_type,
            'params': layer_params,
        }

    return params