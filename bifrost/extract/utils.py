import numpy as np


def size_from_shape(shape):
    return int(np.prod(shape))


def get_param(obj, trans, try_reduce=False):
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
                    # print("Indices not a slice nor integer {}".format(ind[0]))
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
        o = trans[1](o)

    if try_reduce:
        return try_reduce(o)

    return o


def try_reduce(obj):
    try:
        if np.allclose(obj[:1], obj):
            return np.asscalar(obj[0])
    except Exception as e:
        if np.ndim(obj) == 0:
            return np.asscalar(obj)
    else:
        return obj

