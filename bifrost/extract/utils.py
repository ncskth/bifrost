import numpy as np


def size_from_shape(shape):
    return int(np.prod(shape))


def try_reduce_param(param):
    """
    NOTE: while most matrix-aware software handles with the scalar case
          transparently, sPyNNaker is not the case so we just intercept
          and convert such values prior to any sPyNNaker inspection
    """
    try:
        if np.allclose(param[:1], param):
            return np.asscalar(param[0])
    except Exception as e:
        if np.ndim(param) == 0:
            return param.item()
    else:
        return param


def get_param(object, translation, try_reduce=False):
    """ This function tries to parse access to object parameters with . (dot) and
        [] (indexing) in single place. These 'routes' to parameters are given as
        strings in (translations; see bifrost.extract.ml_genn.translations.py)
    :param object: Designed for ml_genn objects but probably could be used by others
    :param translation: A tuple containing at least: a 'route' or way to access a
    parameter/attribute from the object; it could also have a function which will
    operate on the resulting parameter
    :param try_reduce: Whether we should try to reduce into a scalar the
    extracted parameter
    :return parameter value: Either scalar or array or matrix
    """

    param_chain = translation[0].split('.')
    for part in param_chain:
        indices = None
        if '[' in part:
            l = part.split('[')
            ind = l[1][:-1].split(':')
            if len(ind) == 1:
                try:
                    indices = int(ind[0])
                except:
                    # print("Indices not a slice nor integer {}".format(ind[0]))
                    indices = ind[0]
            else:
                indices = slice(*[None if ii =='' else int(ii)for ii in ind])
            part = l[0]

        if part == '':
            raise Exception('Parameter chain split encountered an empty element\n'
                            '{}\n{}'.format(param_chain, object))
        object = getattr(object, part)
        if indices is not None:
            object = object[indices]

    if len(translation) == 2:
        # if we want to apply an operation to the resulting parameter/attribute
        # this is specified as the second part of the translation
        object = translation[1](object)

    if try_reduce:
        return try_reduce_param(object)

    return object

