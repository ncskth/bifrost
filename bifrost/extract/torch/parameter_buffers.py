import torch

def set_parameter_buffers(model: torch.nn.Module):
    """:param model: the PyTorch network, will be modified in-line"""
    for k0 in model._modules:
        if isinstance(model._modules[k0], SequentialState):
            for k1 in model._modules[k0]._modules:
                set_parameter_buffers_per_layer(model._modules[k0]._modules[k1])
        elif not isinstance(model._modules[k0], torch.nn.CrossEntropyLoss):
            set_parameter_buffers_per_layer(model._modules[k0])


def set_parameter_buffers_per_layer(module: torch.nn.Module):
    """:param module: a 'layer' (PyTorch module)"""

    # required parameters for translation (neurons)
    _li = ['tau_mem_inv', 'tau_syn_inv', 'v_leak']
    _lif = _li + ['v_reset', 'v_th', 'alpha']
    params = {
        'LICell': {
            'LIFParameters': _lif,
            'LIParameters': _li,
        },
        'LIFCell': {
            'LIFParameters': _lif,
            'LIParameters': _li
        },
        # required parameters for convolution
        'Conv2d': [
            'kernel_size', 'in_channels', 'out_channels',
            'output_padding', 'padding', 'stride'
        ],
        # required parameters for average pooling
        'AvgPool2d': [
            'kernel_size', 'padding', 'stride'
        ],
    }
    mod_name = module.__class__.__name__
    if mod_name not in params:
        return

    if mod_name in ('LICell', 'LIFCell'):
        param_name = module.p.__class__.__name__
        param_list = params[mod_name][param_name]
        source_object = module.p
    else:
        param_list = params[mod_name]
        source_object = module
        
    for p in param_list:
        # get the value (usually a Tensor or Size)
        val = getattr(source_object, p)
        # and add it to _buffer so it will end in state_dict
        module._buffers[p] = val

