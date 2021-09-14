"""
This is just a utility provided so that PyTorch/Norse networks keep variables
which are needed for sPyNNaker but are typically not stored through the state_dict
"""
import torch
import norse
DONT_PARSE_THESE_MODULES = (
    norse.SequentialState,
    torch.nn.modules.loss._Loss, # loss functions
    torch.nn.modules.batchnorm._NormBase # normalizers
)

def set_parameter_buffers(model: torch.nn.Module):
    """:param model: the PyTorch network, will be modified in-line"""
    # NOTE: .named_modules() provides a 'flattened' network (i.e. no loops/blocks)
    #      conveniently, modules end-up being 'links' to the actual modules in
    #      the network/model
    flattened_modules = dict(model.named_modules())
    for k, module in flattened_modules.items():
        if k == '' or isinstance(module, DONT_PARSE_THESE_MODULES):
            continue
        set_parameter_buffers_per_layer(module)

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

