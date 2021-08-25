from compynator import *



# class PyNNLayer(NamedTuple):
#     cell: str
#     params: dict
#     label: str

# def conv2d_to_layer(conv2d):
#     return PyNNLayer()

# def layer_filter(obj):
#     return lambda l: isinstance(l, obj)

# counter = 0
# def layer_to_pynn(model, keys):
#     global counter
#     counter += 1
#     return lambda l: [PyNNLayer(model, {k: getattr(l,k) for k in keys}, f'{counter}_{model}')]

# conv2d = One.where(layer_filter(torch.nn.Conv2d))\
#             .value(layer_to_pynn('IF_curr_delta_conv', ('in_channels', 'kernel_size', 'out_channels','padding', 'stride', 'weight')))

# linear = One.where(layer_filter(torch.nn.Linear)).value(layer_to_pynn)

# skip = One.skip(Success(result=[]))

# p = Forward()
# p.is_(One.where(lambda a: a == 'a').value(lambda a: [0]) | One.skip(p))
# p('ba')

# layer = Forward()
# layer.is_(conv2d)
# layers = Forward()
# layer_list = One.where(lambda l: isinstance(l, list)).value(lambda l: Success([layers(module) for module in l]))
# layers.is_((layer_list | conv2d).repeat(1, value=[]))

# module[3:5]

# list(layers(module[2:4]))