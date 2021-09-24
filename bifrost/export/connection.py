from typing import Optional, Tuple
from bifrost.ir.parameter import ParameterContext
from bifrost.ir.layer import NeuronLayer, Layer
from bifrost.ir.constants import DefaultLayerKeys
from bifrost.ir.connection import (
    Connection, ConvolutionConnector, MatrixConnector, DenseConnector)
from bifrost.export.statement import Statement, ConnectionStatement
from bifrost.ir.synapse import (
    StaticSynapse, ConvolutionSynapse, DenseSynapse)
from bifrost.export.pynn import SIMULATOR_NAME

def export_connection(connection: Connection, context: ParameterContext[str],
                      join_str: str = ",\n", spaces: int = 4) -> Statement:

    # Convolution and Dense are a sub-class of Static
    if not isinstance(connection.post.synapse, StaticSynapse):
        raise ValueError("Unknown Synapse", connection.synapse)

    text_spaces = " " * spaces
    projections = []
    variable_name = connection.variable("", "")
    connector = export_connector(connection, "channel_in", "channel_out",
                                 context, spaces=12)
    synapse = export_synapse(connection)
    projection = [
            f"{SIMULATOR_NAME}.Projection(\n"
            f"{text_spaces * 2}{connection.pre.variable('')}[channel_in]",
            f"{text_spaces}{connection.post.variable('')}[channel_out]",
            f"{text_spaces}{connector.value}",
            f"{text_spaces}{synapse.value})",
    ]
    projection_text = f"{join_str}{text_spaces}".join(projection)
    pre_name = connection.pre.variable('')
    post_name = connection.post.variable('')
    statement_text = (
        f"{variable_name} = {{\n"
        f"channel_in: \n"
        f"{text_spaces}{{channel_out: {projection_text}\n"
        f"{text_spaces * 2}for channel_out in {post_name}\n"
        f"{text_spaces}}}\n"
        f"{text_spaces}for channel_in in {pre_name}\n"
        f"}}\n"
    )

    if len(connector.configuration):
        configuration_text = (
            f"for channel_in in {pre_name}:\n"
            f"{text_spaces}for channel_out in {post_name}:\n"
            f"{text_spaces * 2}{connector.configuration}\n"
        )
    else:
        configuration_text = ""

    return Statement(value=f"{statement_text}\n{configuration_text}",
                     imports=connector.imports,)

def export_synapse(connection: Connection[Layer, Layer]) -> Statement:
    synapse = connection.post.synapse
    if isinstance(synapse, ConvolutionSynapse):
        return Statement(f"{SIMULATOR_NAME}.Convolution()")
    elif isinstance(synapse, DenseSynapse):
        return Statement(f"{SIMULATOR_NAME}.PoolDense()")
    elif isinstance(synapse, StaticSynapse):
        return Statement(f"{SIMULATOR_NAME}.StaticSynapse()")
    else:
        raise ValueError(f"Unknown Synapse type: {synapse}")

def export_connector(connection: Connection[Layer, Layer],
                     channel_in: int, channel_out: int,
                     context: ParameterContext[str],
                     spaces: int = 8) -> Statement:
    connector = connection.connector

    if isinstance(connector, MatrixConnector):
        return export_all_to_all(connection, channel_in, channel_out,
                                 context, spaces)
    elif isinstance(connector, ConvolutionConnector):
        return export_conv(connection, channel_in, channel_out, context, spaces)
    elif isinstance(connector, DenseConnector):
        return export_dense(connection, channel_in, channel_out, context, spaces)
    else:
        raise ValueError(f"Unknown connector: {connector}")

def export_all_to_all(connection: Connection[Layer, Layer],
                      channel_in: int, channel_out: int,
                      context: ParameterContext[str],
                      spaces: int = 8) -> Statement:
    connector = connection.connector
    weights = context.linear_weights(connector.weights_key, channel_in, channel_out)
    variable_name = connection.variable("", "")
    return ConnectionStatement(
        f"{SIMULATOR_NAME}.AllToAllConnector()",
        configuration=f"{variable_name}[{channel_in}][{channel_out}].set(weight={weights})",
    )

def export_conv(connection: Connection[Layer, Layer],
                channel_in: int, channel_out: int,
                context: ParameterContext[str],
                spaces: int = 8) -> Statement:
    spaces_text = " " * spaces
    connector = connection.connector
    weights = context.conv2d_weights(
                connector.weights_key, channel_in, channel_out)
    strides = context.conv2d_strides(connector.weights_key)
    pool_shape, pool_stride = (
        context.conv2d_pooling(connector.pooling_key)
        if (connector.pooling_key != DefaultLayerKeys.POOLING)
        else ('None', 'None')
    )
    padding = context.conv2d_padding(connector.weights_key)
    stt = [f"{SIMULATOR_NAME}.ConvolutionConnector({weights}",
        f"strides={strides}",
        f"pool_shape={pool_shape}",
        f"pool_stride={pool_stride}",
        f"padding={padding})"
    ]
    return ConnectionStatement(f",\n{spaces_text}".join(stt))


def export_dense(connection: Connection[Layer, Layer],
                 channel_in: int, channel_out: int,
                 context: ParameterContext[str],
                 spaces: int = 8) -> Statement:
    spaces_text = " " * spaces
    connector = connection.connector
    weights = context.linear_weights(connector.weights_key, channel_in, channel_out)
    pool_shape, pool_stride = (
        context.conv2d_pooling(connector.pooling_key)
        if (connector.pooling_key != DefaultLayerKeys.POOLING)
        else ('None', 'None')
    )
    # todo: pool padding here needs to be added but I'm not sure if truly needed
    return ConnectionStatement(
        f"{SIMULATOR_NAME}.PoolDenseConnector({weights}, \n"
        f"{spaces_text}pool_shape={pool_shape}, \n"
        f"{spaces_text}pool_stride={pool_stride})",
    )


# def export_connector_conv_to_x(
#     connection: Connection[Conv2dLIFLayer, Layer],
#     connection: ConvolutionConnector,
#     context: ParameterContext[str],
# ) -> ConnectionStatement:
#     pass


# def export_connection_x_to_conv(
#     connection: Connection[Layer, Conv2dLIFLayer], context: ParameterContext[str]
# ) -> ConnectionStatement:
#     # Create n connections
#     statements = []
#     for channel in range(connection.post.channels):
#         var = f"{connection.variable}_{channel}"
#         projection = f"{var} = p.ConvolutionConnector({context.conv2d_weights(connection.post.variable)}"
