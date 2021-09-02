from typing import Optional, Tuple
from bifrost.ir.parameter import ParameterContext
from bifrost.ir.layer import NeuronLayer, Layer
from bifrost.ir.connection import (
    Connection, ConvolutionConnector, MatrixConnector, DenseConnector)
from bifrost.export.statement import Statement, ConnectionStatement
from bifrost.ir.synapse import (
    StaticSynapse, ConvolutionSynapse, DenseSynapse)
from bifrost.export.pynn import SIM_NAME

def export_connection(connection: Connection, context: ParameterContext[str],
                      join_str: str = ",\n", spaces: int = 4) -> Statement:

    # Convolution and Dense are a sub-class of Static
    if not isinstance(connection.post.synapse, StaticSynapse):
        raise ValueError("Unknown Synapse", connection.synapse)

    # todo: this is not true, channel numbers are not necesarily the same
    # assert (
    #     connection.pre.channels == connection.post.channels
    # ), f"LIF -> LIF connection channels not equal! {connection.pre.channels} != {connection.post.channels}"

    sp = " " * spaces
    projections = []
    for ch_in in range(connection.pre.channels):
        for ch_out in range(connection.post.channels):
            var = connection.variable(f"{ch_in}_{ch_out}")
            connector = export_connector(connection, ch_in, ch_out, context)
            synapse = export_synapse(connection)
            projection = [
                f"{var} = {SIM_NAME}.Projection(\n"
                f"{sp}{connection.pre.variable(ch_in)}",
                f"{connection.post.variable(ch_out)}",
                f"{connector.value}",
                f"{synapse.value})",
            ]
            proj = f"{join_str}{sp}".join(projection)
            projections.append(f"{proj}\n{connector.configuration}\n")

    return Statement(
        value="\n".join(projections),
        imports=connector.imports,
    )

def export_synapse(connection: Connection[Layer, Layer]) -> Statement:
    synapse = connection.post.synapse
    if isinstance(synapse, ConvolutionSynapse):
        return Statement(f"{SIM_NAME}.Convolution()")
    elif isinstance(synapse, DenseSynapse):
        return Statement(f"{SIM_NAME}.Dense()")
    elif isinstance(synapse, StaticSynapse):
        return Statement(f"{SIM_NAME}.StaticSynapse()")
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
    weights = context.linear_weights(str(connection.post), channel_in, channel_out)
    var = connection.variable(f"{channel_in}_{channel_out}")
    return ConnectionStatement(
        f"{SIM_NAME}.AllToAllConnector()",
        configuration=f"{var}.set(weight={weights})",
    )

def export_conv(connection: Connection[Layer, Layer],
                channel_in: int, channel_out: int,
                context: ParameterContext[str],
                spaces: int = 8) -> Statement:
    sp = " " * spaces
    weights = context.conv2d_weights(
                str(connection.post), channel_in, channel_out)
    strides = context.conv2d_strides(str(connection.post))
    pool_shape, pool_stride = context.conv2d_pooling(str(connection.post))
    # todo: padding here needs to be somewhat decoded but I'm not sure how to
    return ConnectionStatement(
        f"{SIM_NAME}.ConvolutionConnector({weights}, \n"
        f"{sp}strides={strides}, \n"
        f"{sp}pool_shape={pool_shape}, \n"
        f"{sp}pool_stride={pool_stride})",
    )

def export_dense(connection: Connection[Layer, Layer],
                 channel_in: int, channel_out: int,
                 context: ParameterContext[str],
                 spaces: int = 8) -> Statement:
    sp = " " * spaces
    weights = context.linear_weights(str(connection.post), channel_in, channel_out)
    pool_shape, pool_stride = context.conv2d_pooling(str(connection.post))
    # todo: padding here needs to be somewhat decoded but I'm not sure how to
    return ConnectionStatement(
        f"{SIM_NAME}.DenseConnector({weights}, \n"
        f"{sp}pool_shape={pool_shape}, \n"
        f"{sp}pool_stride={pool_stride})",
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
