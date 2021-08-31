from typing import Optional, Tuple
from bifrost.ir.parameter import ParameterContext
from bifrost.ir.layer import NeuronLayer
from bifrost.ir.connection import *
from bifrost.export.statement import Statement, ConnectionStatement


def export_connection(
    connection: Connection, context: ParameterContext[str]) -> Statement:

    # Convolution and Dense are a sub-class of Static
    if not isinstance(connection.synapse, StaticSynapse):
        raise ValueError("Unknown Synapse", connection.synapse)

    # todo: this is not true, channel numbers are not necesarily the same
    # assert (
    #     connection.pre.channels == connection.post.channels
    # ), f"LIF -> LIF connection channels not equal! {connection.pre.channels} != {connection.post.channels}"

    projections = []
    for channel in range(connection.pre.channels):
        connector = export_connector(connection, channel, context)
        projection = f"{connection.variable(channel)} = p.Projection({connection.pre.variable(channel)}, {connection.post.variable(channel)}, {connector.value}, p.StaticSynapse())"
        projections.append(projection + f"\n{connector.configuration}")

    return Statement(
        value="\n".join(projections),
        imports=connector.imports,
    )


def export_connector(
    connection: Connection[Layer, Layer], channel: int, context: ParameterContext[str]
):
    if isinstance(connection.connector, MatrixConnector):
        return ConnectionStatement(
            f"p.AllToAllConnector()",
            configuration=f"{connection.variable(channel)}.set(weight={context.weights(connection.connector.weights_key, channel)})",
        )
    else:
        raise ValueError("Unknown connector: ", connection.connector)


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
