from bifrost.ir.parameter import ParameterContext
from bifrost.ir.layer import LIFLayer
import torch

from bifrost.ir.connection import *
from .pynn import Statement


def export_connection(connection: Connection, context: ParameterContext[str]):
    if not isinstance(connection.synapse, StaticSynapse):
        raise ValueError("Unknown Synapse", connection.synapse)
    pynn_connector = export_connector(connection.connector, context)
    return Statement(
        f"p.Projection({connection.pre}, {connection.post}, {pynn_connector.value}, p.StaticSynapse())",
        imports=pynn_connector.imports,
    )


def export_connector(connector: Connector, context: ParameterContext) -> Statement:
    if isinstance(connector, AllToAllConnector):
        return Statement("p.AllToAllConnector()")
    elif isinstance(connector, ConvolutionConnector):
        return Statement(
            f"p.ConvolutionConnector({_export_tensor(connector.weights)},padding={_export_tensor(connector.padding)})"
        )
    else:
        raise ValueError("Unknown connector: ", connector)


def _export_tensor(tensor: torch.Tensor) -> str:
    return (
        str(tensor)
        .strip()
        .replace("tensor(", "")
        .replace(")", "")
        .replace("\n", "")
        .replace(" ", "")
    )
