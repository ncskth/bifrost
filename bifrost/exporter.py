from typing import TypeVar
from bifrost.ir.parameter import ParameterContext
from bifrost.export import connection, population, pynn
from bifrost.ir.layer import Layer
from bifrost.ir.network import Network


def export_network(network: Network, context: ParameterContext[str]) -> str:
    pynn_layers = [population.export_layer(l, context) for l in network.layers]
    connections = [
        connection.export_connection(c, context) for c in network.connections
    ]

    statements = []
    imports = set()
    for stmt in pynn_layers + connections:
        if stmt is not None:
            statements.append(stmt.value)
            imports = imports | set(stmt.imports)

    # Header
    header = pynn.pynn_header(timestep=network.timestep) + "\n" + context.preamble
    if len(network.config) > 0:
        header += "\n" + "\n".join(network.config)

    # Body
    body = "\n".join(list(imports) + statements)

    # Footer
    footer = pynn.pynn_footer(runtime=network.runtime)

    return "\n".join([header, body, footer])
