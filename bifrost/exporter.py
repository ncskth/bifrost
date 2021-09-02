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
    imports = set(pynn.pynn_imports)
    preambles = set()
    for stmt in pynn_layers + connections:
        if stmt is not None:
            statements.append(stmt.value)
            imports |= set(stmt.imports)
            preambles |= set(stmt.preambles)

    # imports
    imps = "\n".join(sorted(list(imports)))
    # Header
    header = f"{pynn.pynn_header(timestep=network.timestep)}\n{context.preamble}"
    if len(network.config) > 0:
        cfg = '\n'.join(network.config)
        header = f"{header}\n{cfg}"

    # Body
    body = "\n".join(list(preambles) + statements)

    # Footer
    footer = pynn.pynn_footer(runtime=network.runtime)

    return "\n".join([imps, header, body, footer])
