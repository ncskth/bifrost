from bifrost.export.connection import export_connection
from bifrost.ir.layer import Layer
from bifrost.ir.network import Network

from bifrost.export import connection, population, pynn


def export_network(network: Network) -> str:
    pynn_layers = [population.export_layer(l) for l in network.layers]
    connections = [connection.export_connection(c) for c in network.connections]

    statements = []
    imports = set()
    for stmt in pynn_layers + connections:
        statements.append(stmt.value)
        imports = imports | set(stmt.imports)

    body = "\n".join(list(imports) + statements)
    header = pynn.pynn_header(timestep=network.timestep)
    if len(network.config) > 0:
        header += "\n" + "\n".join(network.config)
    footer = pynn.pynn_footer(runtime=network.runtime)

    return "\n".join([header, body, footer])
