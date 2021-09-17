from typing import TypeVar

import bifrost.export.record
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
    imports = set(pynn.pynn_imports + context.imports)
    preambles = set()
    for stmt in pynn_layers + connections:
        if stmt is not None:
            statements.append(stmt.value)
            imports |= set(stmt.imports)
            preambles |= set(stmt.preambles)

    # imports
    imps = "\n".join(sorted(list(imports)) + [""])
    # Header
    header = f"{pynn.pynn_header(timestep=network.timestep)}{context.preamble}"
    config = "\n".join(network.config)
    constraints = pynn.export_constraints(network.constraints)
    header_list = [header]

    if len(config):
        header_list.append(config)

    if len(constraints.value):
        header_list.append(constraints.value)

    header = "\n".join(header_list)

    # Body
    body = "\n".join(list(preambles) + statements)

    # Simulation run
    runner = pynn.pynn_runner(runtime=network.runtime)

    # Grab recordings
    get_records = bifrost.export.record.export_save_recordings(network).value

    # Footer
    footer = pynn.pynn_footer()

    return "\n".join([imps, header, body, runner, get_records, footer])
