from typing import TypeVar

from bifrost.export.configurations import export_configurations
from bifrost.export.record import export_save_recordings
from bifrost.ir.parameter import ParameterContext
from bifrost.export import connection, population, pynn
from bifrost.ir.layer import Layer
from bifrost.ir.network import Network
from bifrost.text_utils import sanitize
from bifrost.export.constants import SAVE_VARIABLE_NAME

def export_network(network: Network, context: ParameterContext[str]) -> str:
    pynn_layers = [population.export_layer(l, context) for l in network.layers]
    connections = [
        connection.export_connection(c, context) for c in network.connections
    ]

    statements = []
    imports = set(pynn.pynn_imports + context.imports)

    preambles = set()
    any_record = [1 for lyr in network.layers if hasattr(lyr, "record") and len(lyr.record)]
    if any_record:
        preambles |= set([f"{SAVE_VARIABLE_NAME} = {{}}\n"])

    for stmt in pynn_layers + connections:
        if stmt is not None:
            statements.append(stmt.value)
            imports |= set(stmt.imports)
            preambles |= set(stmt.preambles)

    # imports
    imps = "\n".join(sorted(list(imports)) + [""])

    # Header (simulator.setup + whatever context needs to function)
    header = f"{pynn.pynn_header(timestep=network.timestep)}{context.preamble}"

    # Configs (after setup in PyNN)
    config = export_configurations(network, network.configuration).value

    # Body
    body = "\n".join(list(preambles) + statements)

    # Simulation run
    runner = pynn.export_split_run(network, network.runtime,
                                   pynn.pynn_runner).value

    # Grab recordings
    get_records = export_save_recordings(network).value
    if len(get_records):
        save_filename = f"{sanitize(network.name)}_recordings.npz"
        save_output_text = (f"np.savez_compressed(\"{save_filename}\", "\
                            f"**{SAVE_VARIABLE_NAME})")
    else:
        save_output_text = ""

    # Footer
    footer = pynn.pynn_footer()

    simulation_stages = [imps, header, body, config, runner, get_records,
                         save_output_text, footer]

    return "\n".join(simulation_stages)
