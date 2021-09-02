SIM_NAME = 'spynn'

pynn_imports = [f"import spynnaker8 as {SIM_NAME}"]

def pynn_header(timestep=1.0):
    return f"""
{SIM_NAME}.setup({timestep})
"""


def pynn_footer(runtime):
    return f"""
{SIM_NAME}.run({runtime})
{SIM_NAME}.end()
"""
