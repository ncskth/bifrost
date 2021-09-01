SIM_NAME = 'spynn'

def pynn_header(timestep=1.0):
    return f"""
import spynnaker8 as {SIM_NAME}
{SIM_NAME}.setup({timestep})
"""


def pynn_footer(runtime):
    return f"""
{SIM_NAME}.run({runtime})
{SIM_NAME}.end()
"""
