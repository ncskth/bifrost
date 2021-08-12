import spynnaker8 as p
from pyNN.space import Grid2D
from spynnaker.pyNN.utilities.utility_calls import get_n_bits
import socket
from random import randint
from struct import pack
from time import sleep
from spinn_front_end_common.utilities.database import DatabaseConnection
send_fake_spikes = False
# Used if send_fake_spikes is True
sleep_time = 0.1
n_packets = 5
# IP_ADDR = "172.16.223.14"
IP_ADDR = "172.16.223.2"
PORT = 10000
# Run time if send_fake_spikes is False
run_time = 60000
if send_fake_spikes:
    run_time = (n_packets + 1) * sleep_time * 1000
# Constants
P_SHIFT = 15
Y_SHIFT = 0
X_SHIFT = 16
WIDTH = 640
HEIGHT = 480
SUB_WIDTH = 32
SUB_HEIGHT = 16
WEIGHT = 5
def send_retina_input():
    """ This is used to send random input to the Ethernet listening in SPIF
    """
    NO_TIMESTAMP = 0x80000000
    min_x = 0
    min_y = 0
    max_x = WIDTH - 1
    max_y = HEIGHT - 1
    polarity = 1
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    for _ in range(n_packets):
        sleep(sleep_time)
        n_spikes = randint(10, 100)
        data = b""
        for _ in range(n_spikes):
            x = randint(min_x, max_x)
            y = randint(min_y, max_y)
            packed = (
                NO_TIMESTAMP + (polarity << P_SHIFT) +
                (y << Y_SHIFT) + (x << X_SHIFT))
            print(f"Sending x={x}, y={y}, polarity={polarity}, packed={hex(packed)}")
            data += pack("<I", packed)
        sock.sendto(data, (IP_ADDR, PORT))
        sleep(sleep_time)
# Set up PyNN
p.setup(1.0)
# Set the number of neurons per core to a rectangle (creates 512 neurons per core)
p.set_number_of_neurons_per_core(p.IF_curr_exp, (SUB_WIDTH, SUB_HEIGHT))
if send_fake_spikes:
    # This is only used with the above to send data to the Ethernet
    connection = DatabaseConnection(send_retina_input, local_port=None)
    # This is used with the connection so that it starts sending when the simulation
    # starts
    p.external_devices.add_database_socket_address(
        None, connection.local_port, None)
# This is our convolution connector.  This one doesn't do much!
conn = p.ConvolutionConnector([[WEIGHT, WEIGHT, WEIGHT],
                               [WEIGHT, WEIGHT, WEIGHT],
                               [WEIGHT, WEIGHT, WEIGHT]], padding=(1, 1))
# This is our external retina device connected to SPIF
dev = p.Population(None, p.external_devices.SPIFRetinaDevice(
    base_key=0, width=WIDTH, height=HEIGHT, sub_width=SUB_WIDTH, sub_height=SUB_HEIGHT,
    input_x_shift=X_SHIFT, input_y_shift=Y_SHIFT))
# Create some convolutional "layers" (just 2, with 1 convolution each here)
pop = p.Population(WIDTH * HEIGHT, p.IF_curr_exp(), structure=Grid2D(WIDTH / HEIGHT))
#pop_2 = p.Population(WIDTH * HEIGHT, p.IF_curr_exp(), structure=Grid2D(WIDTH / HEIGHT))
# Record the spikes so we know what happened
pop.record("spikes")
#pop_2.record("spikes")
# Create convolution connections from the device -> first pop -> second pop
# These use the same connector, but could be different if desired
p.Projection(dev, pop, conn, p.Convolution())
#p.Projection(pop, pop_2, conn, p.Convolution())
# Run the simulation for long enough for packets to be sent
p.run(run_time)
# Get out the spikes
#spikes = pop.get_data("spikes").segments[0].spiketrains
spikes = pop.spinnaker_get_data("spikes")
#spikes_2 = pop_2.get_data("spikes").segments[0].spiketrains
# Raw data: pop.spinnaker_get_data("v")
# Note: record v (it will complain !)

# Tell the software we are done with the board
p.end()

print(type(spikes), spikes)

# Check which spikes have been received
# for i in range(len(spikes)):
#     #if len(spikes[i]) > 0 or len(spikes_2[i]) > 0:
#     if len(spikes[i]) > 0:
#         x = i % WIDTH
#         y = i // WIDTH
#         print(f"{x}, {y} = {spikes[i]}")
#         #print(f"{x}, {y} = {spikes[i]}; {spikes_2[i]}")