from channels.gaussian_channel import GaussianChannel
from ldpc import LDPC, InputType
from channels.binary_symmetric_channel import BinarySymmetricChannel
import numpy as np
from typing import Callable
from utils.graph_utils import create_tanner_graph
from utils.ldpc_utils import create_parity_check_matrix
import networkx as nx



def run_binary_channel_simulation(
        code: LDPC,
        f: float,
        nr_of_transmissions: int,
        bp_iterations: int,
        ) -> (float, float):
    """
    Run a simulation of a binary symmetric channel with the given parameters.
    """


    bsc = BinarySymmetricChannel(f)

    correctly_transmitted_bits = []
    convergence_list = []

    for i in range(nr_of_transmissions):


        x = np.random.randint(0, 2, code.generator_matrix.shape[1])
        y = code.encode(x)
        y_noisy = bsc.transmit(y)


        converged, y_dec = code.decode(y_noisy, f, bp_iterations, InputType.BINARY)

        if converged:
            correctly_transmitted_bits.append(len(np.where(y == y_dec)[0]))
            convergence_list.append(True)
        else:
            convergence_list.append(False)

    convergence_rate = sum(convergence_list) / len(convergence_list)
    nr_of_transmitted_bits = nr_of_transmissions*code.parity_check_matrix.shape[1]
    error_rate = 1 - sum(correctly_transmitted_bits) / nr_of_transmitted_bits

    return convergence_rate, error_rate

def run_gaussian_channel_simulation(
        code: LDPC,
        sigma: float,
        nr_of_transmissions: int,
        bp_iterations: int,
        ) -> (float, float):

    """
    Run a simulation of a gaussian channel with the given parameters.
    """

    channel = GaussianChannel(sigma)

    correctly_transmitted_bits = []
    convergence_list = []

    for i in range(nr_of_transmissions):

        x = np.random.randint(0, 2, code.generator_matrix.shape[1])
        y = code.encode(x)
        y_noisy = channel.transmit(y)

        converged, y_dec = code.decode(y_noisy, sigma, bp_iterations, InputType.CONTINUOUS)

        if converged:
            correctly_transmitted_bits.append(len(np.where(y == y_dec)[0]))
            convergence_list.append(True)
        else:
            convergence_list.append(False)

    convergence_rate = sum(convergence_list) / len(convergence_list)
    nr_of_transmitted_bits = nr_of_transmissions*code.parity_check_matrix.shape[1]
    error_rate = 1 - sum(correctly_transmitted_bits) / nr_of_transmitted_bits

    return convergence_rate, error_rate

def std_from_signal_to_noise_ration(snr: float) -> float:
    """
    Compute the standard deviation of the noise from the signal to noise ratio.
    """

    return 10**(-snr / 20)