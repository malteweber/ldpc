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

    bsc = BinarySymmetricChannel(f)

    bit_error_rates = []
    convergence_list = []

    for i in range(nr_of_transmissions):

        x = np.random.randint(0, 2, code.generator_matrix.shape[1])
        y = code.encode(x)
        y_noisy = bsc.transmit(y)

        try:
            converged, y_dec = code.bp_tan_decode(y_noisy, f, bp_iterations, InputType.BINARY)
        except:
            continue

        if converged:
            bit_error_rates.append(len(np.where(y != y_dec)[0]) / len(y))
            convergence_list.append(True)
        else:
            convergence_list.append(False)

    convergence_rate = sum(convergence_list) / len(convergence_list)
    error_rate = 0 if len(bit_error_rates) == 0 else sum(bit_error_rates) / len(bit_error_rates)

    return convergence_rate, error_rate


def test_girth_metric_binary_symmetric(
        girth_metric: Callable[[nx.Graph], float],
        nr_of_matrices: int,
        n: int,
        w_r: int,
        w_c: int,
        nr_of_transmissions: int,
        bp_iterations: int,
        flip_rates) -> (list[float], list[float]):
    H, g, girths = choose_from_random_by_girth_metric(girth_metric, nr_of_matrices, n, w_r, w_c)

    code = LDPC(H)

    convergence_rates = []
    bit_error_rates = []

    for f in flip_rates:
        c_rate, e_rate = run_binary_channel_simulation(
            code,
            f,
            nr_of_transmissions,
            bp_iterations
        )

        convergence_rates.append(c_rate)
        bit_error_rates.append(e_rate)
    return convergence_rates, bit_error_rates


def choose_from_random_by_girth_metric(
        girth_metric: Callable[[nx.Graph], float],
        nr_of_matrices: int,
        n: int,
        w_r: int,
        w_c: int) -> (np.ndarray, float, list[float]):
    """
    Choose the best parity check matrix from nr_of_matrices random ones, based on the girth_metric.
    """
    best = (None, 0)
    girths = []

    for i in range(nr_of_matrices):
        H = create_parity_check_matrix(n, w_r, w_c)
        if H is not None:
            g = girth_metric(create_tanner_graph(H))
            girths.append(g)

            if g > best[1]:
                best = H, g

    return best[0], best[1], girths