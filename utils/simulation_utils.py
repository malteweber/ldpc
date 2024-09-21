from ldpc import LDPC, InputType
from channels.binary_symmetric_channel import BinarySymmetricChannel
import numpy as np

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