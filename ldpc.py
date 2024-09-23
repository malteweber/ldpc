from enum import Enum

import numpy as np
import scipy.sparse as sp
from utils.ldpc_utils import probs, create_generator_matrix
from math import atanh, tanh, prod

class InputType(Enum):
    BINARY = 0
    LOG_LIKELIHOODS = 1

class LDPC:
    parity_check_matrix: np.ndarray
    n: int
    m: int
    generator_matrix: np.ndarray
    V: dict
    C: dict

    def __init__(self, parity_check_matrix: np.ndarray):
        G, H_new = create_generator_matrix(parity_check_matrix)
        self.parity_check_matrix = H_new
        self.m, self.n = self.parity_check_matrix.shape
        self.generator_matrix = G
        self.V = {j: [i for i in range(self.n) if self.parity_check_matrix[j, i] == 1] for j in range(self.m)}
        self.C = {i: [j for j in range(self.m) if self.parity_check_matrix[j, i] == 1] for i in range(self.n)}

    def encode(self, x: np.ndarray) -> np.ndarray:
        """
        Encode a word using the generator matrix
        """
        return self.generator_matrix @ x % 2


    def bp_tan_decode(self, x: np.ndarray, f: float, max_iter: int, input_type: InputType = InputType.BINARY) -> (bool, np.ndarray[int]):
        """
                Decode a transmitted word using belief propagation algorithm
                """

        def bp_step_tan(
                n: int,
                V: dict[int, list[int]],
                C: dict[int, list[int]],
                c: np.ndarray,
                q: np.ndarray,
        ) -> (np.ndarray, np.ndarray):
            """
            Perform a single iteration of the belief propagation algorithm
            """

            r = sp.dok_array((q.shape[1], q.shape[0]), dtype=float)
            q_posteriori = np.zeros(n, dtype=float)
            q_new = sp.dok_array(q.shape, dtype=float)

            for j, i_list in V.items():
                for i in i_list:
                    r[j, i] = 2 * np.atanh(np.prod([np.tanh(q[i_1, j] / 2) for i_1 in i_list if i_1 != i]))

            for i, j_list in C.items():
                for j in j_list:
                    q_new[i, j] = c[i] + sum([r[j_1, i] for j_1 in C[i] if j_1 != j])

            for i, j_list in C.items():
                q_posteriori[i] = c[i] + sum([r[j_1, i] for j_1 in j_list])

            return q_new, q_posteriori

        if input_type == InputType.BINARY:
            p = probs(x, f)
            c = np.log(p[:, 0] / p[:, 1])
        else:
            c = x

        q = np.column_stack([c for _ in range(self.m)])

        x_dec = np.zeros(self.n, dtype=int)
        for l in range(max_iter):

            q, q_posteriori = bp_step_tan(self.n, self.V, self.C, c, q)

            x_dec = np.where(q_posteriori <= 0, 1, 0)

            if np.all(self.parity_check_matrix @ x_dec % 2 == 0):
                return True, x_dec

        return False, x_dec
