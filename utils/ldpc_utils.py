import numpy as np
from numpy import ndarray


def probs(x: np.ndarray, f: float) -> ndarray:
    """
    For a binary vector and a flip probabilty f, compute the probabilities of the transmitted vector being 0 or 1.
    """
    r = 1 - f

    result = np.zeros((len(x), 2), dtype=float)

    result[:, 0] = np.where(x == 0, r, f)
    result[:, 1] = np.where(x == 1, r, f)

    return result

def shuffled_columns(rng: np.random.Generator, x: np.ndarray) -> np.ndarray:
    """
    Shuffle the columns of a matrix.
    """
    c = x.copy()
    rng.shuffle(c, axis=1)
    return c


def create_parity_check_matrix(n: int, w_r: int, w_c: int, br: int = 100) -> np.ndarray | None:
    """
    Create a random parity check matrix with given parameters. If the matrix is not full rank, try again, up to br times.
    If no matrix is found, return None.
    """

    assert n % w_r != 0, "n must not be divisible by w_r, otherwise the matrix is not full rank"
    c = 0
    while c < br:
        c += 1

        rng = np.random.default_rng()

        first_block = np.array([
            [1 if i * w_r <= k < (i + 1) * w_r else 0 for k in range(n)]
            for i in range(int(n / w_r))
        ])

        H = np.vstack(
            [first_block] + [shuffled_columns(rng, first_block) for _ in range(w_c - 1)],
        )

        if np.linalg.matrix_rank(H) == min(H.shape) or c >= br:
            return H
        else:
            return None


def gauss_jordan_rref(H: np.ndarray) -> np.ndarray:
    """
    Compute the row reduced echelon form of a matrix H using the Gauss-Jordan algorithm.
    """

    H = np.copy(H)
    m, n = H.shape
    row = 0

    for i in range(n):
        if row >= m:
            break

        max_row = np.argmax(H[row:, i]) + row

        if H[max_row, i] == 0:
            continue

        H[[row, max_row]] = H[[max_row, row]]

        for r in range(m):
            if r != row:
                H[r, :] = H[r, :] - H[r, i] * H[row, :]

        H = H % 2

        row += 1

    return H

def create_generator_matrix(H: np.ndarray) -> (np.ndarray, np.ndarray):
    """
    Create a generator matrix G from a parity check matrix H.
    """

    H_c = H.copy()

    m, n = H_c.shape
    H_RREF = gauss_jordan_rref(H_c)
    col_swaps = permute_columns(H_RREF)

    for i, j in col_swaps:
        H_c[:, [i, j]] = H_c[:, [j, i]]

    H_c[:, range(n-m)], H_c[:, range(n-m, n)] = H_c[:, range(m, n)], H_c[:, range(m)]

    G = np.vstack([np.eye(n - m), H_RREF[:, m:]])

    return G, H_c


def is_canon_unit_vec(v: np.ndarray, index: int) -> bool:
    """
    Check if a vector is a canonical unit vector.
    """

    return v[index] == 1 and v.sum() == 1

def permute_columns(H: np.ndarray) -> list[tuple[int, int]]:
    """
    Permute columns of a matrix until the first part of the matrix is the identity.
    """

    m, n = H.shape

    col_switches = []
    for i in range(m):

        if is_canon_unit_vec(H[:, i], i):
            continue
        else:
            for j in range(i+1, n):
                if is_canon_unit_vec(H[:, j], i):
                    H[:, [i, j]] = H[:, [j, i]]
                    col_switches.append((i,j))
                    break
    return col_switches

def normal_pdf(x: np.ndarray, mean: float, std: float) -> float:
    """
    Compute the normal probability density.
    """
    return (1 / (std * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mean) / std) ** 2)

def binary_log_odds(y: np.ndarray, sigma: float) -> float:
    """
    Compute the binary log odds.
    """
    return np.log(normal_pdf(y, 0, sigma) / normal_pdf(y, 1, sigma))