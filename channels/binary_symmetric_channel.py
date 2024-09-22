import numpy as np
from random import choices


class BinarySymmetricChannel:
    f: float

    def __init__(self, f: float):
        self.f = f

    def transmit(self, x: np.ndarray) -> np.ndarray:
        """
        Simulate transmission of a signal through the channel
        """
        y = x.copy()
        for i in range(x.size):
            e = choices([0, 1], [1 - self.f, self.f], k=1)[0]
            y[i] = (y[i] + e) % 2
        return y

