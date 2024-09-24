import numpy as np


class GaussianChannel:
    noise_std: float

    def __init__(self, noise_std: float):
        self.noise_std = noise_std

    def transmit(self, x: np.ndarray) -> np.ndarray:
        """
        Simulate transmission of a signal through the channel
        """
        return x + np.random.normal(0, self.noise_std, x.shape)