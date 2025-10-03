import numpy as np
from .base_mask import BaseMask

class RandomMask(BaseMask):
    """
    Elementwise random mask: each observation entry has prob=drop_ratio of being zeroed
    when a mask event triggers.
    """
    def __init__(self, p: float, drop_ratio: float = 0.3, rng: np.random.RandomState = None):
        super().__init__(p)
        self.drop_ratio = float(np.clip(drop_ratio, 0.0, 1.0))
        self.rng = rng if rng is not None else np.random.RandomState()

    def apply(self, obs: np.ndarray) -> np.ndarray:
        mask = self.rng.rand(*obs.shape) < self.drop_ratio
        masked = obs.copy()
        masked[mask] = 0.0
        return masked
