import numpy as np
from .base_mask import BaseMask

class ChannelMask(BaseMask):
    """
    Drops (zeros) full dimensions of the observation vector.
    - If obs.ndim == 1 (vector): randomly choose a subset of indices to zero.
    - For images, mask entire channels (last dim).
    """
    def __init__(self, p: float, drop_ratio: float = 0.3, rng: np.random.RandomState = None):
        super().__init__(p)
        self.drop_ratio = float(np.clip(drop_ratio, 0.0, 1.0))
        self.rng = rng if rng is not None else np.random.RandomState()

    def apply(self, obs: np.ndarray) -> np.ndarray:
        masked = obs.copy()
        if obs.ndim == 1:
            n = obs.shape[0]
            k = max(1, int(round(self.drop_ratio * n)))
            idx = self.rng.choice(n, size=k, replace=False)
            masked[idx] = 0.0
        elif obs.ndim == 3:
            # (H,W,C) -> zero random channels
            c = obs.shape[-1]
            k = max(1, int(round(self.drop_ratio * c)))
            idx = self.rng.choice(c, size=k, replace=False)
            masked[..., idx] = 0.0
        else:
            # Fallback: elementwise fraction
            flat = masked.reshape(-1)
            k = max(1, int(round(self.drop_ratio * flat.size)))
            idx = self.rng.choice(flat.size, size=k, replace=False)
            flat[idx] = 0.0
            masked = flat.reshape(obs.shape)
        return masked
