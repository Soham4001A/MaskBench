import numpy as np
from abc import ABC, abstractmethod

class BaseMask(ABC):
    """
    Abstract base class for applying masks to observations.
    Mask is applied per-step with a given probability p in [0,1].
    """
    def __init__(self, p: float):
        self.p = float(np.clip(p, 0.0, 1.0))

    @abstractmethod
    def apply(self, obs: np.ndarray) -> np.ndarray:
        ...

    def maybe_apply(self, obs: np.ndarray) -> np.ndarray:
        if np.random.rand() < self.p:
            return self.apply(obs)
        return obs
