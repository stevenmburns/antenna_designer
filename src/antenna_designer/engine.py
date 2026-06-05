from abc import ABC, abstractmethod
from typing import ClassVar, NamedTuple

import numpy as np


class FarField(NamedTuple):
    rings: list
    max_gain: float
    min_gain: float
    thetas: np.ndarray
    phis: np.ndarray


class SimulationEngine(ABC):
    supports_far_field: ClassVar[bool] = False

    def __init__(self, builder):
        self.builder = builder

    @abstractmethod
    def impedance(self):
        ...

    @abstractmethod
    def impedance_sweep(self, freqs):
        ...

    def far_field(self, *, n_theta, n_phi, del_theta, del_phi):
        raise NotImplementedError(
            f"{type(self).__name__} does not support far-field computation"
        )
