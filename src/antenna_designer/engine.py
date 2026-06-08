from abc import ABC, abstractmethod
from typing import ClassVar, NamedTuple

import numpy as np


class FarField(NamedTuple):
    rings: list
    max_gain: float
    min_gain: float
    thetas: np.ndarray
    phis: np.ndarray


class WireCurrents(NamedTuple):
    """Per-wire knot positions + complex currents at the solve frequency.

    Engines decompose geometry differently — PysimEngine returns one
    entry per polyline (post-translator), PyNECEngine returns one entry
    per build_wires() tuple. Callers (e.g. the web UI) treat each entry
    as an independent rendering primitive rather than assuming the lists
    are aligned across engines.
    """

    knot_positions: np.ndarray  # (M, 3) float
    knot_currents: np.ndarray  # (M,)   complex


class SimulationEngine(ABC):
    supports_far_field: ClassVar[bool] = False

    def __init__(self, builder):
        self.builder = builder

    @abstractmethod
    def impedance(self): ...

    @abstractmethod
    def impedance_sweep(self, freqs): ...

    def far_field(self, *, n_theta, n_phi, del_theta, del_phi):
        raise NotImplementedError(
            f"{type(self).__name__} does not support far-field computation"
        )

    def current_distribution(self):
        """Return list[WireCurrents] at the builder's frequency."""
        raise NotImplementedError(f"{type(self).__name__} does not expose currents yet")
