"""pysim-backed SimulationEngine. Currently impedance-only; far-field is
deferred to a follow-up PR that ports pysim's web/server.py directivity
math into the engine."""
from __future__ import annotations

import numpy as np
from pysim import TriangularPySim

from ..engine import SimulationEngine
from ..geometry import flat_wires_to_polylines

C_LIGHT = 299_792_458.0


class PysimEngine(SimulationEngine):
    supports_far_field = False

    def __init__(self, builder, *, solver=TriangularPySim, wire_radius=0.0005, n_qp_reg=4, n_qp_off=4):
        super().__init__(builder)
        if builder.build_tls():
            raise NotImplementedError("transmission-line cards not supported by PysimEngine yet")

        tups = builder.build_wires()
        translated = flat_wires_to_polylines(tups)
        self._polylines = translated["polylines"]
        self._edge_segments = translated["edge_segments"]
        self._feed_wire_index = translated["feed_wire_index"]
        self._feed_arclength = translated["feed_arclength"]
        self._feed_voltage = translated["feed_voltage"]
        self._solver = solver
        self._wire_radius = wire_radius
        self._n_qp_reg = n_qp_reg
        self._n_qp_off = n_qp_off

    def _make_solver(self, *, wavelength):
        return self._solver(
            wires=self._polylines,
            n_per_edge_per_wire=self._edge_segments,
            feed_wire_index=self._feed_wire_index,
            feed_arclength=self._feed_arclength,
            wavelength=wavelength,
            wire_radius=self._wire_radius,
            n_qp_reg=self._n_qp_reg,
            n_qp_off=self._n_qp_off,
        )

    @staticmethod
    def _wavelength_for(freq_mhz):
        return C_LIGHT / (freq_mhz * 1e6)

    def impedance(self):
        s = self._make_solver(wavelength=self._wavelength_for(self.builder.freq))
        z, _coeffs = s.compute_impedance()
        return [self._feed_voltage * z]

    def impedance_sweep(self, freqs):
        freqs = np.asarray(freqs, dtype=float)
        if freqs.ndim != 1 or freqs.size == 0:
            raise ValueError("freqs must be a 1-D non-empty array")
        # All k-independent setup happens in the constructor; build once.
        s = self._make_solver(wavelength=self._wavelength_for(freqs[0]))
        k_array = 2.0 * np.pi * freqs * 1e6 / C_LIGHT
        zs = s.compute_impedance_swept(k_array)
        return (self._feed_voltage * zs).reshape(-1, 1)
