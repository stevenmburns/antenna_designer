import numpy as np
import PyNEC as nec

from ..engine import FarField, SimulationEngine


class PyNECEngine(SimulationEngine):
    supports_far_field = True

    def __init__(self, builder):
        super().__init__(builder)
        self.tups = builder.build_wires()
        self.tls = builder.build_tls()
        self.excitation_pairs = None
        self._build_geometry()

    def __del__(self):
        # Release the nec_context handle if construction got that far.
        c = getattr(self, "c", None)
        if c is not None:
            del self.c

    def _build_geometry(self):
        conductivity = 5.8e7  # Copper
        ground_conductivity = 0.002
        ground_dielectric = 10

        self.c = nec.nec_context()
        geo = self.c.get_geometry()

        self.excitation_pairs = []
        for idx, (p0, p1, n_seg, ev) in enumerate(self.tups, start=1):
            geo.wire(idx, n_seg, p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], 0.0005, 1.0, 1.0)
            if ev is not None:
                self.excitation_pairs.append((idx, (n_seg + 1) // 2, ev))

        self.c.geometry_complete(0)

        for (idx1, seg1, idx2, seg2, impedance, length) in self.tls:
            self.c.tl_card(idx1, seg1, idx2, seg2, impedance, length, 0, 0, 0, 0)

        self.c.ld_card(5, 0, 0, 0, conductivity, 0.0, 0.0)
        self.c.gn_card(0, 0, ground_dielectric, ground_conductivity, 0, 0, 0, 0)

        for tag, sub_index, voltage in self.excitation_pairs:
            self.c.ex_card(0, tag, sub_index, 0, voltage.real, voltage.imag, 0, 0, 0, 0)

    def _set_freq_and_execute(self):
        self.c.fr_card(0, 1, self.builder.freq, 0)
        self.c.xq_card(0)

    def _impedances_at(self, freq_index, sum_currents=False):
        sc = self.c.get_structure_currents(freq_index)

        indices = []
        for tag, tag_index, voltage in self.excitation_pairs:
            matches = [(i, t) for (i, t) in enumerate(sc.get_current_segment_tag()) if t == tag]
            index = matches[tag_index - 1][0]
            indices.append((index, voltage))

        currents = sc.get_current()
        zs = [voltage / currents[idx] for idx, voltage in indices]

        if sum_currents:
            zs = [1 / sum(1 / z for z in zs)]

        return zs

    def impedance(self, sum_currents=False):
        self._set_freq_and_execute()
        return self._impedances_at(0, sum_currents=sum_currents)

    def impedance_sweep(self, freqs, sum_currents=False):
        freqs = np.asarray(freqs, dtype=float)
        if freqs.ndim != 1 or freqs.size == 0:
            raise ValueError("freqs must be a 1-D non-empty array")
        if freqs.size == 1:
            del_freq = 0.0
        else:
            steps = np.diff(freqs)
            del_freq = float(steps[0])
            if not np.allclose(steps, del_freq):
                raise ValueError(
                    "PyNECEngine.impedance_sweep requires evenly spaced freqs"
                )
        self.c.fr_card(0, freqs.size, float(freqs[0]), del_freq)
        self.c.xq_card(0)
        return np.array([self._impedances_at(i, sum_currents=sum_currents) for i in range(freqs.size)])

    def far_field(self, *, n_theta=90, n_phi=360, del_theta=1, del_phi=1):
        self._set_freq_and_execute()
        return self._collect_pattern(n_theta=n_theta, n_phi=n_phi, del_theta=del_theta, del_phi=del_phi)

    def _collect_pattern(self, *, n_theta, n_phi, del_theta, del_phi):
        assert 90 % n_theta == 0 and 90 == del_theta * n_theta
        assert 360 % n_phi == 0 and 360 == del_phi * n_phi

        self.c.rp_card(0, n_theta, n_phi + 1, 0, 5, 0, 0, 0, 0, del_theta, del_phi, 0, 0)

        thetas = np.linspace(0, 90 - del_theta, n_theta)
        phis = np.linspace(0, 360, n_phi + 1)

        rings = [
            [self.c.get_gain(0, theta_index, phi_index) for phi_index, _ in enumerate(phis)]
            for theta_index, _ in enumerate(thetas)
        ]

        return FarField(
            rings=rings,
            max_gain=self.c.get_gain_max(0),
            min_gain=self.c.get_gain_min(0),
            thetas=thetas,
            phis=phis,
        )
