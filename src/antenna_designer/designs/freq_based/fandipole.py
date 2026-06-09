import logging
import math
from types import MappingProxyType

from ... import AntennaBuilder

logger = logging.getLogger(__name__)


C_LIGHT_MHZ_M = 299.792458


class Builder(AntennaBuilder):
    # Each band's element is a half-wave dipole sized to its own band
    # frequency: half_length = length_factor_NN × (c / freq_NN). The
    # factor ≈ 0.5 (slight end-effect shortening on the higher bands).
    # design_freq is here to satisfy the freq_based.* convention — the
    # adapter wires it into the design-freq UI control — but the
    # geometry pulls each arm's length from its own freq_NN so the
    # cone-shared feed remains tuneable per band.
    default_params = MappingProxyType(
        {
            "design_freq": 14.300,
            "freq": 28.57,
            "base": 7.0,
            "length_factor_20": 0.4892,
            "length_factor_17": 0.4994,
            "length_factor_15": 0.4984,
            "length_factor_12": 0.4971,
            "length_factor_10": 0.5004,
            "freq_20": 14.300,
            "freq_17": 18.1575,
            "freq_15": 21.383,
            "freq_12": 24.97,
            "freq_10": 28.47,
            "slope": 0.5,
            "ui_params": MappingProxyType(
                {
                    "sweep_policy": {
                        "anchor": "meas_freq",
                        "band_locked": True,
                    },
                    "length_factor_20": {"link_meas_freq_to_param": "freq_20"},
                    "length_factor_17": {"link_meas_freq_to_param": "freq_17"},
                    "length_factor_15": {"link_meas_freq_to_param": "freq_15"},
                    "length_factor_12": {"link_meas_freq_to_param": "freq_12"},
                    "length_factor_10": {"link_meas_freq_to_param": "freq_10"},
                    "freq_20": {"link_meas_freq_to_param": "freq_20"},
                    "freq_17": {"link_meas_freq_to_param": "freq_17"},
                    "freq_15": {"link_meas_freq_to_param": "freq_15"},
                    "freq_12": {"link_meas_freq_to_param": "freq_12"},
                    "freq_10": {"link_meas_freq_to_param": "freq_10"},
                }
            ),
        }
    )

    def build_wires(self):
        eps = 0.01

        radius = 0.12
        t0 = radius * math.sqrt(2)

        n = 5

        lst = [
            (math.cos(math.pi * i / 180), math.sin(math.pi * i / 180))
            for i in range(360 // (2 * n), 360, 360 // n)
        ]

        def build_path(lst, ns, ex):
            return ((a, b, ns, ex) for a, b in zip(lst[:-1], lst[1:]))

        def ry(p):
            return p[0], -p[1], p[2]

        Zc = 1 / math.sqrt(1 + self.slope**2)
        Zs = self.slope * Zc

        S = (0, eps, 0)
        T = ry(S)

        C = (S[0], S[1] + t0 * Zc, S[2] - t0 * Zs)

        A = [
            (C[0] + radius * x, C[1] + radius * y * Zs, C[2] + radius * y * Zc)
            for (x, y) in lst
        ]

        def dist(p0, p1):
            return math.sqrt(sum((x0 - x1) ** 2 for x0, x1 in zip(p0, p1)))

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("t0: %s dist: %s", t0, dist(S, C))
            logger.debug("t0: %s dists from C: %s", t0, [dist(C, a) for a in A])
            logger.debug("radius: %s dists from S: %s", radius, [dist(S, a) for a in A])

        # Per-band physical length = length_factor_NN × λ_NN, where
        # λ_NN = c / freq_NN. Ordered low-to-high freq (20m → 10m) to
        # match the spoke ordering in `lst`.
        band_specs = [
            (self.length_factor_10, self.freq_10),
            (self.length_factor_12, self.freq_12),
            (self.length_factor_15, self.freq_15),
            (self.length_factor_17, self.freq_17),
            (self.length_factor_20, self.freq_20),
        ]
        lengths = [factor * (C_LIGHT_MHZ_M / freq) for factor, freq in band_specs]

        ls = [(q / 2 - dist(S, a)) for (q, a) in zip(lengths, A)]

        B = [(AA[0], AA[1] + q * Zc, AA[2] - q * Zs) for q, AA in zip(ls, A)]

        Ay = [ry(p) for p in A]
        By = [ry(p) for p in B]

        for i in range(n):
            wire_length = dist(S, A[i]) + dist(A[i], B[i])
            logger.debug(
                "%d length %s %s %s",
                i,
                wire_length,
                lengths[i] / 2,
                (wire_length - lengths[i] / 2) / lengths[i],
            )

        n_seg0 = 21
        # The feed wire (T → S) needs at least one interior basis
        # function for pysim's triangular solver; n_seg=1 leaves zero
        # interior knots and triangular._feed_basis_indices crashes
        # with "argmin of empty sequence". 3 matches the convention
        # the rest of the design library uses for short feed wires.
        n_seg1 = 3

        tups = []
        for i in range(n):
            tups.extend(build_path([S, A[i], B[i]], n_seg0, None))
            tups.extend(build_path([T, Ay[i], By[i]], n_seg0, None))
        tups.append((T, S, n_seg1, 1 + 0j))

        new_tups = []
        for xoff, yoff, zoff in [(0, 0, self.base)]:
            new_tups.extend(
                [
                    (
                        (x0 + xoff, y0 + yoff, z0 + zoff),
                        (x1 + xoff, y1 + yoff, z1 + zoff),
                        ns,
                        ev,
                    )
                    for ((x0, y0, z0), (x1, y1, z1), ns, ev) in tups
                ]
            )

        return new_tups
