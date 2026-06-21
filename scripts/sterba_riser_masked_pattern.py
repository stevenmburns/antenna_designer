"""Custom far field of the all-wires Sterba with the 8 interior riser currents
ignored.

The inner verticals carry *large* currents (~2.5e-3, comparable to the peak),
but the A/B pair at each junction is ~180° out of phase, so their radiation
cancels. This script proves it: it solves the real all-wires antenna, then
recomputes the far field with the interior-riser segment currents zeroed. If
the verticals truly only phase the curtain (and don't radiate), both the
pattern and the max gain should be essentially unchanged.

Implementation: subclass PysimEngine and mask the riser segments in
`_segment_dipoles`, which `far_field` uses for BOTH the radiated-power
normalisation and the pattern grid — so the directivity stays self-consistent.

    .venv/bin/python scripts/sterba_riser_masked_pattern.py [out.png]
"""

import sys

import numpy as np

from antenna_designer.designs.wire.sterba import Builder
from antenna_designer.engines import PysimEngine
from antenna_designer.far_field import plot_patterns

FF_KW = dict(n_theta=90, n_phi=360, del_theta=1, del_phi=1)


def _interior_junction_ys(b):
    wavelength = 299.792458 / b.design_freq
    h = 0.5 * wavelength * b.length_factor
    q = 0.5 * h
    n = int(b.n_cells)
    yb = [0.0, q] + [q + k * h for k in range(1, n + 1)] + [2 * q + n * h]
    return np.array([yb[k] for k in range(1, n + 2)]), h  # the n+1 interior junctions


class RiserMaskedEngine(PysimEngine):
    """PysimEngine that drops the interior vertical-riser segment currents from
    the far-field integration (everything else — including the solve — is
    identical to the parent)."""

    def __init__(self, builder, **kw):
        super().__init__(builder, **kw)
        self._jys, self._h = _interior_junction_ys(builder)
        self.masked_segments = 0
        self.masked_current_l1 = 0.0

    def _segment_dipoles(self, sim, coeffs):
        mid, dr, i_mid = super()._segment_dipoles(sim, coeffs)
        vertical = (np.abs(dr[:, 2]) > np.abs(dr[:, 1])) & (
            np.abs(dr[:, 2]) > np.abs(dr[:, 0])
        )
        near_junction = (
            np.min(np.abs(mid[:, 1][:, None] - self._jys[None, :]), axis=1) < 0.1
        )
        mask = vertical & near_junction  # the 8 interior risers (A and B × n+1)
        self.masked_segments = int(mask.sum())
        self.masked_current_l1 = float(np.abs(i_mid[mask]).sum())
        i_mid = i_mid.copy()
        i_mid[mask] = 0.0
        return mid, dr, i_mid


def main():
    out = sys.argv[1] if len(sys.argv) > 1 else "sterba_riser_masked_pattern.png"

    b = Builder()
    full = PysimEngine(b, ground=None)
    masked = RiserMaskedEngine(Builder(), ground=None)

    ff_full = full.far_field(**FF_KW)
    ff_mask = masked.far_field(**FF_KW)

    print(f"interior riser segments zeroed : {masked.masked_segments}")
    print(
        f"riser current (Σ|I_mid|)       : {masked.masked_current_l1:.4e}  "
        f"(large — not negligible by magnitude)"
    )
    print(f"max_gain  all wires            : {ff_full.max_gain:.4f} dBi")
    print(f"max_gain  risers ignored       : {ff_mask.max_gain:.4f} dBi")
    print(
        f"Δ max_gain                     : {ff_mask.max_gain - ff_full.max_gain:+.4f} dB"
    )

    # Overlaid azimuth + elevation cuts, all-wires vs risers-ignored.
    plot_patterns(
        [ff_full.rings, ff_mask.rings],
        ["all wires", "risers ignored"],
        ff_full.thetas,
        ff_full.phis,
        fn=out,
    )
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
