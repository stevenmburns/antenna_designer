"""Experiment: can explicit feedpoints replace the Sterba's inner verticals?

Delete the 8 interior risers, then drive each junction's 4 ports (j_At, j_Bt,
j_Ab, j_Bb) so they carry the *current* (amplitude + phase) the all-wires
reference carries at those points. Because the system is linear (I = Y·V), we
solve V = Y⁻¹·I_target and apply those voltages. If co-phased independently-fed
radiators reproduce the reference's ~10.5 dBi, the verticals were pure phasing;
the shortfall measures their own radiation/coupling.

Run incrementally, one junction pair at a time:
    .venv/bin/python scripts/sterba_driven_experiment.py 2
    .venv/bin/python scripts/sterba_driven_experiment.py 2 3
    .venv/bin/python scripts/sterba_driven_experiment.py            # all four
"""

import sys

import numpy as np

from antennaknobs.designs.wire import sterba, sterba_driven
from antennaknobs.engines import MomwireEngine

FF_KW = dict(n_theta=90, n_phi=360, del_theta=1, del_phi=1)
GROUND = None  # free space, to compare against the reference's 10.501 dBi


def _broadside_coherence(cd):
    """How coherently the horizontal radiators add toward broadside.

    The curtain fires in ±x; the broadside field is ∝ |Σ_k I_k·dy_k| over the
    y-directed (horizontal) segments. Normalising by Σ|I_k·dy_k| gives a 0..1
    coherence: ~1 when every section is co-phased (they add), ~0 when phases
    are scattered (they cancel). This is the quantity the gain ultimately
    depends on, with phase handled correctly (complex sum, no wrap artefacts)."""
    num = 0 + 0j
    den = 0.0
    for wc in cd:
        P, I = wc.knot_positions, wc.knot_currents
        for k in range(len(P) - 1):
            d = P[k + 1] - P[k]
            if abs(d[1]) > max(abs(d[0]), abs(d[2])):  # horizontal (along y)
                m = 0.5 * (I[k] + I[k + 1]) * d[1]
                num += m
                den += abs(m)
    return abs(num) / den if den else float("nan")


def _shared_params(active):
    p = dict(sterba_driven.Builder.default_params)
    p["active_junctions"] = active
    return p


def reference_sampler():
    """Run the all-wires reference once; return (max_gain, sample(coord))
    where sample returns the complex current at the knot nearest coord."""
    ref = sterba.Builder()
    # Keep geometry identical to the driven variant (same defaults already).
    eng = MomwireEngine(ref, ground=GROUND)
    gain = eng.far_field(**FF_KW).max_gain
    cd = eng.current_distribution()
    print(
        f"all-wires reference: max_gain = {gain:.3f} dBi, "
        f"broadside coherence = {_broadside_coherence(cd):.3f}"
    )
    pos = np.vstack([wc.knot_positions for wc in cd])
    cur = np.concatenate([wc.knot_currents for wc in cd])

    def sample(coord):
        d = np.linalg.norm(pos - np.asarray(coord), axis=1)
        k = int(np.argmin(d))
        return cur[k], float(d[k])

    return gain, sample


def run(active, ref_gain, sample):
    p = _shared_params(active)
    b0 = sterba_driven.Builder(p)
    _, jname = b0._junction_names()
    name_to_pt = {nm: pt for pt, nm in jname.items()}

    # Pass 1: target currents from the reference at each active port.
    eng0 = MomwireEngine(b0, ground=GROUND)
    wl = 299.792458 / b0.design_freq
    Y = eng0._compute_y_matrix(wl)
    port_to_idx = eng0._reducer.port_to_idx
    ports = sorted(port_to_idx)

    I_target = np.zeros(Y.shape[0], dtype=np.complex128)
    max_d = 0.0
    for nm in ports:
        i_ref, d = sample(name_to_pt[nm])
        I_target[port_to_idx[nm]] = i_ref
        max_d = max(max_d, d)

    # Pass 2: voltages that produce those currents (mutual coupling included).
    V = np.linalg.solve(Y, I_target)
    I_check = Y @ V
    match_err = np.max(np.abs(I_check - I_target)) / np.max(np.abs(I_target))
    cond = np.linalg.cond(Y)

    # Pass 3: evaluate the driven design.
    p2 = _shared_params(active)
    p2["feed_voltages"] = {nm: complex(V[port_to_idx[nm]]) for nm in ports}
    b = sterba_driven.Builder(p2)
    eng = MomwireEngine(b, ground=GROUND)
    gain = eng.far_field(**FF_KW).max_gain
    coh = _broadside_coherence(eng.current_distribution())

    label = "all" if active is None else ",".join(map(str, active))
    print(f"\n=== active junctions: {label}  ({len(ports)} feedpoints) ===")
    print(
        f"  reference current-sample max offset : {max_d:.3f} m (want « half-wave 5.27 m)"
    )
    print(f"  cond(Y)                             : {cond:.2e}")
    print(
        f"  port current-match error            : {match_err:.2e} (V = Y⁻¹ I round-trip)"
    )
    print(
        f"  |I_target| range                    : "
        f"{np.abs(I_target).min():.3e} .. {np.abs(I_target).max():.3e}"
    )
    print(
        f"  broadside coherence (driven)        : {coh:.3f} (reference ≈ 0.90; 1=co-phased)"
    )
    print(f"  max_gain (driven)                   : {gain:.3f} dBi")
    print(f"  max_gain (all-wires reference)      : {ref_gain:.3f} dBi")
    print(f"  gap                                 : {gain - ref_gain:+.3f} dB")
    return gain


def main():
    args = sys.argv[1:]
    active = [int(a) for a in args] if args else None

    ref_gain, sample = reference_sampler()

    if active is not None:
        run(active, ref_gain, sample)
    else:
        # Default: the full incremental sweep.
        for a in ([2], [2, 3], [1, 2, 3, 4]):
            run(a, ref_gain, sample)


if __name__ == "__main__":
    main()
