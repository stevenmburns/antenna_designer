"""pysim-backed SimulationEngine. Impedance via TriangularPySim;
far-field/directivity ported from pysim/web/server.py:_compute_directivity_norm.
"""

from __future__ import annotations

import numpy as np
from pysim import TriangularPySim

from ..engine import FarField, SimulationEngine, WireCurrents
from ..geometry import flat_wires_to_polylines
from ..network import (
    TL,
    Driven,
    Load,
    PortAtEdge,
    PortVirtual,
    TwoPort,
    load_impedance,
    load_series_admittance,
)


def _parity_for_solver(solver, solver_kwargs):
    """The basis types have fixed parity expectations:
      - TriangularPySim (tent / linear B-spline) → even (feed straddles 2 segs)
      - BSplinePySim degree=1 → same as triangular → even
      - BSplinePySim degree=2 → quadratic → odd
      - SinusoidalPySim → odd
    Anything else falls through as "any" (no coercion)."""
    name = getattr(solver, "__name__", "")
    if name == "TriangularPySim":
        return "even"
    if name == "SinusoidalPySim":
        return "odd"
    if name == "BSplinePySim":
        degree = (solver_kwargs or {}).get("degree", 2)
        return "even" if int(degree) == 1 else "odd"
    return "any"


C_LIGHT = 299_792_458.0
EPS0 = 8.854_187_817e-12


def _polyline_knots(polyline, npe_list):
    """Concatenated per-edge knot positions (shared corners deduped).
    Mirrors pysim/web/server.py:_polyline_knots."""
    parts = []
    for i, n_e in enumerate(npe_list):
        seg = np.linspace(polyline[i], polyline[i + 1], n_e + 1)
        parts.append(seg if i == 0 else seg[1:])
    return np.vstack(parts)


def _normalise_ground(ground):
    if ground is None or ground == "free":
        return None
    if ground == "pec":
        return ("pec",)
    if isinstance(ground, tuple) and len(ground) == 3 and ground[0] == "finite":
        return ground
    raise ValueError(f"unrecognised ground spec: {ground!r}")


class PysimEngine(SimulationEngine):
    supports_far_field = True

    def __init__(
        self,
        builder,
        *,
        solver=TriangularPySim,
        wire_radius=0.0005,
        solver_kwargs=None,
        ground=None,
        ground_z=0.0,
    ):
        """
        solver:
          A pysim solver class — TriangularPySim (default), SinusoidalPySim,
          or BSplinePySim. Different bases trade speed vs impedance fidelity;
          on the hentenna sinusoidal is typically closer to PyNEC at modest
          segmentation than triangular.
        solver_kwargs:
          Dict of solver-specific kwargs passed straight to the constructor
          (e.g. `{"n_qp_reg": 8, "n_qp_off": 8}` for TriangularPySim, or
          `{"n_qp_const": 16}` for SinusoidalPySim). None = solver defaults.
        ground:
          None or "free"           — no ground (default)
          "pec"                    — PEC plane at z=ground_z (image method)
          ("finite", eps_r, sigma) — far-field uses PEC image + Fresnel
                                     coefficients on the reflected component;
                                     impedance solve still uses PEC because
                                     pysim only models PEC ground. Cross-
                                     validation against PyNEC's gn_card(0,...)
                                     is approximate.
        """
        super().__init__(builder)

        self._solver = solver
        self._solver_kwargs = dict(solver_kwargs) if solver_kwargs else {}
        # Per-instance parity: triangular wants even, sinusoidal odd,
        # bspline depends on degree. Set before _coerce_wire_tuples runs.
        self.segment_parity = _parity_for_solver(self._solver, self._solver_kwargs)

        # build_wires() must run before build_tls() — some designs populate
        # self.tls inside build_wires() (delta_looparray_with_tls).
        tups = self._coerce_wire_tuples(builder.build_wires())
        self._network = builder.build_network()
        self._tls = [] if self._network is not None else list(builder.build_tls())

        # Resolve TL endpoint tags into augmented tups: any tag whose ev was
        # nullified gets a passive feed (V=0) so pysim assembles the full
        # multi-port Y matrix. Driven ports keep their original voltages.
        augmented_tags = set()
        if self._tls:
            tups = list(tups)
            for tag1, _seg1, tag2, _seg2, _z0, _length in self._tls:
                for tag in (tag1, tag2):
                    t = tups[tag - 1]
                    p0, p1, n_seg, ev = t[0], t[1], t[2], t[3]
                    if ev is None:
                        tups[tag - 1] = (p0, p1, n_seg, 0 + 0j)
                        augmented_tags.add(tag)

        translated = flat_wires_to_polylines(tups)
        self._polylines = translated["polylines"]
        self._edge_segments = translated["edge_segments"]
        self._feeds = translated["feeds"]
        self._feed_names = translated["feed_names"]
        self._junctions = translated["junctions"]
        self._wire_radius = wire_radius
        self._ground = _normalise_ground(ground)
        self._ground_z = ground_z if self._ground is not None else None

        # Map TL tags to feed indices for the legacy build_tls() path.
        self._tag_to_feed = {}
        feed_i = 0
        for tag, t in enumerate(tups, start=1):
            ev = t[3]
            if ev is not None:
                self._tag_to_feed[tag] = feed_i
                feed_i += 1
        # 0-based feed indices that are TL passive ports (V=0, floating).
        self._tl_passive_feed_idx = {self._tag_to_feed[t] for t in augmented_tags}

        if self._network is not None:
            self._init_network()

    def _init_network(self):
        """Build port-index maps and validate that every PortAtEdge resolves
        to a translated feed name. Real ports come first (matching pysim's
        feed list); virtual ports are appended after."""
        net = self._network
        feed_name_to_idx = {n: i for i, n in enumerate(self._feed_names) if n}

        self._port_to_idx = {}
        for name, port in net.ports.items():
            if isinstance(port, PortAtEdge):
                if name not in feed_name_to_idx:
                    raise ValueError(
                        f"network port {name!r} is a PortAtEdge but no edge in "
                        f"build_wires() carries that name; named edges: "
                        f"{sorted(feed_name_to_idx)}"
                    )
                self._port_to_idx[name] = feed_name_to_idx[name]

        # Virtual ports are indexed after the real feeds.
        next_idx = len(self._feeds)
        for name, port in net.ports.items():
            if isinstance(port, PortVirtual):
                self._port_to_idx[name] = next_idx
                next_idx += 1
        self._n_total_ports = next_idx

        # 0-based driven port indices and their applied voltages.
        self._driven_port_idx = []
        self._driven_voltages = []
        for src in net.sources:
            if not isinstance(src, Driven):
                raise NotImplementedError(f"unknown source type: {src!r}")
            self._driven_port_idx.append(self._port_to_idx[src.port])
            self._driven_voltages.append(complex(src.voltage))

        # A Load on a port modifies the MoM Z-diagonal of that segment via
        # Sherman-Morrison (equivalent to NEC2's ld_card on segment k). The
        # right boundary condition at a loaded port is V_external = 0 (no
        # source attached to the segment's external terminal) — NOT
        # I_external = 0, which is the right BC for TL passive ports. With
        # I_ext = 0 forced, the Sherman-Morrison update's effect on the
        # driven-port impedance cancels out algebraically (you can derive
        # it: V_passive picks up a factor (1 − α·Y_kk) that divides out).
        # So track loaded ports separately and pin their V = 0 in the
        # reduction. They take precedence over the floating-passive default.
        self._loaded_port_idx = set()
        for br in net.branches:
            if isinstance(br, Load):
                self._loaded_port_idx.add(self._port_to_idx[br.port])

    def _make_solver(self, *, wavelength):
        return self._solver(
            wires=self._polylines,
            n_per_edge_per_wire=self._edge_segments,
            feeds=self._feeds,
            wavelength=wavelength,
            wire_radius=self._wire_radius,
            ground_z=self._ground_z,
            junctions=self._junctions or None,
            **self._solver_kwargs,
        )

    @staticmethod
    def _wavelength_for(freq_mhz):
        return C_LIGHT / (freq_mhz * 1e6)

    def _tl_admittance_2x2(self, z0, length, wavelength):
        """Lossless ideal-TL nodal admittance between its two terminals.

        For electrical length θ = 2π·length/λ:
            Y_TL = 1/(j Z0 sin θ) · [[cos θ, -1], [-1, cos θ]]
        Singular at sin θ = 0 (TL is a half-wavelength multiple); raise
        rather than return garbage so callers can pick a different length.
        """
        theta = 2.0 * np.pi * length / wavelength
        s, c = np.sin(theta), np.cos(theta)
        if abs(s) < 1e-12:
            raise ValueError(
                f"TL length {length} is ~kλ/2 at f={C_LIGHT / wavelength / 1e6:.4f} MHz "
                "(sin βl ≈ 0); admittance is singular"
            )
        scale = 1.0 / (1j * z0 * s)
        return scale * np.array([[c, -1.0], [-1.0, c]], dtype=np.complex128)

    def _apply_tls(self, Y, wavelength):
        """Y + per-TL stamps at the corresponding feed-index pairs."""
        Y = Y.copy()
        for tag1, _s1, tag2, _s2, z0, length in self._tls:
            a = self._tag_to_feed[tag1]
            b = self._tag_to_feed[tag2]
            y_tl = self._tl_admittance_2x2(z0, length, wavelength)
            Y[np.ix_([a, b], [a, b])] += y_tl
        return Y

    def _resolve_feed_voltages(self, Y_total):
        """Return the full per-feed voltage vector V with passive ports'
        voltages set so I_ext=0 there. The driven ports keep their applied V."""
        n = Y_total.shape[0]
        driven = [i for i in range(n) if i not in self._tl_passive_feed_idx]
        passive = sorted(self._tl_passive_feed_idx)
        v_driven = np.array([self._feeds[i][2] for i in driven], dtype=np.complex128)
        V = np.empty(n, dtype=np.complex128)
        V[driven] = v_driven
        if passive:
            Y_pp = Y_total[np.ix_(passive, passive)]
            Y_pd = Y_total[np.ix_(passive, driven)]
            V[passive] = np.linalg.solve(Y_pp, -Y_pd @ v_driven)
        return V, driven

    def _impedance_from_y(self, Y_total):
        """Driving-point Z at each driven port, with passive (TL-only) ports
        floating (I_ext=0). Matches PyNECEngine's per-driven-port semantics
        when all drivers are excited simultaneously."""
        V, driven = self._resolve_feed_voltages(Y_total)
        I = Y_total @ V
        return [complex(V[i] / I[i]) for i in driven]

    def _compute_y_matrix(self, wavelength):
        """Multi-port short-circuit Y at the configured feeds. Builds one
        solver with the full feed list and calls pysim's compute_y_matrix,
        which since the junction-aware-y-matrix PR handles closed-loop /
        tee-junction antennas correctly (one LU + N back-subs per Y)."""
        return np.asarray(
            self._make_solver(wavelength=wavelength).compute_y_matrix(),
            dtype=np.complex128,
        )

    def _apply_loads(self, Y, omega):
        """Apply every Load branch as a Sherman-Morrison rank-1 update on
        the real-port Y — the network-level equivalent of NEC2's `ld_card`
        modifying the segment's MoM Z[k,k].

        The update has two algebraically-identical forms, dual under
        Z_L ↔ y_L = 1/Z_L:

            impedance:   Y − Z_L/(1 + Z_L·Y_kk) · outer(Y[:,k], Y[k,:])
            admittance:  Y − 1/(y_L + Y_kk)     · outer(Y[:,k], Y[k,:])

        Each Load mode has a resonance where one form divides by an
        intermediate infinity while the other stays finite, so we pick the
        form whose denominator is bounded at that mode's resonance:

          - Parallel-LC trap: Z_L→∞ at ω₀ (open circuit). Use the
            ADMITTANCE form — y_L is the tank admittance, →0 at ω₀, giving
            coefficient 1/Y_kk (the open-circuit Schur complement).
          - Series-LC: Z_L→0 at ω₀ (short circuit = unbroken wire). Use the
            IMPEDANCE form — coefficient →0, i.e. no stamp.

        This way neither path ever forms or tests for infinity.
        """
        Y = Y.copy()
        for br in self._network.branches:
            if not isinstance(br, Load):
                continue
            port = self._network.ports[br.port]
            if not isinstance(port, PortAtEdge):
                raise ValueError(
                    f"Load on virtual port {br.port!r}: a Load is a series "
                    "impedance on an antenna segment, which only PortAtEdge has"
                )
            k = self._port_to_idx[br.port]
            y_col = Y[:, k].copy()

            if br.parallel:
                # Admittance form: y_L is the parallel-LC tank admittance,
                # cleanly 0 at trap resonance (the open-circuit point).
                y_l = load_series_admittance(br, omega)
                denom = y_l + Y[k, k]
                if abs(denom) < 1e-15:
                    raise ValueError(
                        f"Load on port {br.port!r}: y_L + Y[k,k] ≈ 0 (singular)"
                    )
                Y -= np.outer(y_col, y_col) / denom
            else:
                # Impedance form: Z_L is 0 at series-LC resonance (a short
                # = unbroken wire), where the coefficient vanishes anyway.
                z_l = load_impedance(br, omega)
                if z_l == 0:
                    continue
                denom = 1.0 + z_l * Y[k, k]
                if abs(denom) < 1e-15:
                    raise ValueError(
                        f"Load on port {br.port!r}: 1 + Z_L·Y[k,k] ≈ 0 (singular)"
                    )
                Y -= (z_l / denom) * np.outer(y_col, y_col)
        return Y

    def _apply_branches(self, Y, wavelength):
        """Pad Y to include virtual ports, then stamp every network branch.

        Y comes from pysim with shape (n_real, n_real); we return a
        (n_total, n_total) augmented matrix with zeros in the virtual-port
        rows/cols (no antenna admittance) plus the branch contributions.

        Order matters: Load branches modify the antenna's real-port Y first
        (matching ld_card's effect inside the MoM), then TL branches stamp
        on the loaded Y. A TL connected to a loaded port sees the external
        side of the load.
        """
        omega = 2.0 * np.pi * C_LIGHT / wavelength
        Y = self._apply_loads(Y, omega)
        n_total = self._n_total_ports
        Y_full = np.zeros((n_total, n_total), dtype=np.complex128)
        n_real = Y.shape[0]
        Y_full[:n_real, :n_real] = Y
        for br in self._network.branches:
            if isinstance(br, TL):
                a, b = self._port_to_idx[br.a], self._port_to_idx[br.b]
                y_tl = self._tl_admittance_2x2(br.z0, br.length, wavelength)
                Y_full[np.ix_([a, b], [a, b])] += y_tl
            elif isinstance(br, Load):
                continue  # already applied to the real-port Y
            elif isinstance(br, TwoPort):
                raise NotImplementedError(
                    "TwoPort on PysimEngine: sketched but not cross-engine "
                    "validated. See issue #65 piece (B)."
                )
            else:
                raise NotImplementedError(f"branch type {type(br).__name__}")
        return Y_full

    def _resolve_network_voltages(self, Y_total):
        """Return the (n_total,) voltage vector after solving the network:
        driven ports at their applied voltages, every other port floating
        with I_ext = 0."""
        n = Y_total.shape[0]
        driven = list(self._driven_port_idx)
        # Loaded ports get V = 0 forced (correct BC for "no external source
        # on a wire-interior segment"; see _init_network for the derivation).
        floating = [
            i for i in range(n) if i not in driven and i not in self._loaded_port_idx
        ]
        v_driven = np.array(self._driven_voltages, dtype=np.complex128)
        V = np.zeros(n, dtype=np.complex128)
        V[driven] = v_driven
        # V[loaded] = 0 by zeros init.
        if floating:
            # I_ext = 0 at floating ports: Y_ff V_f = -Y_fd V_d - Y_fl V_l.
            # V_l = 0 so the last term drops; same form as before.
            Y_ff = Y_total[np.ix_(floating, floating)]
            Y_fd = Y_total[np.ix_(floating, driven)]
            V[floating] = np.linalg.solve(Y_ff, -Y_fd @ v_driven)
        return V

    def _impedance_from_network_y(self, Y_total):
        """Driven-port impedance: solve the network for V (loaded ports
        pinned at V=0, floating ports satisfying I_ext=0, driven ports at
        their applied V), then read I_driven = (Y_total @ V)_driven."""
        V = self._resolve_network_voltages(Y_total)
        I = Y_total @ V
        return [complex(V[i] / I[i]) for i in self._driven_port_idx]

    def _solved_excited(self, wavelength):
        """Build the excitation-resolved solver and run compute_impedance
        once per (wavelength, feed-voltage) tuple, caching on the engine
        instance. Lets impedance(), current_distribution(), and far_field()
        share one MoM solve when the live UI tick calls them in sequence.

        Cache lives for this engine instance only; the server constructs a
        fresh PysimEngine each tick, so nothing leaks across requests.
        """
        sim = self._make_excited_solver(wavelength=wavelength)
        v_key = tuple((complex(v).real, complex(v).imag) for *_, v in sim.feeds)
        key = (float(wavelength), v_key)
        cached = getattr(self, "_solved_cache", None)
        if cached is not None and cached[0] == key:
            return cached[1]
        z, coeffs = sim.compute_impedance()
        self._solved_cache = (key, (sim, coeffs, z))
        return sim, coeffs, z

    def impedance(self):
        wavelength = self._wavelength_for(self.builder.freq)
        if self._network is not None:
            Y = self._compute_y_matrix(wavelength)
            Y_total = self._apply_branches(Y, wavelength)
            return self._impedance_from_network_y(Y_total)
        if self._tls:
            Y = self._compute_y_matrix(wavelength)
            Y_total = self._apply_tls(Y, wavelength)
            return self._impedance_from_y(Y_total)
        _sim, _coeffs, z = self._solved_excited(wavelength)
        # Single-feed path returns a scalar; multi-feed returns an array.
        # Match PyNECEngine's list-of-Z return shape.
        z_arr = np.atleast_1d(z)
        return [complex(zi) for zi in z_arr]

    def impedance_sweep(self, freqs):
        freqs = np.asarray(freqs, dtype=float)
        if freqs.ndim != 1 or freqs.size == 0:
            raise ValueError("freqs must be a 1-D non-empty array")
        s = self._make_solver(wavelength=self._wavelength_for(freqs[0]))
        k_array = 2.0 * np.pi * freqs * 1e6 / C_LIGHT
        if self._network is not None:
            Y_swept = np.asarray(
                s.compute_y_matrix_swept(k_array), dtype=np.complex128
            )  # (n_k, n_real, n_real)
            n_driven = len(self._driven_port_idx)
            zs = np.empty((freqs.size, n_driven), dtype=np.complex128)
            for ki, freq in enumerate(freqs):
                Y_total = self._apply_branches(Y_swept[ki], self._wavelength_for(freq))
                zs[ki] = self._impedance_from_network_y(Y_total)
            return zs
        if self._tls:
            # Batched Y at every frequency, then per-k TL stamping (βL is
            # frequency-dependent) and driven-port reduction. The Y assembly
            # is amortised across frequencies via the upstream swept solve;
            # the per-k post-processing is O(n_p³) and dwarfed by the solve.
            Y_swept = np.asarray(
                s.compute_y_matrix_swept(k_array), dtype=np.complex128
            )  # (n_k, n_p, n_p)
            n_driven = sum(
                1 for i in range(Y_swept.shape[1]) if i not in self._tl_passive_feed_idx
            )
            zs = np.empty((freqs.size, n_driven), dtype=np.complex128)
            for ki, freq in enumerate(freqs):
                Y_total = self._apply_tls(Y_swept[ki], self._wavelength_for(freq))
                zs[ki] = self._impedance_from_y(Y_total)
            return zs
        zs = s.compute_impedance_swept(k_array)
        # Single-feed: (n_k,); multi-feed: (n_k, n_feeds). Normalise to
        # (n_k, n_feeds) to match PyNECEngine.
        zs = np.asarray(zs)
        if zs.ndim == 1:
            zs = zs.reshape(-1, 1)
        return zs

    def _make_excited_solver(self, *, wavelength):
        """Build a solver whose feed voltages match the actual excitation:
        for plain designs, just the build_wires() voltages; for TL or
        Network designs, the per-port voltages after the network reduction
        so basis coefficients reflect the branch-induced port voltages.
        Without this, network-spec designs (where every named feed carries
        a placeholder V=0) would solve with no excitation — every basis
        coefficient is zero and `compute_impedance`'s V/I returns NaN."""
        if self._network is not None:
            Y = self._compute_y_matrix(wavelength)
            Y_total = self._apply_branches(Y, wavelength)
            V_full = self._resolve_network_voltages(Y_total)
            feeds_resolved = [
                (w, arc, complex(V_full[i]))
                for i, (w, arc, _v) in enumerate(self._feeds)
            ]
        elif self._tls:
            Y = self._compute_y_matrix(wavelength)
            Y_total = self._apply_tls(Y, wavelength)
            V, _ = self._resolve_feed_voltages(Y_total)
            feeds_resolved = [
                (w, arc, complex(V[i])) for i, (w, arc, _v) in enumerate(self._feeds)
            ]
        else:
            return self._make_solver(wavelength=wavelength)
        return self._solver(
            wires=self._polylines,
            n_per_edge_per_wire=self._edge_segments,
            feeds=feeds_resolved,
            wavelength=wavelength,
            wire_radius=self._wire_radius,
            ground_z=self._ground_z,
            junctions=self._junctions or None,
            **self._solver_kwargs,
        )

    def current_distribution(self):
        sim, coeffs, _z = self._solved_excited(self._wavelength_for(self.builder.freq))
        knot_currents = sim.currents_at_knots(coeffs)
        out = []
        for w_idx, polyline in enumerate(self._polylines):
            knots = _polyline_knots(polyline, self._edge_segments[w_idx])
            out.append(
                WireCurrents(
                    knot_positions=np.ascontiguousarray(knots),
                    knot_currents=np.ascontiguousarray(knot_currents[w_idx]),
                )
            )
        return out

    def _segment_dipoles(self, sim, coeffs):
        """Returns (mid, dr, i_mid) — concatenated per-segment midpoints,
        edge vectors, and midpoint currents from the MoM solution."""
        knot_currents = sim.currents_at_knots(coeffs)
        mids, drs, i_mids = [], [], []
        for w_idx, polyline in enumerate(self._polylines):
            knots = _polyline_knots(polyline, self._edge_segments[w_idx])
            cur = knot_currents[w_idx]
            drs.append(knots[1:] - knots[:-1])
            mids.append(0.5 * (knots[1:] + knots[:-1]))
            i_mids.append(0.5 * (cur[1:] + cur[:-1]))
        return (
            np.concatenate(mids, axis=0),
            np.concatenate(drs, axis=0),
            np.concatenate(i_mids, axis=0),
        )

    def _evaluate_M_perp(self, mid, dr, i_mid, k, theta, phi, freq_hz):
        """|M_perp(θ,φ)|² on the (theta, phi) grids (radians).

        With ground enabled, adds the geometric-image contribution with PEC
        polarity, then layers Fresnel coefficients on the reflected wave so
        ρ_h=−1, ρ_v=+1 recovers the PEC limit exactly. Returns a real
        (n_theta, n_phi) array."""
        sin_t, cos_t = np.sin(theta), np.cos(theta)
        cos_p, sin_p = np.cos(phi), np.sin(phi)

        rx = sin_t[:, None] * cos_p[None, :]
        ry = sin_t[:, None] * sin_p[None, :]
        rz = np.broadcast_to(cos_t[:, None], rx.shape)
        rhat = np.stack([rx, ry, rz], axis=-1)

        phase = k * np.einsum("ijc,nc->ijn", rhat, mid)
        expp = np.exp(1j * phase)
        weighted = i_mid[:, None] * dr
        M = np.einsum("ijn,nc->ijc", expp, weighted)
        m_dot_r = np.sum(M * rhat, axis=-1)
        M_perp = M - m_dot_r[..., None] * rhat

        if self._ground is None:
            return np.sum(M_perp.real**2 + M_perp.imag**2, axis=-1)

        # Geometric image — horizontal current flipped, vertical preserved,
        # mirrored across z = ground_z.
        z0 = self._ground_z
        mid_img = mid.copy()
        mid_img[:, 2] = 2 * z0 - mid[:, 2]
        dr_img = dr * np.array([-1.0, -1.0, 1.0])
        weighted_img = i_mid[:, None] * dr_img
        phase_img = k * np.einsum("ijc,nc->ijn", rhat, mid_img)
        expp_img = np.exp(1j * phase_img)
        M_img = np.einsum("ijn,nc->ijc", expp_img, weighted_img)
        m_img_dot_r = np.sum(M_img * rhat, axis=-1)
        M_img_perp = M_img - m_img_dot_r[..., None] * rhat

        if self._ground[0] == "pec":
            M_perp = M_perp + M_img_perp
            return np.sum(M_perp.real**2 + M_perp.imag**2, axis=-1)

        # ("finite", eps_r, sigma): polarisation basis at each ray and
        # Fresnel reflection on the image wave.
        _, eps_r, sigma = self._ground
        s = np.sqrt(rx * rx + ry * ry)
        s_safe = np.where(s > 1e-12, s, 1.0)
        h_hat = np.stack([-ry / s_safe, rx / s_safe, np.zeros_like(rx)], axis=-1)
        v_hat = np.stack([-rx * rz / s_safe, -ry * rz / s_safe, s], axis=-1)
        M_img_h = np.sum(M_img_perp * h_hat, axis=-1)
        M_img_v = np.sum(M_img_perp * v_hat, axis=-1)

        omega = 2 * np.pi * freq_hz
        eps_c = eps_r - 1j * sigma / (omega * EPS0)
        cos_ti = rz
        sin2_ti = s * s
        Q = np.sqrt(eps_c - sin2_ti)
        rho_h = (cos_ti - Q) / (cos_ti + Q)
        rho_v = (eps_c * cos_ti - Q) / (eps_c * cos_ti + Q)

        # PEC reflection corresponds to ρ_h=−1, ρ_v=+1. The image we built
        # already has the PEC sign convention baked in, so we need the
        # Fresnel-vs-PEC ratio per polarisation:
        #     reflected_h = (−ρ_h) · M_img_h        # PEC was +1·M_img_h
        #     reflected_v = (+ρ_v) · M_img_v        # PEC was +1·M_img_v
        M_refl = (rho_v * M_img_v)[..., None] * v_hat - (rho_h * M_img_h)[
            ..., None
        ] * h_hat
        M_perp = M_perp + M_refl
        return np.sum(M_perp.real**2 + M_perp.imag**2, axis=-1)

    def far_field(self, *, n_theta=90, n_phi=360, del_theta=1, del_phi=1):
        assert 90 % n_theta == 0 and 90 == del_theta * n_theta
        assert 360 % n_phi == 0 and 360 == del_phi * n_phi

        wavelength = self._wavelength_for(self.builder.freq)
        k = 2.0 * np.pi / wavelength
        freq_hz = self.builder.freq * 1e6

        sim, coeffs, _z = self._solved_excited(wavelength)
        mid, dr, i_mid = self._segment_dipoles(sim, coeffs)

        # Cell-centred integration grid for ∫|M_perp|² dΩ. With ground the
        # antenna only radiates into the upper hemisphere, so integrate
        # there only (otherwise we'd double-count the image contribution
        # against zero-amplitude below the ground plane).
        n_th_int, n_ph_int = 90, 180
        if self._ground is not None:
            theta_int = (np.arange(n_th_int) + 0.5) * (np.pi / 2 / n_th_int)
            dtheta = np.pi / 2 / n_th_int
        else:
            theta_int = (np.arange(n_th_int) + 0.5) * (np.pi / n_th_int)
            dtheta = np.pi / n_th_int
        phi_int = np.arange(n_ph_int) * (2 * np.pi / n_ph_int)
        dphi = 2 * np.pi / n_ph_int

        mag2_int = self._evaluate_M_perp(mid, dr, i_mid, k, theta_int, phi_int, freq_hz)
        p_rad = float(np.sum(mag2_int * np.sin(theta_int)[:, None]) * dtheta * dphi)
        if p_rad <= 0:
            raise RuntimeError("computed zero radiated power")
        directivity_norm = 4 * np.pi / p_rad

        # Evaluate on the user grid (NEC convention: θ from 0 to 90−Δθ).
        theta_deg = np.linspace(0, 90 - del_theta, n_theta)
        phi_deg = np.linspace(0, 360, n_phi + 1)
        theta_user = np.deg2rad(theta_deg)
        phi_user = np.deg2rad(phi_deg)

        mag2_user = self._evaluate_M_perp(
            mid, dr, i_mid, k, theta_user, phi_user, freq_hz
        )
        D = directivity_norm * mag2_user
        # Floor before log so points where M_perp is exactly zero (poles,
        # nulls below quantisation) don't produce −inf.
        dBi = 10.0 * np.log10(np.maximum(D, 1e-30))

        rings = dBi.tolist()
        return FarField(
            rings=rings,
            max_gain=float(np.max(dBi)),
            min_gain=float(np.min(dBi)),
            thetas=theta_deg,
            phis=phi_deg,
        )
