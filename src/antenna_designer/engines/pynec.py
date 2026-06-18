import numpy as np
import PyNEC as nec

from ..engine import FarField, SimulationEngine, WireCurrents
from ..network import (
    DiffTL,
    Driven,
    Load,
    PortAtEdge,
    PortVirtual,
    TL,
)
from ..network_reduce import C_LIGHT, NetworkReducer

WIRE_RADIUS = 0.0005
COPPER_CONDUCTIVITY = 5.8e7


DEFAULT_GROUND = ("finite", 10.0, 0.002)  # (kind, dielectric, conductivity)


class PyNECEngine(SimulationEngine):
    supports_far_field = True
    # NEC's source placement uses (n_seg+1)//2, which lands on the centre
    # segment for odd n_seg. Even counts get bumped up so the feed sits
    # at a true wire midpoint instead of off-centre.
    segment_parity = "odd"

    def __init__(self, builder, *, ground=DEFAULT_GROUND):
        """
        ground:
          None or "free"               — no gn_card (free space)
          "pec"                        — perfectly conducting ground
          ("finite", eps_r, sigma)     — Sommerfeld finite ground (default,
                                         matches the historical hard-coded
                                         eps_r=10, sigma=0.002)
        """
        super().__init__(builder)
        self.tups = self._coerce_wire_tuples(builder.build_wires())
        self._network = builder.build_network()
        # build_tls() is only consulted when there's no Network spec; with a
        # Network, the engine drives ex_card/tl_card calls off the spec instead.
        self.tls = [] if self._network is not None else builder.build_tls()
        self.ground = ground
        self.excitation_pairs = None
        # Loads alone are handled natively (ld_card) and accurately by NEC, so
        # only divert to the multiport-Y + NetworkReducer path for what NEC
        # *can't* do natively: transmission lines (TL/DiffTL) and virtual
        # drivers. Those skip the baked NEC context entirely — impedance uses
        # per-port solves and far-field/current build an excitation-resolved
        # context on demand. Load-only and plain designs keep the native path.
        self._use_reducer = self._network is not None and self._network_uses_reducer()
        if self._use_reducer:
            self._init_network()
        else:
            self._build_geometry()

    def __del__(self):
        # Release the nec_context handle if construction got that far.
        c = getattr(self, "c", None)
        if c is not None:
            del self.c

    def _build_geometry(self):
        conductivity = 5.8e7  # Copper

        self.c = nec.nec_context()
        geo = self.c.get_geometry()

        # Walk build_wires(): emit `geo.wire` cards, collect ex_card pairs from
        # any tuple with a non-None `ev`, and remember (tag, mid_seg) for every
        # named edge so the network path can resolve PortAtEdge later.
        self.excitation_pairs = []
        self._feed_name_to_loc = {}
        for idx, t in enumerate(self.tups, start=1):
            p0, p1, n_seg, ev = t[0], t[1], t[2], t[3]
            name = t[4] if len(t) >= 5 else None
            geo.wire(
                idx, n_seg, p0[0], p0[1], p0[2], p1[0], p1[1], p1[2], 0.0005, 1.0, 1.0
            )
            mid_seg = (n_seg + 1) // 2
            if name is not None:
                self._feed_name_to_loc[name] = (idx, mid_seg, p0, p1, n_seg)
            # In the network path the spec drives ex_card emission; ignore ev's
            # (they're typically placeholders to keep a segment marked at the
            # feed edge for the geometry translator).
            if ev is not None and self._network is None:
                self.excitation_pairs.append((idx, mid_seg, ev))

        self._network_port_loc = {}

        self.c.geometry_complete(0)

        if self._network is not None:
            # Only Load-only networks reach here; TL/DiffTL/virtual-driver
            # networks take the NetworkReducer path and never build this
            # context.
            self._resolve_network_ports()
            self._emit_network_cards()
        else:
            for idx1, seg1, idx2, seg2, impedance, length in self.tls:
                self.c.tl_card(idx1, seg1, idx2, seg2, impedance, length, 0, 0, 0, 0)

        self.c.ld_card(5, 0, 0, 0, conductivity, 0.0, 0.0)
        self._apply_ground_card()

        for tag, sub_index, voltage in self.excitation_pairs:
            self.c.ex_card(0, tag, sub_index, 0, voltage.real, voltage.imag, 0, 0, 0, 0)

    def _resolve_network_ports(self):
        """Resolve every PortAtEdge to its (tag, sub_seg) for native ld_card /
        ex_card emission. Only reached for Load-only networks; TL/DiffTL and
        virtual-driver networks take the NetworkReducer path instead."""
        for name, port in self._network.ports.items():
            if isinstance(port, PortAtEdge):
                if name not in self._feed_name_to_loc:
                    raise ValueError(
                        f"network port {name!r} is a PortAtEdge but no edge in "
                        f"build_wires() carries that name; named edges: "
                        f"{sorted(self._feed_name_to_loc)}"
                    )
                tag, mid_seg, _p0, _p1, _ns = self._feed_name_to_loc[name]
                self._network_port_loc[name] = (tag, mid_seg)

    def _emit_network_cards(self):
        """Emit a Load-only network as native NEC2 ld_cards + ex_cards. Called
        after geometry_complete(). (TL/DiffTL/virtual-driver networks never
        reach here — they go through the multiport-Y NetworkReducer path.)

        Load branches become ld_cards (type 0 = series RLC, type 1 = parallel
        RLC) on a single segment; a zero R/L/C means that element is absent,
        matching the Load dataclass's optional fields.
        """
        net = self._network
        for br in net.branches:
            if not isinstance(br, Load):
                raise NotImplementedError(
                    f"{type(br).__name__} reached PyNEC's native network path; "
                    "only Load is handled natively (TL/DiffTL/virtual-driver "
                    "networks use the NetworkReducer path)"
                )
            port = net.ports[br.port]
            if not isinstance(port, PortAtEdge):
                raise ValueError(
                    f"Load on virtual port {br.port!r}: a Load is a series "
                    "impedance on an antenna segment, which only PortAtEdge has"
                )
            tag, seg = self._network_port_loc[br.port]
            r = float(br.r) if br.r is not None else 0.0
            l = float(br.l) if br.l is not None else 0.0
            c = float(br.c) if br.c is not None else 0.0
            if r == 0.0 and l == 0.0 and c == 0.0:
                continue
            ldtyp = 1 if br.parallel else 0
            self.c.ld_card(ldtyp, tag, seg, seg, r, l, c)
        for src in net.sources:
            if not isinstance(src, Driven):
                raise NotImplementedError(f"unknown source type: {src!r}")
            tag, seg = self._network_port_loc[src.port]
            v = complex(src.voltage)
            self.excitation_pairs.append((tag, seg, v))

    def _apply_ground_card(self, c=None):
        c = c if c is not None else self.c
        g = self.ground
        if g is None or g == "free":
            return  # no gn_card -> free space
        if g == "pec":
            c.gn_card(1, 0, 0, 0, 0, 0, 0, 0)
            return
        if isinstance(g, tuple) and len(g) == 3 and g[0] == "finite":
            _, eps_r, sigma = g
            c.gn_card(0, 0, eps_r, sigma, 0, 0, 0, 0)
            return
        raise ValueError(f"unrecognised ground spec: {g!r}")

    # ----- network-spec path: multiport Y + shared NetworkReducer -----
    #
    # NEC2's tl_card can't represent a virtual driver behind a line (it needs
    # both endpoints on real segments, and a synthesised dummy stub injects a
    # huge parasitic reactance that the line fails to transform away). So for
    # `build_network()` designs we don't emit tl_cards at all: we extract the
    # antenna's multiport short-circuit Y at the real ports and hand it to the
    # engine-agnostic NetworkReducer (the EZNEC approach — transmission lines
    # as a circuit post-process on the field solution). Shared with pysim.

    def _network_uses_reducer(self):
        """True iff the network needs the Y-matrix reduction path — i.e. it
        has a transmission line (TL/DiffTL) or a virtual driver. Load-only
        networks are handled natively by NEC's ld_card."""
        net = self._network
        if any(isinstance(b, (TL, DiffTL)) for b in net.branches):
            return True
        return any(isinstance(p, PortVirtual) for p in net.ports.values())

    def _init_network(self):
        """Build the port-index map (real PortAtEdge ports first, virtual
        ports after) and the NetworkReducer. Validates that every PortAtEdge
        names an edge in build_wires()."""
        net = self._network
        if any(isinstance(b, DiffTL) for b in net.branches):
            # The multiport-Y + NetworkReducer path *can* express a DiffTL
            # (NEC2's tl_card cannot), but DiffTL on PyNEC isn't cross-
            # validated yet — keep it PysimEngine-only for now.
            raise NotImplementedError(
                "DiffTL (4-terminal differential transmission line) on "
                "PyNECEngine is not enabled; use PysimEngine for differential "
                "lines."
            )
        named = {t[4] for t in self.tups if len(t) >= 5 and t[4] is not None}
        self._real_port_names = [
            n for n, p in net.ports.items() if isinstance(p, PortAtEdge)
        ]
        for name in self._real_port_names:
            if name not in named:
                raise ValueError(
                    f"network port {name!r} is a PortAtEdge but no edge in "
                    f"build_wires() carries that name; named edges: {sorted(named)}"
                )
        port_to_idx = {n: i for i, n in enumerate(self._real_port_names)}
        next_idx = len(self._real_port_names)
        for name, port in net.ports.items():
            if isinstance(port, PortVirtual):
                port_to_idx[name] = next_idx
                next_idx += 1
        self._reducer = NetworkReducer(net, port_to_idx, next_idx)

    def _make_real_context(self):
        """A fresh nec_context with only the real build_wires() geometry, wire
        conductivity, and ground — no virtual stubs, no tl_cards. Returns
        (context, {edge_name: (tag, mid_seg)})."""
        c = nec.nec_context()
        geo = c.get_geometry()
        loc = {}
        for idx, t in enumerate(self.tups, start=1):
            p0, p1, n_seg = t[0], t[1], t[2]
            name = t[4] if len(t) >= 5 else None
            geo.wire(
                idx,
                n_seg,
                p0[0],
                p0[1],
                p0[2],
                p1[0],
                p1[1],
                p1[2],
                WIRE_RADIUS,
                1.0,
                1.0,
            )
            if name is not None:
                loc[name] = (idx, (n_seg + 1) // 2)
        c.geometry_complete(0)
        c.ld_card(5, 0, 0, 0, COPPER_CONDUCTIVITY, 0.0, 0.0)
        self._apply_ground_card(c)
        return c, loc

    @staticmethod
    def _port_current(sc, tag, seg):
        """Complex current in sub-segment `seg` (1-based) of wire `tag`."""
        matches = [k for k, t in enumerate(sc.get_current_segment_tag()) if t == tag]
        return sc.get_current()[matches[seg - 1]]

    def _compute_y_matrix(self, wavelength):
        """Multiport short-circuit Y at the real ports, via one NEC solve per
        port: drive that port's gap at 1 V (the other named ports stay
        continuous = shorted) and read the resulting current at every port —
        column j of Y. The geometry's interaction matrix is refactored once
        per solve; small antennas make the N solves cheap."""
        freq = C_LIGHT / wavelength / 1e6
        names = self._real_port_names
        n = len(names)
        Y = np.zeros((n, n), dtype=np.complex128)
        for j, drv in enumerate(names):
            c, loc = self._make_real_context()
            tag, seg = loc[drv]
            c.ex_card(0, tag, seg, 0, 1.0, 0.0, 0, 0, 0, 0)
            c.fr_card(0, 1, freq, 0)
            c.xq_card(0)
            sc = c.get_structure_currents(0)
            for i, name in enumerate(names):
                Y[i, j] = self._port_current(sc, *loc[name])
            del c
        return Y

    def _excited_real_context(self, wavelength):
        """Fresh real-geometry context driven at the network-resolved real-
        port voltages (each real port a delta-gap at its resolved V), so
        far-field / current readouts reflect the network. fr_card is left to
        the caller."""
        Y = self._compute_y_matrix(wavelength)
        V = self._reducer.resolve_voltages(self._reducer.apply_branches(Y, wavelength))
        c, loc = self._make_real_context()
        for i, name in enumerate(self._real_port_names):
            tag, seg = loc[name]
            v = complex(V[i])
            c.ex_card(0, tag, seg, 0, v.real, v.imag, 0, 0, 0, 0)
        return c

    def _set_freq_and_execute(self):
        self.c.fr_card(0, 1, self.builder.freq, 0)
        self.c.xq_card(0)

    def _impedances_at(self, freq_index, sum_currents=False):
        sc = self.c.get_structure_currents(freq_index)

        indices = []
        for tag, tag_index, voltage in self.excitation_pairs:
            matches = [
                (i, t) for (i, t) in enumerate(sc.get_current_segment_tag()) if t == tag
            ]
            index = matches[tag_index - 1][0]
            indices.append((index, voltage))

        currents = sc.get_current()
        zs = [voltage / currents[idx] for idx, voltage in indices]

        if sum_currents:
            zs = [1 / sum(1 / z for z in zs)]

        return zs

    def impedance(self, sum_currents=False):
        if self._use_reducer:
            wl = C_LIGHT / (self.builder.freq * 1e6)
            return self._reducer.driven_impedance(self._compute_y_matrix(wl), wl)
        self._set_freq_and_execute()
        return self._impedances_at(0, sum_currents=sum_currents)

    def impedance_sweep(self, freqs, sum_currents=False):
        freqs = np.asarray(freqs, dtype=float)
        if freqs.ndim != 1 or freqs.size == 0:
            raise ValueError("freqs must be a 1-D non-empty array")
        if self._use_reducer:
            zs = np.empty((freqs.size, self._reducer.n_driven), dtype=np.complex128)
            for k, f in enumerate(freqs):
                wl = C_LIGHT / (float(f) * 1e6)
                zs[k] = self._reducer.driven_impedance(self._compute_y_matrix(wl), wl)
            return zs
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
        return np.array(
            [
                self._impedances_at(i, sum_currents=sum_currents)
                for i in range(freqs.size)
            ]
        )

    def current_distribution(self):
        """Per-tuple knot positions + complex currents. Each build_wires()
        tuple becomes one wire entry with n_seg+1 knot positions; per-knot
        currents are the average of the two adjacent NEC segment-centre
        currents, with boundary knots zeroed (open-wire BC). The pysim web
        backend uses the same averaging convention."""
        if self._use_reducer:
            self.c = self._excited_real_context(C_LIGHT / (self.builder.freq * 1e6))
        self._set_freq_and_execute()
        sc = self.c.get_structure_currents(0)
        all_tags = list(sc.get_current_segment_tag())
        all_cur = sc.get_current()

        out = []
        for tag_idx, t in enumerate(self.tups, start=1):
            p0, p1, n_seg = t[0], t[1], t[2]
            seg_idxs = [i for i, t in enumerate(all_tags) if t == tag_idx]
            cur_per_seg = np.array([all_cur[i] for i in seg_idxs], dtype=np.complex128)
            knots = np.linspace(p0, p1, n_seg + 1)
            knot_cur = np.zeros(n_seg + 1, dtype=np.complex128)
            if n_seg >= 2:
                knot_cur[1:-1] = 0.5 * (cur_per_seg[:-1] + cur_per_seg[1:])
            elif n_seg == 1:
                # 1-segment wire: no interior knot, leave boundaries at 0.
                pass
            out.append(
                WireCurrents(
                    knot_positions=knots,
                    knot_currents=knot_cur,
                )
            )
        return out

    def far_field(self, *, n_theta=90, n_phi=360, del_theta=1, del_phi=1):
        if self._use_reducer:
            self.c = self._excited_real_context(C_LIGHT / (self.builder.freq * 1e6))
        self._set_freq_and_execute()
        return self._collect_pattern(
            n_theta=n_theta, n_phi=n_phi, del_theta=del_theta, del_phi=del_phi
        )

    def _collect_pattern(self, *, n_theta, n_phi, del_theta, del_phi):
        assert 90 % n_theta == 0 and 90 == del_theta * n_theta
        assert 360 % n_phi == 0 and 360 == del_phi * n_phi

        self.c.rp_card(
            0, n_theta, n_phi + 1, 0, 5, 0, 0, 0, 0, del_theta, del_phi, 0, 0
        )

        thetas = np.linspace(0, 90 - del_theta, n_theta)
        phis = np.linspace(0, 360, n_phi + 1)

        rings = [
            [
                self.c.get_gain(0, theta_index, phi_index)
                for phi_index, _ in enumerate(phis)
            ]
            for theta_index, _ in enumerate(thetas)
        ]

        return FarField(
            rings=rings,
            max_gain=self.c.get_gain_max(0),
            min_gain=self.c.get_gain_min(0),
            thetas=thetas,
            phis=phis,
        )
