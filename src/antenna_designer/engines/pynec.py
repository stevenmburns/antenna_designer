import numpy as np
import PyNEC as nec

from ..engine import FarField, SimulationEngine, WireCurrents
from ..network import Driven, Load, PortAtEdge, PortVirtual, TL, TwoPort


DEFAULT_GROUND = ("finite", 10.0, 0.002)  # (kind, dielectric, conductivity)


def _seg_midpoint(p0, p1, n_seg, sub_idx):
    """Centre of segment `sub_idx` (1-based) on the wire from p0 to p1
    discretised into n_seg uniform segments. NEC2's tl_card / ex_card
    refer to sub-segments by this index."""
    p0 = np.asarray(p0, dtype=float)
    p1 = np.asarray(p1, dtype=float)
    return p0 + ((sub_idx - 0.5) / n_seg) * (p1 - p0)


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

        # Synthesise stub wires for virtual ports (extends self.tups so the
        # tag space picks up where the user-supplied wires left off). Must
        # happen before geometry_complete().
        self._network_port_loc = {}
        if self._network is not None:
            self._synthesise_virtual_stubs(geo)

        self.c.geometry_complete(0)

        if self._network is not None:
            self._emit_network_cards()
        else:
            for idx1, seg1, idx2, seg2, impedance, length in self.tls:
                self.c.tl_card(idx1, seg1, idx2, seg2, impedance, length, 0, 0, 0, 0)

        self.c.ld_card(5, 0, 0, 0, conductivity, 0.0, 0.0)
        self._apply_ground_card()

        for tag, sub_index, voltage in self.excitation_pairs:
            self.c.ex_card(0, tag, sub_index, 0, voltage.real, voltage.imag, 0, 0, 0, 0)

    def _synthesise_virtual_stubs(self, geo):
        """For each PortVirtual referenced by a TL branch, add a short stub
        wire (~λ/100, single segment) anchored above the centroid of its
        TL-connected real-port feed points. NEC2's tl_card needs both
        endpoints on real segments; the stub is the minimum geometry that
        carries the driver node without polluting radiation.

        Resolves every PortAtEdge to its (tag, sub_seg) in the same pass so
        downstream branch/source emission only consults self._network_port_loc.
        """
        net = self._network
        # Real ports: resolve via the name → (tag, mid_seg) map built above.
        for name, port in net.ports.items():
            if isinstance(port, PortAtEdge):
                if name not in self._feed_name_to_loc:
                    raise ValueError(
                        f"network port {name!r} is a PortAtEdge but no edge in "
                        f"build_wires() carries that name; named edges: "
                        f"{sorted(self._feed_name_to_loc)}"
                    )
                tag, mid_seg, _p0, _p1, _ns = self._feed_name_to_loc[name]
                self._network_port_loc[name] = (tag, mid_seg)

        # Collect, per virtual port, the set of real ports it's TL-connected to.
        virtual_neighbours = {
            name: [] for name, p in net.ports.items() if isinstance(p, PortVirtual)
        }
        for br in net.branches:
            if not isinstance(br, TL):
                continue
            ends = (br.a, br.b)
            virt_ends = [e for e in ends if isinstance(net.ports[e], PortVirtual)]
            real_ends = [e for e in ends if isinstance(net.ports[e], PortAtEdge)]
            if len(virt_ends) == 2:
                raise ValueError(
                    f"TL between two virtual ports ({br.a!r}, {br.b!r}) has no "
                    "real-segment endpoints; cannot map to NEC2 tl_card"
                )
            for v in virt_ends:
                virtual_neighbours[v].extend(real_ends)

        wavelength = 299.792458 / self.builder.design_freq
        stub_len = wavelength / 100.0
        next_tag = len(self.tups) + 1

        for name, neighbours in virtual_neighbours.items():
            if not neighbours:
                raise ValueError(
                    f"virtual port {name!r} has no TL branches; can't synthesise "
                    "a stub wire without at least one real-port neighbour"
                )
            # Centroid of neighbour feed midpoints.
            mids = np.stack(
                [
                    _seg_midpoint(
                        *self._feed_name_to_loc[r][2:5],
                        sub_idx=self._feed_name_to_loc[r][1],
                    )
                    for r in neighbours
                ]
            )
            c = mids.mean(axis=0)
            # Offset 2 stub-lengths above the highest neighbour midpoint in z,
            # so the stub doesn't collide with the antenna geometry.
            z_top = mids[:, 2].max()
            base = np.array([c[0], c[1], z_top + 2.0 * stub_len])
            tip = base + np.array([0.0, 0.0, stub_len])
            # Single-segment stub (parity "odd" is already satisfied).
            geo.wire(
                next_tag,
                1,
                base[0],
                base[1],
                base[2],
                tip[0],
                tip[1],
                tip[2],
                0.0005,
                1.0,
                1.0,
            )
            self._network_port_loc[name] = (next_tag, 1)
            next_tag += 1

    def _emit_network_cards(self):
        """Translate Network branches into tl_card calls and Network sources
        into ex_card calls. Called after geometry_complete()."""
        net = self._network
        for br in net.branches:
            if isinstance(br, TL):
                tag_a, seg_a = self._network_port_loc[br.a]
                tag_b, seg_b = self._network_port_loc[br.b]
                self.c.tl_card(tag_a, seg_a, tag_b, seg_b, br.z0, br.length, 0, 0, 0, 0)
            elif isinstance(br, (Load, TwoPort)):
                raise NotImplementedError(
                    f"{type(br).__name__} on PyNECEngine is a follow-up piece of "
                    "issue #65 (ld_card / nt_card translation); use PysimEngine "
                    "for now"
                )
            else:
                raise NotImplementedError(f"branch type {type(br).__name__}")
        for src in net.sources:
            if not isinstance(src, Driven):
                raise NotImplementedError(f"unknown source type: {src!r}")
            tag, seg = self._network_port_loc[src.port]
            v = complex(src.voltage)
            self.excitation_pairs.append((tag, seg, v))

    def _apply_ground_card(self):
        g = self.ground
        if g is None or g == "free":
            return  # no gn_card -> free space
        if g == "pec":
            self.c.gn_card(1, 0, 0, 0, 0, 0, 0, 0)
            return
        if isinstance(g, tuple) and len(g) == 3 and g[0] == "finite":
            _, eps_r, sigma = g
            self.c.gn_card(0, 0, eps_r, sigma, 0, 0, 0, 0)
            return
        raise ValueError(f"unrecognised ground spec: {g!r}")

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
