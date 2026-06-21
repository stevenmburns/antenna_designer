"""Port-based network spec for transmission lines and lumped elements.

A `Network` describes how the antenna's feed-edges (named real ports) and
purely-logical nodes (virtual ports) hook together via two-port branches.
Engines consume the network as a post-processing layer on top of the
multi-port antenna Y matrix: each branch contributes a frequency-dependent
2×2 admittance stamp into Y at the right port pair, then the system is
reduced to the driven-port impedance via nodal analysis with passive ports
floating (I_ext=0).

Compared with the legacy `build_tls()` API:
  - No dummy stub wire is required for the driver — virtual ports exist
    only in the network reduction, not in the geometry.
  - Branches refer to ports by name; no manual segment-index counting.
  - Same shape covers transmission lines (`TL`) and (planned) lumped
    elements (`Load`, `TwoPort` — coming in a follow-up).

For PyNECEngine, this spec gets translated back into the NEC2-shaped
`tl_card` / `ld_card` / `nt_card` calls, with virtual ports synthesised
as tiny stub wires at sensible locations.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Union


@dataclass(frozen=True)
class PortAtEdge:
    """Real port at a named edge from `build_wires()`. The named edge's
    feed segment becomes the port location; pysim places a delta-gap at
    the edge midpoint to read/inject current there."""

    name: str


@dataclass(frozen=True)
class PortVirtual:
    """Logical port with no geometry. Exists only as a row/column in the
    network Y matrix; doesn't radiate, doesn't have a basis function.
    Used for driver feeds that branch out via TLs to real ports."""

    name: str


Port = Union[PortAtEdge, PortVirtual]


@dataclass(frozen=True)
class TL:
    """Lossless transmission line between two ports.

    z0:     characteristic impedance in Ω
    length: physical length in meters

    The electrical length βl is computed at solve time from the antenna's
    operating wavelength. Both endpoints can be either real or virtual.

    transposed: crossed ("half-twist") line — inverts port B's polarity,
    flipping the sign of the off-diagonal transfer terms. This is the
    phase reversal a transposed-feeder array (LPDA, ZL-Special) needs.
    Prefer it over a negative z0, which would wrongly negate the diagonal
    self terms too.
    """

    a: str  # port name
    b: str
    z0: float
    length: float
    transposed: bool = False


@dataclass(frozen=True)
class DiffTL:
    """Differential (twisted-pair) lossless transmission line.

    Where `TL` couples two single-ended ports (each pinned to one antenna
    segment), `DiffTL` is a genuine 2-conductor line whose two ports are
    each a *differential pair* of independent real ports:
        port A = (a_pos, a_neg),  port B = (b_pos, b_neg).
    This is the 4-terminal element NEC2's `tl_card` cannot express, so it
    is pysim-only (PyNECEngine raises on it).

    transposed=True applies the physical half-twist of the pair (swap the
    b-port terminals), flipping the A<->B coupling sign — the phase
    inversion that keeps a Sterba curtain's sections co-phased.

    z0 is the differential-mode impedance. z0_cm, if given, adds the
    common-mode line (the two conductors in parallel) — the through-current
    a real wire pair carries that a pure-differential line omits. Leave it
    None for the pure-differential element.

    Inherits the half-wave singularity of the ideal line; pick `length`
    slightly off kλ/2.
    """

    a_pos: str
    a_neg: str
    b_pos: str
    b_neg: str
    z0: float
    length: float
    transposed: bool = False
    z0_cm: float | None = None


# Reserved for follow-up PR — sketched here so the discriminated-union
# pattern is established but not consumed yet by any engine.
@dataclass(frozen=True)
class Load:
    """R/L/C load inserted in series with a single segment's current path.

    `parallel=False` (default): series R + jωL + 1/(jωC). The whole expression
    is a single series impedance Z_load that adds to the segment's MoM Z[k,k].
    NEC2 calls this `ld_card` type 0.

    `parallel=True`: parallel R || jωL || 1/(jωC). The branch's effective
    series impedance Z_load = 1 / (1/R + 1/(jωL) + jωC). At ω₀ = 1/√(LC)
    the parallel-LC has Y → 0 → Z_load → ∞, which is exactly the trap idiom:
    the segment's current is interrupted at the trap's resonant frequency.
    NEC2 calls this `ld_card` type 1.

    Either way the effect is "lumped impedance in series with the segment":
    Load modifies a single segment's self-Z (rank-1 update on the MoM
    matrix). The classic dual-band trap dipole uses Load(parallel=True) at
    a single segment in each arm — see designs/multiband/trap_dipole.py.
    """

    port: str
    r: float | None = None
    l: float | None = None
    c: float | None = None
    parallel: bool = False


@dataclass(frozen=True)
class TwoPort:
    """Lumped R+jωL+1/(jωC) between two ports. NEC2's `nt_card` is the
    direct backing on PyNECEngine. Any of r/l/c may be None (omitted).

    NOTE: Implementation is sketched on both engines but has not been
    cross-validated. Use at your own risk; prefer `Load(parallel=True)`
    for the trap-dipole idiom (series Z on a wire-interior segment, which
    is what `ld_card` was designed for). See issue #65 piece (B) for the
    open work — cross-engine sanity tests revealed disagreement when the
    branch is conducting (R small), suggesting either a BC issue in the
    pysim reduction or a `nt_card` semantics mismatch we haven't
    untangled yet."""

    a: str
    b: str
    r: float | None = None
    l: float | None = None
    c: float | None = None


Branch = Union[TL, DiffTL, Load, TwoPort]


def _branch_port_refs(br):
    """Port names a branch references, regardless of branch type."""
    if isinstance(br, DiffTL):
        return (br.a_pos, br.a_neg, br.b_pos, br.b_neg)
    if hasattr(br, "a"):  # TL, TwoPort
        return (br.a, br.b)
    return (br.port,)  # Load


def _series_rlc_impedance(r, l, c, omega):
    """Series R + jωL + 1/(jωC). Any of r/l/c may be None (omitted term)."""
    z = 0.0 + 0.0j
    if r is not None:
        z += r
    if l is not None:
        z += 1j * omega * l
    if c is not None:
        z += 1.0 / (1j * omega * c)
    return z


def _parallel_rlc_admittance(r, l, c, omega):
    """Parallel 1/R + 1/(jωL) + jωC. Any of r/l/c may be None (omitted term).
    Trap dipoles use this: parallel-LC has Y → 0 at ω₀ = 1/√(LC), so the
    branch opens at the trap's resonant frequency."""
    y = 0.0 + 0.0j
    if r is not None:
        y += 1.0 / r
    if l is not None:
        y += 1.0 / (1j * omega * l)
    if c is not None:
        y += 1j * omega * c
    return y


def load_series_admittance(br, omega):
    """Series-branch admittance y_load = 1/Z_load of a Load branch at ω.

    This is the natural quantity for the Sherman-Morrison port-Y stamp
    (see network_reduce.NetworkReducer.apply_loads): the stamp coefficient is
    1/(y_load + Y_kk), which stays finite exactly where Z_load blows up.

    Parallel mode: y_load IS the parallel-LC tank admittance,
        y = 1/R + 1/(jωL) + jωC,
    which goes cleanly to 0 at ω₀ = 1/√(LC) — the trap-resonance open
    circuit. No singularity: the "infinite impedance" only ever appeared
    when we formed Z_load = 1/y and then took 1/Z_load again.

    Series mode: y_load = 1/(R + jωL + 1/(jωC)); returns complex inf when
    the series impedance is exactly 0 (series-LC short circuit), which the
    caller treats as "no series element" (the wire is unbroken)."""
    if br.parallel:
        return _parallel_rlc_admittance(br.r, br.l, br.c, omega)
    z = _series_rlc_impedance(br.r, br.l, br.c, omega)
    if z == 0:
        return complex(float("inf"), 0.0)
    return 1.0 / z


def load_impedance(br, omega):
    """Effective series impedance of a Load branch at angular ω.
    Series mode: Z = R + jωL + 1/(jωC).
    Parallel mode: Z = 1 / (1/R + 1/(jωL) + jωC) — equals the parallel-LC
    tank impedance, diverging at ω₀ = 1/√(LC) (the trap idiom).

    Returns complex inf at parallel-LC resonance rather than raising —
    Z→∞ is the physically-intended open circuit of a trap. Consumers that
    stamp the load into a port-Y matrix should prefer
    `load_series_admittance`, which avoids forming this infinity at all."""
    if br.parallel:
        y = _parallel_rlc_admittance(br.r, br.l, br.c, omega)
        if y == 0:
            return complex(float("inf"), 0.0)
        return 1.0 / y
    return _series_rlc_impedance(br.r, br.l, br.c, omega)


@dataclass(frozen=True)
class Driven:
    """Voltage source applied at a port. Multiple Driven entries are
    allowed — they're all driven simultaneously with their specified
    voltages (matching the multi-feed Y semantics)."""

    port: str
    voltage: complex = 1 + 0j


Source = Driven


@dataclass
class Network:
    """Complete network spec returned by `build_network()`.

    ports:    dict mapping name → Port (real or virtual)
    branches: list of Branch (TL / Load / TwoPort)
    sources:  list of Source (currently just Driven)

    The engine's job: assemble the antenna Y matrix at the real ports,
    pad to include virtual ports, stamp every branch, then reduce to the
    driven-port impedances.
    """

    ports: dict[str, Port]
    branches: list[Branch] = field(default_factory=list)
    sources: list[Source] = field(default_factory=list)

    def __post_init__(self):
        for name, port in self.ports.items():
            if port.name != name:
                raise ValueError(
                    f"port dict key {name!r} doesn't match Port.name {port.name!r}"
                )
        port_names = set(self.ports)
        for br in self.branches:
            for ref in _branch_port_refs(br):
                if ref not in port_names:
                    raise ValueError(f"branch {br!r} references unknown port {ref!r}")
        for src in self.sources:
            if src.port not in port_names:
                raise ValueError(f"source {src!r} references unknown port")
        if not self.sources:
            raise ValueError("Network has no driven sources")
