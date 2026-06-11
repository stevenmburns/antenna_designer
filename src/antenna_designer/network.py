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
    """

    a: str  # port name
    b: str
    z0: float
    length: float


# Reserved for follow-up PR — sketched here so the discriminated-union
# pattern is established but not consumed yet by any engine.
@dataclass(frozen=True)
class Load:
    """Series R+jωL+1/(jωC) inserted at a single port (an in-line load).

    NEC2's `ld_card` type 0 is the direct backing on PyNECEngine. On
    PysimEngine the effect is the same — a series impedance modifies the
    segment's Z diagonal. Any of r/l/c may be None (omitted).
    """

    port: str
    r: float | None = None
    l: float | None = None
    c: float | None = None


@dataclass(frozen=True)
class TwoPort:
    """Lumped R+jωL+1/(jωC) between two ports. NEC2's `nt_card` is the
    direct backing on PyNECEngine. Any of r/l/c may be None (omitted)."""

    a: str
    b: str
    r: float | None = None
    l: float | None = None
    c: float | None = None


Branch = Union[TL, Load, TwoPort]


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
            refs = (br.a, br.b) if hasattr(br, "a") else (br.port,)
            for ref in refs:
                if ref not in port_names:
                    raise ValueError(f"branch {br!r} references unknown port {ref!r}")
        for src in self.sources:
            if src.port not in port_names:
                raise ValueError(f"source {src!r} references unknown port")
        if not self.sources:
            raise ValueError("Network has no driven sources")
