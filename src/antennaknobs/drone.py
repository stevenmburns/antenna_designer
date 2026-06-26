"""A 3D 'turtle' for describing antenna geometry as a flight path.

Most designs build their wire list by computing absolute corner coordinates
(see e.g. ``designs/loops/delta_loop.py``). For path-shaped antennas — loops,
inverted-vee arms, rhombics, zig-zags — it can be far clearer to *describe the
walk*: fly a leg, turn, fly the next leg, paying out wire as you go. The
``Drone`` is the 3D analog of turtle graphics' pen: it carries a pose
(position + orientation) and compiles to the exact same edge list every
``build_wires`` returns::

    ((x0, y0, z0), (x1, y1, z1), nsegs, excitation), ...

so it is purely an alternate *authoring* style — nothing downstream changes.

Verbs
-----
- ``pay_out()`` / ``cut()``  — start/stop letting out (structural) wire,
  the pen-down / pen-up of turtle graphics.
- ``feed(excitation)``       — like ``pay_out`` but the wire laid down is the
  driven segment (carries the source excitation).
- ``forward(dist)``          — fly along the nose; lays an edge when the pen
  is down.  ``jump(dist)`` moves without wire.
- ``yaw`` / ``pitch`` / ``roll`` — turn in the body frame (degrees), so they
  are relative to the current heading, exactly like a turtle's ``left``/
  ``right`` generalized to 3D.
- ``face(heading, up)``      — point the nose along ``heading`` with the given
  ``up`` (useful to start a planar figure: pick an in-plane heading and a
  plane-normal ``up``, after which ``yaw`` keeps you in that plane).
- ``close()``                — fly straight back to where the current
  pen-down stroke began (turtle ``home``), e.g. to lay the closing edge of a
  loop or its feed gap.

Orientation is a single homogeneous :class:`~antennaknobs.transform.Transform`
(a rotation matrix), so it composes with the existing ``Transform``/
``TransformStack`` machinery and has no gimbal-lock failure in the
representation itself.
"""

import math

import numpy as np

from .transform import Transform


class Drone:
    """A pen-carrying drone. Body convention: local +x is 'forward' (the
    nose), local +z is 'up' (the yaw axis), local +y completes a
    right-handed frame ('left')."""

    def __init__(self, position=(0.0, 0.0, 0.0), *, nominal_nsegs=21, ref=1.0):
        # ``ref`` is the reference length a wire of which gets ``nominal_nsegs``
        # segments (usually a quarter-wavelength), matching AntennaBuilder.
        self.pose = Transform.translate(*position)
        self.nominal_nsegs = nominal_nsegs
        self.ref = ref
        self._pen = None  # None = up; ("ex", value) when down
        self._origin = None  # world position where the current stroke began
        self.edges = []

    # -- state ----------------------------------------------------------
    @property
    def position(self):
        return self.pose.hit((0.0, 0.0, 0.0))

    @property
    def heading(self):
        """World-space unit vector the nose (local +x) points along."""
        p0 = np.array(self.position)
        p1 = np.array(self.pose.hit((1.0, 0.0, 0.0)))
        return tuple(p1 - p0)

    # -- pen ------------------------------------------------------------
    def pay_out(self):
        """Lower the pen: subsequent forward() lays structural wire."""
        self._set_pen(("ex", None))
        return self

    def feed(self, excitation=1 + 0j):
        """Lower the pen for the driven segment: subsequent forward() lays
        wire carrying ``excitation`` (the source)."""
        self._set_pen(("ex", excitation))
        return self

    def cut(self):
        """Raise the pen: subsequent forward() moves without laying wire."""
        self._pen = None
        self._origin = None
        return self

    def _set_pen(self, pen):
        if self._pen is None:  # transitioning up -> down starts a new stroke
            self._origin = self.position
        self._pen = pen

    # -- segmentation ---------------------------------------------------
    def _seg(self, length):
        n = max(3, round(self.nominal_nsegs * abs(length) / self.ref))
        return n if n % 2 == 1 else n + 1

    def _emit(self, p0, p1, nsegs):
        if self._pen is not None:
            _, ex = self._pen
            self.edges.append((p0, p1, nsegs or self._seg(math.dist(p0, p1)), ex))

    # -- moves ----------------------------------------------------------
    def forward(self, dist, nsegs=None):
        """Fly ``dist`` along the nose, laying an edge if the pen is down.
        ``nsegs`` overrides the auto (length-proportional) segment count."""
        p0 = self.position
        self.pose = self.pose.postmult(Transform.translate(dist, 0, 0))
        self._emit(p0, self.position, nsegs)
        return self

    def jump(self, dist):
        """Fly ``dist`` along the nose without laying wire (pen ignored)."""
        self.pose = self.pose.postmult(Transform.translate(dist, 0, 0))
        return self

    def move_to(self, position):
        """Relocate to ``position`` keeping orientation; lays no wire."""
        A = self.pose.A.copy()
        A[0:3, 3] = position
        self.pose = Transform(A)
        return self

    def close(self, nsegs=None):
        """Fly straight to where the current pen-down stroke began, laying
        the closing edge with the current pen. No-op if the pen is up or we
        are already home."""
        if self._origin is None:
            return self
        p0 = self.position
        if math.dist(p0, self._origin) > 1e-12:
            self._emit(p0, tuple(self._origin), nsegs)
        self.move_to(self._origin)
        return self

    # -- turns (body-relative) -----------------------------------------
    def yaw(self, deg):
        """Turn left/right about the local up axis (degrees)."""
        self.pose = self.pose.postmult(Transform.rotZ(deg))
        return self

    def pitch(self, deg):
        """Nose up/down about the local left axis (degrees)."""
        self.pose = self.pose.postmult(Transform.rotY(deg))
        return self

    def roll(self, deg):
        """Bank about the nose axis (degrees)."""
        self.pose = self.pose.postmult(Transform.rotX(deg))
        return self

    def face(self, heading, up=(0.0, 0.0, 1.0)):
        """Point the nose along ``heading`` with the given ``up``, keeping the
        current position. ``up`` is orthogonalized against ``heading``; the
        two must not be parallel. Handy to start a planar figure: face an
        in-plane heading with the plane normal as ``up``, then ``yaw`` turns
        stay in the plane."""
        x = np.array(heading, float)
        nx = np.linalg.norm(x)
        if nx < 1e-12:
            raise ValueError("heading must be non-zero")
        x /= nx
        u = np.array(up, float)
        z = u - x * np.dot(u, x)  # component of up perpendicular to heading
        nz = np.linalg.norm(z)
        if nz < 1e-12:
            raise ValueError("up must not be parallel to heading")
        z /= nz
        y = np.cross(z, x)  # right-handed: x (nose) × y (left) = z (up)
        A = np.eye(4)
        A[0:3, 0] = x
        A[0:3, 1] = y
        A[0:3, 2] = z
        A[0:3, 3] = self.position
        self.pose = Transform(A)
        return self

    def wires(self):
        """The accumulated edge list, in the same shape build_wires returns."""
        return list(self.edges)
