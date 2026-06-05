"""Translate the flat (p0, p1, n_seg, excitation) wire list used by
AntennaBuilder.build_wires() into the polyline + feed-arclength shape
that pysim's TriangularPySim consumes.

The flat list expresses connectivity implicitly: two tuples are part of
the same electrical wire when they share an endpoint (within `eps`).
pysim wants each electrical wire as one (M, 3) polyline with junction
information explicit; this module recovers that graph and chains each
connected component into a polyline.

Limitations of the current translator (acceptable for the dipole-class
designs we're cross-validating first; will be lifted as needed):
  * Exactly one excited segment per geometry.
  * No degree-3+ nodes (no tee junctions) and no closed loops — every
    connected component must be a simple path with two degree-1 ends.
"""
from __future__ import annotations

import numpy as np


def _round_point(p, eps):
    # Quantize endpoints onto an eps-spaced grid so 1e-14 floating-point
    # noise doesn't fragment what is logically a shared node.
    return tuple(round(float(c) / eps) * eps for c in p)


def flat_wires_to_polylines(tups, *, eps=1e-6):
    """Convert flat wire tuples to pysim polyline form.

    Parameters
    ----------
    tups : list of (p0, p1, n_seg, ev)
        Same shape produced by AntennaBuilder.build_wires().
    eps : float
        Endpoint-coalescence tolerance (m).

    Returns
    -------
    dict with keys
        polylines           : list of (M, 3) np.ndarray
        edge_segments       : list of list[int] — n_seg per edge per polyline
        feed_wire_index     : int
        feed_arclength      : float — distance along feed polyline to the
                              midpoint of the excited segment, matching
                              PyNEC's behaviour of feeding at segment
                              `(n_seg+1)//2`.
        feed_voltage        : complex — the excitation amplitude.
    """
    if not tups:
        raise ValueError("no wires to translate")

    # Build endpoint->node map and adjacency list.
    node_of = {}
    nodes = []  # list of np.ndarray(3,)
    edges = []  # list of (a, b, n_seg, ev, tup_index)

    def node_id(p):
        key = _round_point(p, eps)
        if key not in node_of:
            node_of[key] = len(nodes)
            nodes.append(np.asarray(p, dtype=float))
        return node_of[key]

    for i, (p0, p1, n_seg, ev) in enumerate(tups):
        a = node_id(p0)
        b = node_id(p1)
        if a == b:
            raise ValueError(f"tuple {i}: degenerate edge (p0==p1 within eps)")
        edges.append((a, b, int(n_seg), ev, i))

    adj = [[] for _ in nodes]
    for ei, (a, b, _, _, _) in enumerate(edges):
        adj[a].append((b, ei))
        adj[b].append((a, ei))

    for nid, neigh in enumerate(adj):
        if len(neigh) > 2:
            raise NotImplementedError(
                f"node {nid} has degree {len(neigh)}; tee junctions are "
                "not yet supported by the translator"
            )

    # Walk connected components into ordered polylines.
    edge_seen = [False] * len(edges)
    node_seen = [False] * len(nodes)

    polylines = []
    edge_segments = []
    edge_to_polyline = {}  # tup_index -> (polyline_index, edge_index_within)

    for start in range(len(nodes)):
        if node_seen[start]:
            continue
        if len(adj[start]) == 0:
            raise ValueError(f"node {start} is isolated")
        if len(adj[start]) == 2:
            # Interior node of a chain — wait until we hit an endpoint.
            continue

        # start has degree 1: walk the chain.
        path_nodes = [start]
        path_edges = []
        prev = None
        cur = start
        while True:
            node_seen[cur] = True
            next_step = None
            for nb, ei in adj[cur]:
                if ei == prev or edge_seen[ei]:
                    continue
                next_step = (nb, ei)
                break
            if next_step is None:
                break
            nb, ei = next_step
            edge_seen[ei] = True
            path_edges.append(ei)
            path_nodes.append(nb)
            prev = ei
            cur = nb

        polylines.append(np.stack([nodes[n] for n in path_nodes], axis=0))
        edge_segments.append([edges[ei][2] for ei in path_edges])
        for k, ei in enumerate(path_edges):
            edge_to_polyline[edges[ei][4]] = (len(polylines) - 1, k)

    # Closed loops would leave node_seen=False everywhere along them.
    if any(not s for s in node_seen):
        raise NotImplementedError(
            "closed loops in the wire graph are not supported"
        )

    # Locate the excitation and convert to (polyline_index, arclength).
    excited = [(i, ev) for i, (_, _, _, ev, _) in enumerate(edges) if ev is not None]
    if len(excited) == 0:
        raise ValueError("no excitation found in wire list")
    if len(excited) > 1:
        raise NotImplementedError(
            f"{len(excited)} excitations found; pysim engine currently "
            "supports a single feed per geometry"
        )

    tup_index, voltage = excited[0]
    feed_pl, feed_edge_idx = edge_to_polyline[tup_index]
    polyline = polylines[feed_pl]
    edge_lengths = np.linalg.norm(np.diff(polyline, axis=0), axis=1)

    # PyNEC feeds at segment `(n_seg+1)//2` which is the middle segment
    # 1-indexed; physically that's the centre of the wire. Reproduce that
    # by feeding at the midpoint of the feed edge.
    feed_arclength = float(edge_lengths[:feed_edge_idx].sum() + 0.5 * edge_lengths[feed_edge_idx])

    return {
        "polylines": polylines,
        "edge_segments": edge_segments,
        "feed_wire_index": feed_pl,
        "feed_arclength": feed_arclength,
        "feed_voltage": complex(voltage),
    }
