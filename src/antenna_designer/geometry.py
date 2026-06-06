"""Translate the flat (p0, p1, n_seg, excitation) wire list used by
AntennaBuilder.build_wires() into the polyline + feed-arclength shape
that pysim's TriangularPySim consumes.

The flat list expresses connectivity implicitly: two tuples are part of
the same electrical wire when they share an endpoint (within `eps`).
pysim wants each electrical wire as one (M, 3) polyline with junction
information (KCL at shared nodes) explicit; this module recovers the
graph, decomposes it into maximal chains between junction/endpoint
nodes, and emits the junction list.

Supported topologies:
  * Each connected component must contain at least one degree-1 endpoint
    OR at least one junction (degree >= 3); pure cycles aren't handled.
  * Any number of junctions of any degree (tees, X's, hentenna-style
    multi-junction, fandipole-style multi-spoke feeds).
  * Exactly one excited segment per geometry.
"""
from __future__ import annotations

import numpy as np


def _round_point(p, eps):
    # Quantize endpoints onto an eps-spaced grid so 1e-14 floating-point
    # noise doesn't fragment what is logically a shared node.
    return tuple(round(float(c) / eps) * eps for c in p)


def flat_wires_to_polylines(tups, *, eps=1e-6):
    """Convert flat wire tuples to pysim polyline form.

    Returns a dict with keys:
        polylines       : list of (M, 3) np.ndarray
        edge_segments   : list of list[int] — n_seg per edge per polyline
        feed_wire_index : int — polyline holding the excited segment
        feed_arclength  : float — distance along the feed polyline to
                          the midpoint of the excited segment
        feed_voltage    : complex
        junctions       : list of list[(wire_idx, "start"|"end")] —
                          shared-node groups, suitable to pass directly
                          to TriangularPySim(junctions=...). Empty list
                          if every component is a simple path.
    """
    if not tups:
        raise ValueError("no wires to translate")

    # Build endpoint->node map and per-tuple edge list.
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

    # adj[nid] = list of (other_node, edge_index), in registration order.
    adj = [[] for _ in nodes]
    for ei, (a, b, _, _, _) in enumerate(edges):
        adj[a].append((b, ei))
        adj[b].append((a, ei))

    for nid, neigh in enumerate(adj):
        if len(neigh) == 0:
            raise ValueError(f"node {nid} is isolated")

    # Polyline boundaries are exactly the non-degree-2 nodes: degree-1
    # ends an open polyline, degree>=3 is a junction that ends one
    # polyline and starts another. Walk every edge out of every boundary
    # node, threading through degree-2 nodes until the next boundary.
    is_boundary = [len(a) != 2 for a in adj]
    edge_seen = [False] * len(edges)

    polylines = []
    edge_segments = []
    # junction_ends[node_id] -> list of (polyline_index, "start"|"end").
    # Filled as we walk; only meaningful for degree>=3 nodes, but we
    # collect it for all boundary nodes and filter later.
    junction_ends = {nid: [] for nid in range(len(nodes)) if is_boundary[nid]}
    # tup_index -> (polyline_index, edge_index_within)
    edge_to_polyline = {}

    def walk_from(start_nid, first_edge):
        path_nodes = [start_nid]
        path_edges = []
        prev_edge = None
        cur = start_nid
        next_edge = first_edge
        while True:
            edge_seen[next_edge] = True
            path_edges.append(next_edge)
            a, b, _, _, _ = edges[next_edge]
            nxt = b if a == cur else a
            path_nodes.append(nxt)
            cur = nxt
            if is_boundary[cur]:
                return path_nodes, path_edges
            prev_edge = next_edge
            # Degree-2 interior: take the unique outgoing edge.
            next_edge = None
            for _nb, ei in adj[cur]:
                if ei != prev_edge:
                    next_edge = ei
                    break
            assert next_edge is not None, f"degree-2 node {cur} had no continuation"

    for start in range(len(nodes)):
        if not is_boundary[start]:
            continue
        for _nb, ei in adj[start]:
            if edge_seen[ei]:
                continue
            path_nodes, path_edges = walk_from(start, ei)

            polyline_idx = len(polylines)
            polylines.append(np.stack([nodes[n] for n in path_nodes], axis=0))
            edge_segments.append([edges[e][2] for e in path_edges])
            for k, e in enumerate(path_edges):
                edge_to_polyline[edges[e][4]] = (polyline_idx, k)
            junction_ends[path_nodes[0]].append((polyline_idx, "start"))
            junction_ends[path_nodes[-1]].append((polyline_idx, "end"))

    # Closed loops with no boundary node would leave edges unseen.
    if not all(edge_seen):
        raise NotImplementedError(
            "closed loops with no junctions/endpoints are not supported"
        )

    # Junctions = nodes where >= 2 polylines meet. Single-end records
    # (degree-1 free ends) and lone polyline starts aren't junctions.
    junctions = [ends for ends in junction_ends.values() if len(ends) >= 2]

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
    # PyNEC feeds at segment `(n_seg+1)//2` of the excited tuple — the
    # middle segment 1-indexed, i.e. the physical midpoint of the wire.
    # The excited tuple is one edge of its polyline; feed at that edge's
    # midpoint.
    feed_arclength = float(
        edge_lengths[:feed_edge_idx].sum() + 0.5 * edge_lengths[feed_edge_idx]
    )

    return {
        "polylines": polylines,
        "edge_segments": edge_segments,
        "feed_wire_index": feed_pl,
        "feed_arclength": feed_arclength,
        "feed_voltage": complex(voltage),
        "junctions": junctions,
    }
