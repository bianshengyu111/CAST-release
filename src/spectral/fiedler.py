# -*- coding: utf-8 -*-
"""Spectral utilities: exact Fiedler vector and the double-sweep BFS approximation.

Manuscript Sec. III-F.  The spectral edge score is the first-order marginal gain
``G_ij^sp = (q_i - q_j)^2`` (Ghosh & Boyd), where ``q`` is the unit Fiedler vector of
the graph Laplacian.  For ``N_s < 500`` the vector is computed by Lanczos
(``scipy.sparse.linalg.eigsh``); above that threshold a linear-time double-sweep BFS
approximation is used.  The approximation cuts per-slot computation from ~1.24 s to
~0.155 s at N = 1,401 with cosine similarity > 0.94 (Table 3).
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np


def _adjacency(n: int, edges: Iterable[Sequence[int]]) -> list[list[int]]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for i, j in edges:
        adj[int(i)].append(int(j))
        adj[int(j)].append(int(i))
    return adj


def laplacian(n: int, edges: Iterable[Sequence[int]]) -> np.ndarray:
    L = np.zeros((n, n))
    for i, j in edges:
        i, j = int(i), int(j)
        L[i, i] += 1.0
        L[j, j] += 1.0
        L[i, j] -= 1.0
        L[j, i] -= 1.0
    return L


def fiedler_exact(n: int, edges: Iterable[Sequence[int]]) -> np.ndarray:
    """Unit Fiedler vector via Lanczos; zeros when the graph has < 2 nodes."""
    if n < 2:
        return np.zeros(n)
    edges = list(edges)
    if not edges:
        return np.zeros(n)
    L = laplacian(n, edges)
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    vals, vecs = eigsh(csr_matrix(L), k=2, which="SM")
    order = np.argsort(vals)
    q = vecs[:, order[1]]
    # deterministic sign convention: largest-magnitude entry positive
    if q[np.argmax(np.abs(q))] < 0:
        q = -q
    nrm = np.linalg.norm(q)
    return q / nrm if nrm > 0 else q


def algebraic_connectivity(n: int, edges: Iterable[Sequence[int]]) -> float:
    """lambda_2, the second-smallest Laplacian eigenvalue (0 if disconnected)."""
    if n < 2:
        return 0.0
    edges = list(edges)
    if len(edges) < n - 1:
        return 0.0
    L = laplacian(n, edges)
    from scipy.sparse import csr_matrix
    from scipy.sparse.linalg import eigsh
    try:
        vals = eigsh(csr_matrix(L), k=2, which="SM", return_eigenvectors=False)
    except Exception:
        vals = np.linalg.eigvalsh(L)[:2]
    vals = np.sort(vals)
    return float(max(vals[1], 0.0))


def _bfs_dist(adj: list[list[int]], root: int) -> np.ndarray:
    n = len(adj)
    dist = np.full(n, -1.0)
    dist[root] = 0.0
    frontier = [root]
    while frontier:
        nxt = []
        for u in frontier:
            for v in adj[u]:
                if dist[v] < 0:
                    dist[v] = dist[u] + 1.0
                    nxt.append(v)
        frontier = nxt
    return dist


def fiedler_approx(n: int, edges: Iterable[Sequence[int]]) -> np.ndarray:
    """Double-sweep BFS approximation of the Fiedler vector (O(M)).

    (1) BFS from an arbitrary node r to the farthest node u; (2) BFS from u to the
    farthest node v; (3) q_i = (d(u,i) - d(v,i)) / max_j |d(u,j) - d(v,j)|.
    """
    if n < 2:
        return np.zeros(n)
    adj = _adjacency(n, edges)
    # start from the highest-degree node for a more stable first sweep
    r = int(np.argmax([len(a) for a in adj])) if adj else 0
    d_r = _bfs_dist(adj, r)
    reach = np.flatnonzero(d_r >= 0)
    if reach.size == 0:
        return np.zeros(n)
    u = int(reach[np.argmax(d_r[reach])])
    du = _bfs_dist(adj, u)
    fu = np.flatnonzero(du >= 0)
    if fu.size == 0:
        return np.zeros(n)
    v = int(fu[np.argmax(du[fu])])
    dv = _bfs_dist(adj, v)
    raw = du - dv
    raw[raw < 0] = raw[raw < 0]      # keep sign; unreachable nodes handled below
    unreachable = (du < 0) | (dv < 0)
    raw[unreachable] = 0.0
    m = np.max(np.abs(raw))
    q = raw / m if m > 0 else raw
    nrm = np.linalg.norm(q)
    return q / nrm if nrm > 0 else q


def fiedler_vector(n: int, edges: Iterable[Sequence[int]],
                   node_threshold: int = 500) -> np.ndarray:
    """Dispatch to the exact or approximate Fiedler computation."""
    return fiedler_exact(n, edges) if n < node_threshold else fiedler_approx(n, edges)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))
