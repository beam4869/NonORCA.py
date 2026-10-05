"""Objective grouping utilities for ORCA adjacency matrices."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np


GROUP_LABEL_START = 1


def relabel_groups(labels: np.ndarray) -> np.ndarray:
    """Relabel arbitrary community labels to 1, 2, ... in order of first appearance."""

    arr = np.asarray(labels, dtype=int)
    mapping: dict[int, int] = {}
    out = np.zeros_like(arr, dtype=int)
    next_label = GROUP_LABEL_START
    for idx, label in enumerate(arr.tolist()):
        if label not in mapping:
            mapping[label] = next_label
            next_label += 1
        out[idx] = mapping[label]
    return out


def group_objectives_average_linkage(adj_matrix: np.ndarray, num_groups: int) -> np.ndarray:
    """Group objectives by deterministic average-linkage clustering on similarity."""

    adj = np.asarray(adj_matrix, dtype=float)
    n = adj.shape[0]
    _validate_grouping_inputs(adj, num_groups)
    if num_groups == n:
        return np.arange(GROUP_LABEL_START, n + GROUP_LABEL_START, dtype=int)
    if num_groups == 1:
        return np.ones(n, dtype=int)

    groups = [{idx} for idx in range(n)]
    while len(groups) > num_groups:
        best_pair: Optional[tuple[int, int]] = None
        best_similarity = float("-inf")
        for a in range(len(groups)):
            for b in range(a + 1, len(groups)):
                sims = [adj[i, j] for i in groups[a] for j in groups[b]]
                sim = float(np.mean(sims)) if sims else float("-inf")
                if sim > best_similarity:
                    best_similarity = sim
                    best_pair = (a, b)
        if best_pair is None:
            raise RuntimeError("failed to identify a pair of objective groups to merge")
        a, b = best_pair
        groups[a] = groups[a].union(groups[b])
        del groups[b]

    labels = np.zeros(n, dtype=int)
    for label, group in enumerate(groups, start=GROUP_LABEL_START):
        for idx in sorted(group):
            labels[idx] = label
    return relabel_groups(labels)


def group_objectives_leiden(
    adj_matrix: np.ndarray,
    num_groups: int,
    *,
    resolution_grid: Optional[Sequence[float]] = None,
    resolution_start: float = 0.01,
    resolution_stop: float = 5.0,
    resolution_steps: int = 200,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Group objectives using Leiden community detection.

    The function searches over a resolution grid and returns the membership with
    a number of groups closest to ``num_groups``. ``random_seed`` is forwarded
    to Leiden so archived analyses can reproduce the same partition.
    """

    try:
        import igraph as ig  # type: ignore
        import leidenalg  # type: ignore
    except ImportError as exc:
        raise ImportError("Install python-igraph and leidenalg to use Leiden grouping") from exc

    adj = np.asarray(adj_matrix, dtype=float)
    n = adj.shape[0]
    _validate_grouping_inputs(adj, num_groups)
    if num_groups == n:
        return np.arange(GROUP_LABEL_START, n + GROUP_LABEL_START, dtype=int)
    if num_groups == 1:
        return np.ones(n, dtype=int)

    graph_adj = adj.copy()
    np.fill_diagonal(graph_adj, 0.0)
    graph = ig.Graph.Weighted_Adjacency(graph_adj.tolist(), mode="UNDIRECTED", attr="weight", loops=False)

    if resolution_grid is None:
        resolution_grid = np.linspace(resolution_start, resolution_stop, resolution_steps)

    best_membership: Optional[np.ndarray] = None
    best_gap = n + 1
    for resolution in resolution_grid:
        partition = leidenalg.find_partition(
            graph,
            leidenalg.RBConfigurationVertexPartition,
            weights="weight",
            resolution_parameter=float(resolution),
            seed=random_seed,
        )
        membership = np.asarray(partition.membership, dtype=int)
        found_groups = len(np.unique(membership))
        gap = abs(found_groups - num_groups)
        if gap < best_gap:
            best_gap = gap
            best_membership = membership
        if found_groups == num_groups:
            break

    if best_membership is None:
        raise RuntimeError("Leiden did not return a valid membership")
    return relabel_groups(best_membership)


def group_objectives(
    adj_matrix: np.ndarray,
    num_groups: int,
    *,
    method: str = "auto",
    resolution_grid: Optional[Sequence[float]] = None,
    resolution_start: float = 0.01,
    resolution_stop: float = 5.0,
    resolution_steps: int = 200,
    random_seed: Optional[int] = None,
) -> np.ndarray:
    """Group objectives from an ORCA adjacency matrix."""

    method_normalized = method.lower()
    if method_normalized in {"auto", "leiden"}:
        try:
            return group_objectives_leiden(
                adj_matrix,
                num_groups,
                resolution_grid=resolution_grid,
                resolution_start=resolution_start,
                resolution_stop=resolution_stop,
                resolution_steps=resolution_steps,
                random_seed=random_seed,
            )
        except ImportError:
            if method_normalized == "leiden":
                raise
    if method_normalized in {"auto", "average_linkage", "average-linkage"}:
        return group_objectives_average_linkage(adj_matrix, num_groups)
    raise ValueError(f"Unknown objective grouping method: {method}")


def _validate_grouping_inputs(adj: np.ndarray, num_groups: int) -> None:
    if adj.ndim != 2 or adj.shape[0] != adj.shape[1]:
        raise ValueError("adj_matrix must be square")
    n = adj.shape[0]
    if num_groups < 1 or num_groups > n:
        raise ValueError("num_groups must be between 1 and the number of objectives")
