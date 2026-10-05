"""Entry points for nonlinear ORCA."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from orca.config import NonlinearORCAConfig
from orca.data_schema import ORCAInteractionData, ORCAResult, SampledGradientBlocks
from orca.nonlinear.fixed_point_generation import generate_fixed_points
from orca.nonlinear.gradient_evaluation import evaluate_sampled_gradient_blocks
from orca.nonlinear.problem_interface import NonlinearORCAProblem
from orca.utils.correlation_aggregation import aggregate_interactions_to_adjacency
from orca.utils.interaction_weights import compute_interaction_weights
from orca.utils.local_objective_interactions import compute_nonlinear_local_objective_interactions
from orca.utils.objective_grouping import group_objectives


def nonlinear_orca_from_sampled_gradients(
    blocks: SampledGradientBlocks,
    config: Optional[NonlinearORCAConfig] = None,
) -> ORCAResult:
    """Run nonlinear ORCA from already sampled gradients and Jacobians."""

    cfg = config or NonlinearORCAConfig()
    strengths, valid_mask, local_metadata = compute_nonlinear_local_objective_interactions(
        blocks.objective_gradients,
        blocks.constraint_jacobians,
        constraint_values=blocks.constraint_values,
        active_constraint_tolerance=cfg.active_constraint_tolerance,
        atol=cfg.atol,
    )
    weights = compute_interaction_weights(
        strengths,
        valid_mask=valid_mask,
        equality_mask=None,
        alpha_weight=cfg.alpha_weight,
        beta_weight=cfg.beta_weight,
    )
    adj, total_weights, weighted_strengths = aggregate_interactions_to_adjacency(
        strengths,
        weights,
        zero_weight_value=cfg.zero_weight_value,
        self_correlation=cfg.self_correlation,
        min_correlation=cfg.min_correlation,
        max_correlation=cfg.max_correlation,
        rescaling_factor=cfg.correlation_rescaling_factor,
        atol=cfg.atol,
    )
    groups = group_objectives(
        adj,
        cfg.num_groups,
        method=cfg.grouping_method,
        resolution_start=cfg.leiden_resolution_start,
        resolution_stop=cfg.leiden_resolution_stop,
        resolution_steps=cfg.leiden_resolution_steps,
    )
    interaction_data = ORCAInteractionData(
        strengths=strengths,
        weights=weights,
        total_weights=total_weights,
        valid_mask=valid_mask,
        projected_directions=local_metadata.pop("projected_directions", None),
        metadata={**local_metadata, "weighted_strengths": weighted_strengths},
    )
    return ORCAResult(
        adj_matrix=adj,
        groups=groups,
        interaction_data=interaction_data,
        input_data=blocks,
        metadata={
            "algorithm": "nonlinear_orca",
            "num_groups": cfg.num_groups,
            "alpha_weight": cfg.alpha_weight,
            "beta_weight": cfg.beta_weight,
            "grouping_method": cfg.grouping_method,
            "step_size": cfg.step_size,
            "num_points_per_seed": cfg.num_points_per_seed,
            "active_constraint_tolerance": cfg.active_constraint_tolerance,
        },
    )


def nonlinear_orca(
    problem: NonlinearORCAProblem,
    config: Optional[NonlinearORCAConfig] = None,
    *,
    points: Optional[Sequence[np.ndarray] | np.ndarray] = None,
    seeds: Optional[Sequence[np.ndarray]] = None,
) -> ORCAResult:
    """Run the full nonlinear ORCA workflow.

    If ``points`` are supplied, they are used directly as selected fixed points.
    Otherwise, fixed points are generated from ``seeds`` or from
    ``problem.initial_points()``.
    """

    cfg = config or NonlinearORCAConfig()
    if points is None:
        pts = generate_fixed_points(problem, cfg, seeds=seeds)
    else:
        pts = np.asarray(points, dtype=float)
    blocks = evaluate_sampled_gradient_blocks(problem, pts)
    return nonlinear_orca_from_sampled_gradients(blocks, config=cfg)
