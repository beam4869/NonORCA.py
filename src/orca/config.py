"""Configuration objects for linear and nonlinear ORCA workflows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class ORCAConfig:
    """Shared ORCA configuration.

    Parameters
    ----------
    num_groups:
        Desired number of objective groups returned by objective grouping.
    alpha_weight:
        Amplitude of the logistic competition weight. Larger values discount
        strongly positive local correlations more aggressively.
    beta_weight:
        Steepness of the logistic competition weight around zero local
        interaction strength.
    zero_weight_value:
        Neutral adjacency value used when an objective pair has zero total
        valid interaction weight.
    grouping_method:
        Objective grouping method. Supported values are ``"auto"``,
        ``"leiden"``, and ``"average_linkage"``.
    atol:
        Absolute numerical tolerance used for zero-norm checks and division
        safeguards.
    """

    num_groups: int = 2
    alpha_weight: float = 0.9
    beta_weight: float = 100.0
    zero_weight_value: float = 0.5
    grouping_method: str = "auto"
    atol: float = 1.0e-14

    # Adjacency rescaling and clipping.
    self_correlation: float = 1.0
    min_correlation: float = 0.0
    max_correlation: float = 1.0
    correlation_rescaling_factor: float = 0.5

    # Leiden resolution search.
    leiden_resolution_start: float = 0.01
    leiden_resolution_stop: float = 5.0
    leiden_resolution_steps: int = 200


@dataclass(frozen=True)
class LinearORCAConfig(ORCAConfig):
    """Configuration for linear ORCA.

    Parameters
    ----------
    filter_inactive_inequalities:
        If true, include an inequality surface only when both objective descent
        directions point toward the infeasible side of that inequality. This
        matches the linear ORCA boundary-relevance check used in the earlier
        implementation.
    """

    filter_inactive_inequalities: bool = True


@dataclass(frozen=True)
class NonlinearORCAConfig(ORCAConfig):
    """Configuration for nonlinear ORCA.

    Parameters
    ----------
    step_size:
        Step length used when generating new feasible fixed points. This is the
        nonlinear fixed-point generation step size, not the logistic weighting
        ``alpha_weight``.
    num_points_per_seed:
        Number of additional selected fixed points generated from each seed.
    include_seed_points:
        If true, seed points are included in the selected-point set.
    random_seed:
        Optional seed for reproducible random conic direction generation.
    min_point_distance:
        Minimum Euclidean distance required before a generated point is kept.
    max_projection_failures:
        Maximum failed attempts allowed while trying to generate fixed points
        from one seed.
    active_constraint_tolerance:
        If set, only constraints satisfying ``g_k(x_n) >= -tol`` are used in
        nonlinear local interaction calculations. If ``None``, all sampled
        constraint Jacobian rows are used.
    constraint_selection:
        Rule used when selecting a constraint normal for fixed-point direction
        generation. Supported values are ``"active_or_nearest"`` and
        ``"nearest"``.
    normalize_generated_direction:
        If true, normalize the random conic direction before applying
        ``step_size``.
    """

    step_size: float = 0.03
    num_points_per_seed: int = 40
    include_seed_points: bool = True
    random_seed: Optional[int] = 0
    min_point_distance: float = 1.0e-8
    max_projection_failures: int = 20
    active_constraint_tolerance: Optional[float] = None
    constraint_selection: str = "active_or_nearest"
    normalize_generated_direction: bool = True
