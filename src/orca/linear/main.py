"""Entry points for linear orca_python."""

from __future__ import annotations

from typing import Optional, Sequence

import numpy as np

from orca.config import LinearORCAConfig
from orca.data_schema import LinearMatrixBlocks, ORCAInteractionData, ORCAResult
from orca.linear.matrix_extraction import extract_linear_blocks_pyomo
from orca.utils.correlation_aggregation import aggregate_interactions_to_adjacency
from orca.utils.interaction_weights import compute_interaction_weights
from orca.utils.local_objective_interactions import compute_linear_local_objective_interactions
from orca.utils.objective_grouping import group_objectives


def linear_orca_from_blocks(blocks: LinearMatrixBlocks, config: Optional[LinearORCAConfig] = None) -> ORCAResult:
    """Run linear ORCA from extracted matrix blocks."""

    cfg = config or LinearORCAConfig()
    strengths, valid_mask, local_metadata = compute_linear_local_objective_interactions(
        blocks.Jobj,
        blocks.Jineq,
        blocks.Jeq,
        filter_inactive_inequalities=cfg.filter_inactive_inequalities,
        atol=cfg.atol,
    )
    equality_mask = local_metadata.get("equality_mask")
    weights = compute_interaction_weights(
        strengths,
        valid_mask=valid_mask,
        equality_mask=equality_mask,
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
            "algorithm": "linear_orca",
            "num_groups": cfg.num_groups,
            "alpha_weight": cfg.alpha_weight,
            "beta_weight": cfg.beta_weight,
            "grouping_method": cfg.grouping_method,
            "filter_inactive_inequalities": cfg.filter_inactive_inequalities,
        },
    )


def linear_orca_from_matrices(
    Jineq: np.ndarray,
    Jobj: np.ndarray,
    Jeq: Optional[np.ndarray] = None,
    *,
    config: Optional[LinearORCAConfig] = None,
    var_names: Optional[list[str]] = None,
) -> ORCAResult:
    """Run linear ORCA from matrix arrays."""

    blocks = LinearMatrixBlocks(Jineq=np.asarray(Jineq, dtype=float), Jobj=np.asarray(Jobj, dtype=float), Jeq=Jeq, var_names=var_names)
    return linear_orca_from_blocks(blocks, config=config)


def linear_orca_from_pyomo(
    model,
    objective_exprs: Sequence,
    *,
    config: Optional[LinearORCAConfig] = None,
) -> tuple[ORCAResult, LinearMatrixBlocks]:
    """Extract linear blocks from a Pyomo model and run linear ORCA."""

    blocks = extract_linear_blocks_pyomo(model, objective_exprs)
    result = linear_orca_from_blocks(blocks, config=config)
    return result, blocks
