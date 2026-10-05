"""ORCA objective-reduction framework."""

from orca.config import LinearORCAConfig, NonlinearORCAConfig, ORCAConfig
from orca.data_schema import (
    LinearMatrixBlocks,
    LocalLinearizationBlocks,
    ORCAInteractionData,
    ORCAResult,
    SampledGradientBlocks,
)
from orca.linear.main import linear_orca_from_blocks, linear_orca_from_matrices, linear_orca_from_pyomo
from orca.nonlinear.main import nonlinear_orca, nonlinear_orca_from_sampled_gradients

__all__ = [
    "ORCAConfig",
    "LinearORCAConfig",
    "NonlinearORCAConfig",
    "LinearMatrixBlocks",
    "SampledGradientBlocks",
    "LocalLinearizationBlocks",
    "ORCAInteractionData",
    "ORCAResult",
    "linear_orca_from_blocks",
    "linear_orca_from_matrices",
    "linear_orca_from_pyomo",
    "nonlinear_orca",
    "nonlinear_orca_from_sampled_gradients",
]
