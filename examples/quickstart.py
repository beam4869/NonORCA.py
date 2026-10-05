"""Small nonlinear ORCA example; needs only the base package dependencies."""
import numpy as np

from orca import NonlinearORCAConfig, nonlinear_orca
from orca.benchmarks import DTLZ5Problem


def main():
    # Small deterministic seed set for demonstrating the API, not a paper run.
    problem = DTLZ5Problem(
        intrinsic_dimension=2,
        num_objectives=3,
        k_tail=2,
        initial_point_strategy="deterministic",
    )
    result = nonlinear_orca(
        problem,
        NonlinearORCAConfig(
            num_groups=2,
            num_points_per_seed=8,
            step_size=0.03,
            random_seed=0,
            grouping_method="average_linkage",
        ),
    )
    np.set_printoptions(precision=4, suppress=True)
    print("Selected points:", len(result.input_data.points))
    print("ORCA adjacency matrix:\n", result.adj_matrix)
    print("Objective group labels:", result.groups)


if __name__ == "__main__":
    main()
