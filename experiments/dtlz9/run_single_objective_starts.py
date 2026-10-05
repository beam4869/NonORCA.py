"""DTLZ9 sampling from analytical single-objective optima, with matched controls.

Exact optimal seeds contain zero coordinates. The ORIGINAL problem repair and
ORCA generator are deliberately used unchanged; their endpoint perturbations,
sampling shortfall and grouping tie sensitivity are measured, not concealed.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from orca import NonlinearORCAConfig
from orca.nonlinear.fixed_point_generation import generate_fixed_points
from .problem import DTLZ9Problem
from .geometry import (
    edge_summary, expected_signed_matrix, fixed_k_groups, grouping_metrics,
    matrices_from_sampled_blocks, positive_component_groups, sampled_blocks,
)
from .run_phase1_3 import _partial_active_points

ROOT = Path(__file__).resolve().parents[2]
REGIMES = ('single_objective_optima', 'interior_pf_control', 'partial_active_control')


def optimal_seeds(problem):
    """One global minimizer per objective, Pareto-efficient tie breaking.

    Every objective is nonnegative. The associated seed achieves exactly zero,
    so global optimality follows without trusting a numerical NLP termination.
    Nonunique minimizers are resolved to equal shares within each block and to
    the two Pareto endpoints in objective space.
    """
    m, b = problem.num_objectives(), problem.block_size
    low_first = np.zeros(problem.num_variables())
    low_first[problem.block_slices[-1]] = (1.0 / b) ** (1.0 / problem.p)
    low_last = np.full(problem.num_variables(), (1.0 / b) ** (1.0 / problem.p))
    low_last[problem.block_slices[-1]] = 0.0
    seeds = np.vstack([low_first.copy() for _ in range(m - 1)] + [low_last])
    for i, x in enumerate(seeds):
        assert problem.objective_values(x)[i] == 0.0
        assert np.max(problem.constraint_values(x)) <= 1e-12
        assert np.all((x >= 0) & (x <= 1))
    return seeds


def starts_for(problem, regime, seed):
    m, b = problem.num_objectives(), problem.block_size
    if regime == 'single_objective_optima':
        return optimal_seeds(problem)
    if regime == 'interior_pf_control':
        return problem.exact_pf_decisions(np.linspace(0.15, np.pi / 2 - 0.15, m))
    rng = np.random.default_rng(700000 + 10000 * m + 100 * b + seed)
    return _partial_active_points(problem, rng, q=0.5, count=m, inactive_delta=0.08)


def permutation_recovery(matrix, truth, seed, count=30):
    """Avoid confusing fixed-K tie breaking with identifiable recovery."""
    rng = np.random.default_rng(940000 + seed)
    success = []
    for _ in range(count):
        perm = rng.permutation(len(truth))
        groups = fixed_k_groups(matrix[np.ix_(perm, perm)], 2)
        success.append(grouping_metrics(truth[perm], groups)['exact_recovery'])
    return float(np.mean(success))


def run_case(problem, regime, seed, points_per_seed, step_size, output):
    m, b = problem.num_objectives(), problem.block_size
    key = f'm{m}_b{b}_{regime}_s{seed:02d}'
    starts = starts_for(problem, regime, seed)
    repaired = np.vstack([problem.project_feasible(x) for x in starts])
    f_start = np.vstack([problem.objective_values(x) for x in starts])
    f_repaired = np.vstack([problem.objective_values(x) for x in repaired])
    cfg = NonlinearORCAConfig(
        num_groups=2, grouping_method='average_linkage',
        active_constraint_tolerance=1e-10, num_points_per_seed=points_per_seed,
        include_seed_points=True, random_seed=seed, step_size=step_size,
        max_projection_failures=20,
    )
    begin = time.perf_counter()
    points = generate_fixed_points(problem, cfg, seeds=starts)
    blocks = sampled_blocks(problem, points)
    matrices, meta = matrices_from_sampled_blocks(blocks, active_tolerance=1e-10)
    f = np.vstack([problem.objective_values(x) for x in points])
    g = blocks.constraint_values
    active = (g >= -1e-10).sum(axis=1)
    finite = bool(all(np.isfinite(a).all() for a in [points, f, g, blocks.objective_gradients, blocks.constraint_jacobians]))
    pair_mask = ~np.eye(m, dtype=bool)
    total_weight = meta['current_result'].interaction_data.total_weights
    metrics = {
        'M': m, 'n': problem.num_variables(), 'block_size': b, 'regime': regime,
        'seed': seed, 'step_size': step_size, 'points_per_seed': points_per_seed,
        'input_seeds': len(starts), 'distinct_input_seeds': len(np.unique(starts, axis=0)),
        'selected_points': len(points), 'target_points_before_dedup': len(starts) * (points_per_seed + 1),
        'max_feasibility_violation': float(max(0.0, g.max(), -points.min(), points.max()-1)),
        'finite_values_and_derivatives': finite,
        'all_active_fraction': float(np.mean(active == m-1)),
        'any_active_fraction': float(np.mean(active > 0)),
        'active_constraints_mean': float(active.mean()),
        'seed_repair_max_objective_change': float(np.max(np.abs(f_repaired - f_start))),
        'seed_repair_min_coordinate': float(repaired.min()),
        'repaired_seed_all_active_fraction': float(np.mean(np.all(np.vstack([problem.constraint_values(x) for x in repaired]) >= -1e-10, axis=1))),
        'current_positive_weight_pair_fraction': float(np.mean(total_weight[pair_mask] > 0)),
        'f_last_min': float(f[:, -1].min()), 'f_last_max': float(f[:, -1].max()),
        'seconds': time.perf_counter() - begin,
    }
    assert finite and metrics['max_feasibility_violation'] <= 1e-10
    truth = np.array([1] * (m-1) + [2])
    rows = []
    for method in ['Raw-Cosine', 'ORCA-current', 'ORCA-joint']:
        matrix = matrices[method]
        common = {
            **metrics, 'method': method,
            'finite_matrix_fraction': float(np.mean(np.isfinite(matrix))),
            'permuted_fixed_k_recovery': permutation_recovery(matrix, truth, seed),
            'signed_matrix_mae': float(np.nanmean(np.abs(matrix-expected_signed_matrix(m)))),
            **edge_summary(matrix),
        }
        for mode, groups in [('fixed_K2', fixed_k_groups(matrix, 2)),
                             ('positive_components', positive_component_groups(matrix, 1e-8))]:
            rows.append({**common, 'grouping_mode': mode,
                         'labels': ' '.join(map(str, groups)), **grouping_metrics(truth, groups)})
    payload = dict(points=points, objective_values=f, constraint_values=g,
                   input_seeds=starts, repaired_seeds=repaired,
                   input_seed_objectives=f_start, repaired_seed_objectives=f_repaired,
                   current_pair_total_weights=total_weight)
    for method, matrix in matrices.items():
        payload['matrix_'+method.replace('-', '_')] = matrix
    np.savez_compressed(output/'runs'/f'{key}.npz', **payload)
    return metrics, rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT/'outputs/dtlz9_single_objective_starts')
    parser.add_argument('--seeds', type=int, default=10)
    parser.add_argument('--objectives', type=int, nargs='+', default=[3,5,10])
    parser.add_argument('--block-sizes', type=int, nargs='+', default=[1,10])
    parser.add_argument('--points-per-seed', type=int, default=3)
    parser.add_argument('--step-size', type=float, default=0.01)
    args=parser.parse_args()
    if args.seeds < 1 or args.points_per_seed < 0 or args.step_size <= 0:
        parser.error('seeds>=1, points-per-seed>=0 and step-size>0 are required')
    output=args.output_dir.resolve()
    if (output/'grouping_runs.csv').exists():
        parser.error('Output already contains results; select a new directory.')
    (output/'runs').mkdir(parents=True,exist_ok=True)
    start=time.perf_counter()
    sampling, grouping, seed_checks=[],[],[]
    for m in args.objectives:
        for b in args.block_sizes:
            problem=DTLZ9Problem(m,m*b)
            exact=optimal_seeds(problem)
            try:
                problem.objective_gradients(exact[0])
                singular_error=None
            except ValueError as exc:
                singular_error=str(exc)
            seed_checks.append({'M':m,'n':m*b,'unique_exact_seeds':len(np.unique(exact,axis=0)),
                                'assigned_optimal_values': [float(problem.objective_values(x)[i]) for i,x in enumerate(exact)],
                                'exact_endpoint_gradient_error':singular_error})
            for regime in REGIMES:
                for seed in range(args.seeds):
                    metrics, rows=run_case(problem,regime,seed,args.points_per_seed,args.step_size,output)
                    sampling.append(metrics); grouping.extend(rows)
                print(f'Completed M={m}, block={b}, {regime}, {args.seeds} seeds',flush=True)
                pd.DataFrame(sampling).to_csv(output/'sampling_runs.csv',index=False)
                pd.DataFrame(grouping).to_csv(output/'grouping_runs.csv',index=False)
    sf=pd.DataFrame(sampling); gf=pd.DataFrame(grouping)
    sf.groupby(['M','n','block_size','regime'],as_index=False).agg(
        runs=('seed','size'), selected_mean=('selected_points','mean'),
        selected_min=('selected_points','min'),selected_max=('selected_points','max'),
        target_points=('target_points_before_dedup','first'),
        all_active_fraction=('all_active_fraction','mean'),
        any_active_fraction=('any_active_fraction','mean'),
        max_violation=('max_feasibility_violation','max'),
        repair_objective_change=('seed_repair_max_objective_change','max'),
    ).to_csv(output/'sampling_summary.csv',index=False)
    gf.groupby(['M','n','block_size','regime','method','grouping_mode'],as_index=False).agg(
        runs=('seed','size'), exact_recovery_rate=('exact_recovery','mean'),
        mean_ari=('ari','mean'), groups_mean=('estimated_groups','mean'),
        permuted_fixed_k_recovery=('permuted_fixed_k_recovery','mean'),
        within_signed_mean=('within_mean','mean'),between_signed_mean=('between_mean','mean'),
        separation_margin_mean=('separation_margin','mean'),
    ).to_csv(output/'grouping_summary.csv',index=False)
    source_paths=['experiments/dtlz9/run_single_objective_starts.py','experiments/dtlz9/problem.py',
                  'experiments/dtlz9/geometry.py','src/orca/nonlinear/fixed_point_generation.py']
    info={
        'date':'2026-10-05','python':platform.python_version(),'numpy':np.__version__,
        'parameters':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'sampling_runs':len(sf),'grouping_rows':len(gf),'seconds':time.perf_counter()-start,
        'endpoint_checks':seed_checks,
        'core_changed':False,
        'seed_tie_break':'Pareto-efficient endpoints, equal allocation within each objective block',
        'repair_note':'Original project_feasible clips x to [1e-30,1], including before the first derivative evaluation.',
        'control_note':'All three arms receive M seed entries and identical generator settings. The optima arm has repeated endpoints. The partial-active control uses M starts rather than the older experiment\'s two.',
        'sources':{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in source_paths},
    }
    (output/'run_metadata.json').write_text(json.dumps(info,indent=2)+'\n')
    print(json.dumps({'sampling_runs':len(sf),'grouping_rows':len(gf),'seconds':info['seconds']}),flush=True)


if __name__=='__main__':
    main()
