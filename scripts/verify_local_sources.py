"""Validate manuscript results against the recovered local source tree.

This recomputes statistics, the ellipse example and CCUS aggregation/conditional
loss from stored numerical inputs. It does not rerun the Julia NLP solves.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load(name: str, relative: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def audit(output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    checks = []

    def check(name, passed, **details):
        checks.append(dict(name=name, passed=bool(passed), **details))

    raw = pd.read_csv(ROOT / 'data/dtlz/orca_dtlz5_5_16_nsga3_raw.csv')
    recorded = pd.read_csv(ROOT / 'data/dtlz/orca_dtlz5_5_16_nsga3_summary.csv')
    driver = load('verify_dtlz_table2', 'experiments/dtlz/run_dtlz5_5_16_orca_nsga3.py')
    rebuilt = driver.summarize(raw)
    check('Table 2: three methods, paired seeds 0..4', len(raw) == 15 and
          all(sorted(group.seed.tolist()) == list(range(5)) for _, group in raw.groupby('method')))
    for column in recorded.select_dtypes(include='number').columns:
        check('Table 2: raw-to-summary ' + column,
              np.allclose(rebuilt[column], recorded[column], atol=1e-12, rtol=1e-12, equal_nan=True))
    rows = []
    for method, group in raw.groupby('method', sort=False):
        for metric in ['total_seconds', 'approx_hv', 'empirical_igd']:
            rows.append(dict(method=method, metric=metric, n=len(group), mean=group[metric].mean(),
                             reported_population_sd=group[metric].std(ddof=0),
                             sample_sd=group[metric].std(ddof=1)))
    pd.DataFrame(rows).to_csv(output / 'table2_standard_deviation_audit.csv', index=False)
    rebuilt.to_csv(output / 'table2_summary_recomputed.csv', index=False)

    table1 = pd.read_csv(ROOT / 'data/dtlz/orca_dtlz5_dtlz6_gradient_grouping_raw.csv')
    orca = table1[table1.method == 'ORCA']
    check('Table 1: 30 archived ORCA runs all recover expected groups', len(orca) == 30 and
          orca.exact_match.eq(1).all() and np.allclose(orca.ari, 1))
    check('Table 1: six cases each with seeds 0..4', orca.dataset.nunique() == 6 and
          all(sorted(group.seed.tolist()) == list(range(5)) for _, group in orca.groupby('dataset')))

    demo = load('verify_oval', 'examples/ellipse/oval_objective_reduction_demo.py')
    config = demo.NonlinearORCAConfig(num_groups=3, num_points_per_seed=16,
        include_seed_points=True, random_seed=14, step_size=0.24,
        grouping_method='average_linkage', active_constraint_tolerance=1e-8)
    result = demo.nonlinear_orca(demo.OvalORCAProblem(), config)
    stored_points = pd.read_csv(ROOT / 'examples/ellipse/oval_selected_points.csv')
    stored_matrix = pd.read_csv(ROOT / 'examples/ellipse/oval_correlation_matrix.csv', index_col=0)
    points = result.input_data.points
    check('Ellipse: rerun reproduces all 68 points', len(points) == 68 and
          np.allclose(points, stored_points[['x1','x2']], atol=2e-8, rtol=0),
          max_error=float(np.max(np.abs(points - stored_points[['x1','x2']].to_numpy()))))
    check('Ellipse: rerun reproduces stored correlation matrix',
          np.allclose(result.adj_matrix, stored_matrix, atol=2e-8, rtol=0),
          max_error=float(np.max(np.abs(result.adj_matrix-stored_matrix.to_numpy()))))
    check('Ellipse: recovered groups', demo.group_sets_from_labels(result.groups) == [[1,2],[3],[4]])
    pd.DataFrame(result.adj_matrix).to_csv(output/'ellipse_adjacency_recomputed.csv', index=False)

    ccus = load('verify_ccus_aggregation', 'experiments/ccus/scripts/run_paper_aligned_orca.py')
    loss = load('verify_ccus_loss', 'experiments/ccus/scripts/analyze_exact_information_loss.py')
    results = ROOT / 'experiments/ccus/results'
    all_pairs, loss_rows = [], []
    for profile, label, count in [('direct_use_expansion','paper_main',20),
                                   ('supply_chain_condition2','condition2_main',10)]:
        prefix = f'{profile}_paper_orca_{label}_'
        points = pd.read_csv(results / f'{prefix}points.csv')
        gradients = pd.read_csv(results / f'{prefix}objective_gradients.csv')
        dz = ccus.numeric_columns(gradients, 'dz')
        eq = pd.read_csv(results / f'{prefix}equality_jacobian.csv')[dz].to_numpy()
        ineq = pd.read_csv(results / f'{prefix}inequality_jacobian.csv')[dz].to_numpy()
        gc = ccus.numeric_columns(points, 'g')
        archived = pd.read_csv(results/f'{prefix}pairwise.csv')
        grouped = pd.read_csv(results/f'{prefix}groupings.csv')
        check(profile + ': archived point and replicate counts',
              len(points) == 123*count and points.replicate.nunique() == count)
        max_error = 0.0
        partition_match = True
        for rep, sub in points.groupby('replicate', sort=True):
            jac = np.stack([gradients[gradients.point_id == pid].set_index('objective')
                            .loc[ccus.OBJECTIVES,dz].to_numpy() for pid in sub.point_id])
            for handling, include_eq in [('equality_tangent', True), ('inequalities_only', False)]:
                actual = ccus.aggregate_one_replicate(jac, sub[gc].to_numpy(), ineq, eq,
                                                    include_equalities=include_eq)
                reference = archived[(archived.replicate == rep) &
                                     (archived.constraint_handling == handling)].set_index('pair')
                for i,j in ccus.PAIRS:
                    pair = f'{ccus.OBJECTIVES[i]}__{ccus.OBJECTIVES[j]}'
                    value = float(actual['adjacency'][i,j])
                    error = abs(value - reference.loc[pair,'orca_adjacency_01'])
                    max_error = max(max_error,error)
                    all_pairs.append(dict(profile=profile,replicate=int(rep),handling=handling,
                                          pair=pair,recomputed=value,absolute_error=error))
                saved = grouped[(grouped.replicate == rep) & (grouped.constraint_handling == handling)].iloc[0]
                for method in ['average','leiden']:
                    a = actual[method]
                    b = saved[[method+'_'+obj for obj in ccus.OBJECTIVES]].to_numpy()
                    partition_match &= np.array_equal(np.equal.outer(a,a),np.equal.outer(b,b))
        check(profile + ': all archived ORCA pair values reproduced', max_error < 1e-10,
              max_absolute_error=max_error, replicates=count, constraint_modes=2)
        check(profile + ': both grouping partitions reproduced', partition_match)

        # Recompute the normalization and conditional-slice analysis from the
        # stored local NLP solutions; no claim of global or new NLP solutions.
        pref = f'{profile}_exact_info_loss_quantile21_'
        pts = pd.read_csv(results/f'{pref}points.csv')
        frontier = pd.read_csv(results/f'{profile}_pareto_frontier.csv')
        vals = np.vstack([pts[loss.OBJECTIVES],frontier[loss.OBJECTIVES]])
        provisional_lo = vals.min(axis=0)
        provisional_range = vals.max(axis=0)-provisional_lo
        mask = loss.nondominated_mask((vals-provisional_lo)/provisional_range)
        lower,upper = vals[mask].min(axis=0),vals[mask].max(axis=0)
        pts['globally_nondominated'] = mask[:len(pts)]
        slices = loss.calculate_slices(pts,frontier,lower,upper-lower,11,21,
                                      int(pts.retained_index.max()),int(pts.epsilon_index.max()))
        summary = loss.summarize(slices).set_index('grouping')
        reference = pd.read_csv(results/f'{pref}summary.csv')
        reference = reference[(reference.retained_resolution==11)&(reference.epsilon_resolution==21)].set_index('grouping')
        delta = float(np.max(np.abs(summary.loc[reference.index,'mean_information_loss'] - reference.mean_information_loss)))
        check(profile + ': conditional loss reproduced from stored NLP points',delta < 1e-10,
              max_absolute_error=delta)
        for grouping,row in summary.iterrows():
            loss_rows.append(dict(profile=profile,grouping=grouping,mean_information_loss=row.mean_information_loss))
    pd.DataFrame(all_pairs).to_csv(output/'ccus_pairwise_recomputed.csv', index=False)
    pd.DataFrame(loss_rows).to_csv(output/'ccus_conditional_loss_recomputed.csv',index=False)
    report=dict(checks=checks, passed=all(c['passed'] for c in checks),
                warnings=['Table 2 caption says sample SD; archived driver and values use ddof=0.',
                          'Julia NLP sampling/Pareto/conditional solves were not rerun.',
                          'Figure 4 full DTLZ5(5,20) heatmap run grid has not been located.'])
    (output/'local_verification.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'passed':report['passed'],'checks':len(checks),
                      'failed':[c for c in checks if not c['passed']]},indent=2))
    return report


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=ROOT/'outputs/local_verification')
    args=parser.parse_args()
    raise SystemExit(0 if audit(args.output)['passed'] else 1)
