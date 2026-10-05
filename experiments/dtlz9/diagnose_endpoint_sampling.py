"""Factorial diagnosis of decision-space step and duplicate threshold at DTLZ9 optima."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from orca import NonlinearORCAConfig
from orca.nonlinear.fixed_point_generation import generate_fixed_points
from .problem import DTLZ9Problem
from .geometry import sampled_blocks, matrices_from_sampled_blocks, grouping_metrics, fixed_k_groups, edge_summary
from .run_single_objective_starts import optimal_seeds, permutation_recovery


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,default=Path('outputs/dtlz9_endpoint_diagnosis'))
    parser.add_argument('--seeds',type=int,default=10)
    args=parser.parse_args()
    output=args.output_dir.resolve()
    if (output/'diagnosis_runs.csv').exists(): parser.error('Use a fresh output directory.')
    (output/'runs').mkdir(parents=True,exist_ok=True)
    p=DTLZ9Problem(5,50)
    starts=optimal_seeds(p)
    projected=np.vstack([p.project_feasible(x) for x in starts])
    endpoint_distance=float(np.linalg.norm(projected[0]-projected[-1]))
    rows=[]
    for step in [0.01,1e-11]:
        for distance in [1e-8,1e-16]:
            for seed in range(args.seeds):
                cfg=NonlinearORCAConfig(num_groups=2,grouping_method='average_linkage',
                    active_constraint_tolerance=1e-10,num_points_per_seed=3,
                    include_seed_points=True,random_seed=seed,step_size=step,
                    min_point_distance=distance,max_projection_failures=20)
                pts=generate_fixed_points(p,cfg,seeds=starts)
                blocks=sampled_blocks(p,pts)
                matrices,meta=matrices_from_sampled_blocks(blocks,active_tolerance=1e-10)
                f=np.vstack([p.objective_values(x) for x in pts])
                truth=np.array([1,1,1,1,2])
                for method in ['ORCA-current','ORCA-joint']:
                    mat=matrices[method]
                    rows.append(dict(step_size=step,min_point_distance=distance,seed=seed,method=method,
                        selected_points=len(pts),all_active_fraction=float(np.mean(np.all(blocks.constraint_values>=-1e-10,axis=1))),
                        any_active_fraction=float(np.mean(np.any(blocks.constraint_values>=-1e-10,axis=1))),
                        max_violation=max(0.0,float(blocks.constraint_values.max())),
                        f_last_min=float(f[:,-1].min()),f_last_max=float(f[:,-1].max()),
                        permuted_fixed_k_recovery=permutation_recovery(mat,truth,seed),
                        **edge_summary(mat),**grouping_metrics(truth,fixed_k_groups(mat,2))))
                np.savez_compressed(output/'runs'/f'step{step:g}_distance{distance:g}_seed{seed:02d}.npz',
                    points=pts,objectives=f,constraints=blocks.constraint_values,
                    matrix_current=matrices['ORCA-current'],matrix_joint=matrices['ORCA-joint'])
    frame=pd.DataFrame(rows)
    frame.to_csv(output/'diagnosis_runs.csv',index=False)
    summary=frame.groupby(['step_size','min_point_distance','method'],as_index=False).agg(
        runs=('seed','size'),points_mean=('selected_points','mean'),points_min=('selected_points','min'),
        points_max=('selected_points','max'),all_active_fraction=('all_active_fraction','mean'),
        any_active_fraction=('any_active_fraction','mean'),recovery=('exact_recovery','mean'),
        permuted_recovery=('permuted_fixed_k_recovery','mean'),within=('within_mean','mean'),
        between=('between_mean','mean'),separation=('separation_margin','mean'))
    summary.to_csv(output/'diagnosis_summary.csv',index=False)
    (output/'metadata.json').write_text(json.dumps(dict(
        date='2026-10-05',M=5,n=50,seeds=args.seeds,endpoint_decision_distance=endpoint_distance,
        positive_endpoint_coordinate=(1/10)**10,zero_replacement_coordinate=1e-30,
        zero_replacement_objective=10*(1e-30)**0.1,
        note='Diagnostic parameter changes only; original core and problem adapter unchanged. Small-step value is a case-specific diagnostic, not a general default.',
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()),indent=2)+'\n')
    print(summary.to_string(index=False))
    print('Distance between repaired endpoints:',endpoint_distance)


if __name__=='__main__': main()
