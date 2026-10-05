"""Create a compact report and exact matrix visualization from saved experiments."""
from __future__ import annotations
import base64
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[2]
RESULTS=ROOT/'results/dtlz9_single_objective_starts'


def main():
    sampling=pd.read_csv(RESULTS/'sampling_summary.csv')
    grouping=pd.read_csv(RESULTS/'grouping_summary.csv')
    diagnosis=pd.read_csv(RESULTS/'diagnosis/diagnosis_summary.csv')
    key=['M','n','block_size','regime']
    main_rows=grouping[(grouping.regime=='single_objective_optima') &
                       (grouping.method=='ORCA-current') & (grouping.grouping_mode=='fixed_K2')]
    table=main_rows.merge(sampling,on=key)
    table=table[['M','n','selected_mean','target_points','exact_recovery_rate','permuted_fixed_k_recovery','between_signed_mean']]
    diag=diagnosis[diagnosis.method=='ORCA-current'][['step_size','min_point_distance','points_mean','all_active_fraction','permuted_recovery']]
    matrices=[]
    for stem in ['step0.01_distance1e-08','step1e-11_distance1e-16']:
        matrices.append(np.mean([np.load(RESULTS/f'diagnosis/runs/{stem}_seed{i:02d}.npz')['matrix_current'] for i in range(10)],axis=0))
    fig,axes=plt.subplots(1,2,figsize=(8,3.7),layout='constrained')
    for ax,signed,title in zip(axes,matrices,['Original settings: 1 point','Scale-adjusted: 15.1 points (mean)']):
        adjacency=(1+signed)/2
        im=ax.imshow(adjacency,vmin=0,vmax=1,cmap='RdBu')
        ax.set(xticks=range(5),yticks=range(5),xticklabels=['f1','f2','f3','f4','f5'],yticklabels=['f1','f2','f3','f4','f5'],title=title)
        for i in range(5):
            for j in range(5):
                ax.text(j,i,f'{adjacency[i,j]:.2f}',ha='center',va='center',fontsize=9,color='white' if adjacency[i,j]>.85 else '#17212b')
    fig.colorbar(im,ax=axes,label='ORCA graph score A (0.5 = neutral)',shrink=.8)
    fig.savefig(RESULTS/'current_matrix_comparison.png',dpi=180)
    plt.close(fig)
    img=base64.b64encode((RESULTS/'current_matrix_comparison.png').read_bytes()).decode()
    table.columns=['M','n','平均保留点数','去重前目标点数','原顺序固定K正确率','打乱顺序正确率','跨组平均有符号关系']
    diag.columns=['步长','去重距离阈值','平均保留点数','所有结构约束均活跃的点比例','打乱顺序正确率']
    main_html=table.to_html(index=False,border=0,float_format=lambda x:f'{x:.4g}')
    diag_html=diag.to_html(index=False,border=0,float_format=lambda x:f'{x:.4g}')
    html=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>DTLZ9：从单目标最优解出发的 ORCA 测试</title>
<style>body{{font:16px/1.7 system-ui,"Noto Sans CJK SC",sans-serif;max-width:1080px;margin:40px auto;padding:0 24px;color:#183044}}h1{{font-size:30px}}h2{{font-size:21px;margin-top:32px}}table{{border-collapse:collapse;width:100%;font-size:14px}}th,td{{padding:9px 10px;border-bottom:1px solid #dce4e9;text-align:right}}th{{background:#eef3f6}}.lead{{padding:18px;background:#edf4f7;border-left:4px solid #23738a}}code{{background:#f0f3f5;padding:2px 5px}}img{{max-width:100%}}small{{color:#587080}}li{{margin:6px 0}}</style>
<h1>DTLZ9：从单目标最优解出发的 ORCA 测试</h1><small>2026-10-05 · 180 次主对照 + 40 次参数诊断 · 原核心算法未修改</small>
<p class="lead"><b>结论：</b>可以输入每个目标的精确最优点，但标准 n=10M 下，当前数值设置导致采样退化为一个点；固定 K=2 的表面正确率来自并列权重顺序。对 M=5、n=50 同时缩小步长和去重阈值后，10 个随机种子及其目标顺序扰动均得到正确分组。</p>
<h2>1. 起点如何构造</h2><p>标准 DTLZ9 每个目标为一个变量块的 Σx<sup>0.1</sup>，结构约束为 f<sub>i</sub>²+f<sub>M</sub>²≥1。每个目标都非负；本次为其构造达到目标值 0 的可行点，因此单目标全局最优性可直接验证。</p>
<p>采用 Pareto 有效的解作为非唯一最优解的统一选择：前 M−1 个目标使用目标向量 (0,…,0,1)，最后一个使用 (1,…,1,0)。块内采用相等变量分配。保留 M 个起点条目，实际只有两个不同端点。本结论限定于这种最优解代表的选择，不覆盖所有非唯一最优解。</p>
<p><b>精确端点的零变量使导数 0.1x<sup>−0.9</sup>发散。</b>原有 problem adapter 在采样前把变量下限抬到 10<sup>−30</sup>。因此实际梯度在修复后的近端点求值；本实验没有声称直接在零点计算了有限梯度。每块 10 个变量时，原本为 0 的目标被抬到约 0.01，结构约束也随之从活跃变为松弛。</p>
<h2>2. 主对照设置与结果</h2><p>M=3、5、10；n=M 和 n=10M；每种设置 10 个随机种子。三种初始化为单目标最优端点、内侧 Pareto 点和部分活跃可行点。三组均输入 M 个起点，每个起点尝试增加 3 个点，步长 0.01，去重阈值 10<sup>−8</sup>，活跃约束容差 10<sup>−10</sup>。完整 180 次采样均可行，最大结构约束违反小于 5×10<sup>−16</sup>。</p>
<p>下表只列单目标最优端点初始化的 ORCA-current；目标分组为 {{f₁,…,f<sub>M−1</sub>}} 与 {{f<sub>M</sub>}}，预先指定 K=2。每次结果另做 30 次随机目标顺序扰动后重新分组。</p>{main_html}
<p>n=M 的 30 次运行均提供可区分的跨组负关系。n=10M 的 30 次运行均只保留一个点，所有非对角图权重均为 0.5；打乱顺序后成功率接近随机挑中最后一个目标成为单独一组的 1/M。不能把原顺序下的 100% 记作成功识别。</p>
<h2>3. M=5、n=50 的尺度诊断</h2><p>块内实现目标值 1 的变量值仅为 (1/10)<sup>10</sup>=10<sup>−10</sup>，两个修复后端点的距离约为 7.071×10<sup>−10</sup>，小于原去重阈值 10<sup>−8</sup>。执行了步长×去重阈值的 2×2 对照，每格 10 个随机种子。</p>{diag_html}
<img src="data:image/png;base64,{img}" alt="原参数与尺度调整后的 ORCA 图权重比较"><p>同时使用步长 10<sup>−11</sup> 与去重阈值 10<sup>−16</sup>时，保留 15–16 个点（均值 15.1），ORCA-current 的平均跨组有符号关系约 −0.3367，10/10 次固定 K=2 分组正确，300/300 次顺序扰动也正确。仅改变其中一个参数没有恢复可靠识别。</p>
<h2>4. 解释范围</h2><ul><li>这是现有采样实现、可行性修复和参数尺度下的结果，不证明“从单目标最优解开始”这一策略本身无效。</li><li>参数修正只在 M=5、n=50 测试；10 个随机种子不构成全面稳健性验证，不建议直接把小步长改成所有模型的默认值。</li><li>ORCA-current 与 ORCA-joint 的结果分开保存。本次“正确分组”主要指已知 K=2。ORCA-current 的正边连通分量诊断仍把目标分成 M 组，未证明可以自动发现正确 K。</li><li>原 DTLZ9 adapter 只把 M−1 个结构约束提供给 ORCA，变量边界在修复中处理；修复也不是精确欧氏投影。实验保留了这些约定。</li><li>内侧 Pareto 起点的对照在六种规模下固定 K=2 及顺序扰动均正确，但 n=10M 也有保留点数不足，说明尺度问题影响范围不限于端点。</li></ul>
<h2>5. 文件与重跑</h2><p>主结果见 sampling_runs.csv、grouping_runs.csv 及各自 summary；每次实际点、目标、约束、起点与关系矩阵保存在 runs/*.npz。诊断结果在 diagnosis/。run_metadata.json 记录了设置、端点检查和源文件哈希。</p>
<pre>python -m experiments.dtlz9.run_single_objective_starts --output-dir outputs/dtlz9_optima_new
python -m experiments.dtlz9.diagnose_endpoint_sampling --output-dir outputs/dtlz9_diagnosis_new</pre>
<p>命令从更新后的 ORCA 仓库根目录运行，先安装 requirements.txt。使用新的输出目录可避免覆盖已保存实验。</p></html>'''
    (RESULTS/'DTLZ9_SingleObjective_Start_Report.html').write_text(html)
    (RESULTS/'README.md').write_text('''# DTLZ9: sampling from single-objective optima

Date: 2026-10-05. The saved experiment uses 180 sampling runs (three start
families, six problem sizes and ten random seeds) plus 40 factorial diagnostic
runs. The ORCA core and the original DTLZ9 adapter are unchanged.

Read `DTLZ9_SingleObjective_Start_Report.html` for the complete Chinese report.

The original settings collapse to one selected point for n=10M. Every off-
diagonal ORCA-current graph score is neutral (0.5); apparent fixed-K recovery
in original objective order is a tie-breaking artifact. In the M=5,n=50
diagnostic, reducing both step size to 1e-11 and duplicate distance to 1e-16
restores 15–16 points and correct fixed-K=2 groups in all ten random seeds
and all 300 objective-order permutations. This is a case-specific diagnostic,
not a validated general default.

Analytical seed choices achieve each assigned objective's nonnegative lower
bound of zero. Pareto-efficient tie breaking produces two distinct endpoints
across M entries. Zero-coordinate derivatives are singular; the unchanged
repair clips to 1e-30 before differentiation, so the evaluated starting points
are perturbed optima. For block size ten the zero objective becomes 0.01.

The two controls use M seed entries as well: evenly spaced interior Pareto
points and partially active feasible points. The latter differs from the
older diagnostic's two-start budget. Selected-point counts vary because of
deduplication and stopping. All six configurations use step 0.01, three extra
points per seed, active tolerance 1e-10 and duplicate distance 1e-8.

Files:
- `sampling_runs.csv`, `sampling_summary.csv`: feasibility and coverage.
- `grouping_runs.csv`, `grouping_summary.csv`: Raw-Cosine, ORCA-current and
  ORCA-joint, fixed K=2 and positive-component grouping kept separate.
- `runs/*.npz`: actual selected decisions, objectives, constraints, exact and
  repaired seeds, graph matrices and ORCA-current total weights.
- `diagnosis/`: the 2x2 step-size/deduplication-distance experiment.
- `run_metadata.json`: parameters, analytic endpoint checks and source hashes.
- `current_matrix_comparison.png`: mean current-method graph matrices for the
  original and scaled M=5,n=50 settings.

Reproduce from the ORCA repository root after installing `requirements.txt`:

```bash
python -m experiments.dtlz9.run_single_objective_starts --output-dir outputs/dtlz9_optima_new
python -m experiments.dtlz9.diagnose_endpoint_sampling --output-dir outputs/dtlz9_diagnosis_new
```

The unknown-K diagnostic does not establish automatic group-count recovery
for ORCA-current. Original Julia/DTLZ9 projections, variants and historical
manuscript results were not replaced by these new experiments.
''')
    print('Wrote report and matrix figure to',RESULTS)


if __name__=='__main__': main()
