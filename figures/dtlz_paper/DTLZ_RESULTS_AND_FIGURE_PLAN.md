# DTLZ5/DTLZ6 结果汇总与论文图方案

## 1. 可以写进 Results & Discussion 的主线

当前结果最适合形成三层证据链：

1. **结构恢复：** nonlinear ORCA 从目标间的 affinity 中恢复已知的退化目标分组。
2. **几何有效性：** 对 DTLZ5(2,3)，按恢复分组进行 L2 聚合后，所得解集贴近解析真实 Pareto 前沿。
3. **下游影响：** DTLZ5 的目标降维同时改善解集质量并降低自然预算下的优化时间；DTLZ6 则表现出明显的速度–质量权衡，说明其主要瓶颈由“分组识别”转向了 distance variables 的收敛。

最稳妥的论文表述不是“reduction 在所有 DTLZ 问题上都更优”，而是：

> The nonlinear grouping accurately recovered the prescribed objective structure across the tested standard DTLZ5 and DTLZ6 instances. This structural recovery translated into clear downstream gains on DTLZ5, whereas DTLZ6 exposed a convergence-limited regime in which objective reduction alone was insufficient to preserve full-space front coverage.

## 2. 当前全部测试结果

### 2.1 标准 DTLZ5/DTLZ6：结构恢复

数据源：`orca_dtlz5_dtlz6_gradient_grouping_raw.csv`、`orca_dtlz5_dtlz6_gradient_grouping_summary.csv`。

- 测试了 DTLZ5 和 DTLZ6 的 6 个设置：`(I=5,M=16,k=10)`、`(I=5,M=20,k=100)`、`(I=5,M=20,k=500)`；每个设置 5 个 seeds；ORCA affinity 与 gradient-cosine 两种方法。
- 两种方法在全部 60 个“case × method × seed”组合中均达到 `ARI = 1`、exact match = 100%。
- ORCA 分组时间约为 0.037–0.197 s，长 tail 时成本上升。
- 这一组结果能证明算法稳定恢复**已知结构**，但不能证明 ORCA 比 gradient-cosine 更准确，因为该 baseline 在这些正控制问题上同样达到满分。
- 当前 average-linkage 使用预先给定的 `K=I=5`；因此正文应写成 “fixed-K objective-affinity partition”，不能写成完全自动发现 community 数目。

### 2.2 DTLZ5(5,16)：下游优化

数据源：`orca_dtlz5_5_16_nsga3_raw.csv`、`orca_dtlz5_5_16_nsga3_summary.csv`；5 seeds。

| 方法 | evaluations | approximate HV | empirical IGD | optimizer time |
|---|---:|---:|---:|---:|
| Full NSGA-III | 13,600 | 2.640 | 0.410 | 14.84 s |
| Reduced, natural budget | 3,500 | 3.689 | 0.174 | 4.95 s |
| Reduced, approximately equal evaluations | 约 13,600 | 3.891 | 0.166 | 16.18 s |

在这个协议内，natural-budget reduction 约快 3 倍，同时 HV 更高、empirical IGD 更低；equal-evaluation reduction 的解集质量进一步改善，但时间优势消失。后续 mean/max 聚合的 3-seed 复现实验方向一致：full 的 HV/IGD 为 2.784/0.434，equal-evaluation mean/max 约为 3.929/0.119。

### 2.3 DTLZ5(2,3)：解析真实前沿验证

新数据源：`dtlz3_pareto_solutions.csv`、`dtlz3_pareto_metrics.csv`；5 paired seeds，约等 evaluations（5,600 vs 5,616）。

已知分组为 \(\{f_1,f_2\},\{f_3\}\)。采用

\[
z_1=\sqrt{f_1^2+f_2^2},\qquad z_2=f_3
\]

后，解析前沿恰为单位四分之一圆 \(z=(\cos\theta,\sin\theta)\)。并且

\[
\|z\|_2-1=g(x),
\]

因此图中的径向残差有直接的优化含义，而不是仅用于视觉展示。

| 指标（mean ± SD） | Full NSGA-III | ORCA-reduced L2 | reduced/full |
|---|---:|---:|---:|
| GD to analytic front | 0.002621 ± 0.001564 | 0.000544 ± 0.000725 | 0.21 |
| IGD from analytic front | 0.075511 ± 0.002587 | 0.017846 ± 0.000140 | 0.24 |
| Mean of run-wise median radial gap | 0.000442 | 0.0000375 | 0.085 |

当前 5-seed 结果适合作为清晰的几何验证图；正式投稿前建议按第 6 节协议扩展到 30 paired seeds。

### 2.4 DTLZ6(5,16)：速度–质量权衡

数据源：`orca_dtlz6_m16_l2_aggregation_raw.csv`、`orca_dtlz6_m16_l2_aggregation_summary.csv`；3 seeds。

- Full 的 approximate HV/empirical IGD 为 4.139/0.285；equal-evaluation L2 reduction 为 4.519/0.482。
- 两个指标给出相反排序：HV 上升，但 IGD 明显变差。因此不能只引用 HV 宣称 reduction 整体更优。
- L2 reduction 将 median \(g\) 从 7.686 降到 3.715，但仍远离真实前沿要求的 \(g=0\)。
- 这表明该设置中分组恢复不是主要瓶颈；distance variables 的收敛和全空间 front coverage 才是限制因素。

### 2.5 DTLZ6(5,20)：高维与长 tail 的结果

数据源：`orca_vs_nsga_analytic_active_summary.csv`、`orca_equal_eval_and_weighted_sum_summary.csv`、`orca_epsilon_constraint_pilot_summary.csv`、`orca_downstream_steps_1_5_pilot_summary.csv`。

- Reduced-budget NSGA-III 约快 6.1–6.2 倍，但 IGD 从 full 的约 0.05 恶化到 0.864–0.892；这是明确的速度–质量代价，不是 quality parity。
- Weighted-sum pilot 仅需约 80 evaluations。`k=500` 时 IGD 0.249，接近该文件内 full 的 0.270；但存在平均 2.8–5.0 次 SLSQP failure，且某些 HV 跨 seed 零方差，需要进一步审计。
- Epsilon-constraint pilot 某些设置有较好 HV 或 IGD，但平均约 9–11 次 solver failure，并存在内部约束违反，不宜作为主结果。
- Staged refinement 是更有前景的方向：full 为 HV/IGD 6.547/0.184、22.69 s；reduced→full refinement 为 6.319/0.211、15.22 s。它回收了大部分质量并节省约三分之一时间，但目前只有 3 seeds。

### 2.6 约束、敏感性与规模化测试

- 自定义 constrained-DTLZ6：active-constraint、all-constraints 和 gradient baseline 均为 ARI=1；active 版本比 all-constraints 快 31.0×（M=12）和 34.6×（M=20）。这是实现效率证据，但不是 canonical DTLZ6 结果。
- Selected-point sensitivity 的现有文件之间存在冲突。20 points 是当前最一致的设置；10 或 40 points 的结果不能在重跑前写成稳健结论。
- Runtime-scaling 文件每个 case 只有一个 pilot；部分大-\(k\) case 的快速结果伴随错误分组，只能用于制定正式实验，不宜进入主文统计。

### 2.7 Application 相关的现有结果

- MONAS proxy（5 seeds）：full 的 HV/IGD/time 为 1.126/0.242/0.329 s；reduced-2 为 1.636/0.108/0.278 s。
- NASBench-201 sensitivity：高质量 surrogate 加 40 selected points 时达到 ARI=1、exact match=100%；训练样本仅 250 时 surrogate 的预测 \(R^2\) 为负，ARI 约 0.4、exact match 仅约 5–10%。应用侧的首要瓶颈是 surrogate quality，而不是 community detection 本身。
- 这些结果适合放入 application 小节；与 DTLZ 的 HV/IGD 不应跨文件直接比较。

### 2.8 不应作为标准 DTLZ 主证据的旧结果

- 旧 Julia 文件 `DTLZ(5, 12) delta size of 0.0005 one iteration.csv` 与既有 Figure 3 v5 把 \(g\) 对全部 variables 求和，而标准 DTLZ5 只对 tail/distance variables 求和。
- 旧 Figure 3 v4 还存在把 \(f_4\) 错误并入大组的问题。
- 这些内容最多可标成 legacy implementation audit，不能与当前 Python 标准 DTLZ5/6 合并统计。

## 3. 已制作的 Matplotlib 论文图

### Figure A：目标 affinity、community 与结构恢复

文件：`figure_dtlz_grouping.svg/.pdf/.png/.tiff`。

- **a**：以 objectives 为 nodes、affinity 为 edge width 的 network；颜色表示已恢复的 groups。
- **b**：完整 16×16 affinity heatmap，避免 network 稀疏化隐藏信息。
- **c**：known structure、fixed-K average-linkage 与 Leiden partition 的并列比较。Average-linkage 精确恢复；当前 Leiden target-K 只产生 2 groups（ARI=0.898），这一失败被如实保留。
- **d**：6 个 DTLZ5/6 设置上的 grouping time 和 60/60 exact recovery。
- 当前 edge weight 是算法的 objective affinity，不是 Pearson/Spearman correlation；图和正文应使用 “affinity” 或 “relationship strength”。

建议英文图注：

> **Objective-affinity structure and partition recovery on standard DTLZ problems.** (a) Objective-affinity network for DTLZ5(5,16), with node color denoting the fixed-K average-linkage partition and edge width denoting affinity. For legibility, all edges within the dense block and only the strongest edge incident to each singleton are shown. (b) The unsparsified affinity matrix. (c) Comparison with the prescribed structure. Average linkage exactly recovers the five groups, whereas the present Leiden configuration merges the four singleton objectives. (d) Grouping time across DTLZ5 and DTLZ6 cases. Both ORCA affinity and the gradient-cosine baseline recover the prescribed partition in all 60 runs.

### Figure B：DTLZ5/DTLZ6 下游性能

文件：`figure_dtlz_downstream.svg/.pdf/.png/.tiff`。

- **a**：DTLZ5 的总时间–empirical IGD；展示 natural/equal-evaluation 和 mean/max 聚合。
- **b**：DTLZ6 的总时间–empirical IGD；显示 L2 reduction 的速度–质量权衡。
- **c**：DTLZ6 的 median \(g\)，直接说明“结构恢复正确但未收敛至 \(g=0\)”的机制。

建议英文图注：

> **Downstream consequences of objective reduction differ between DTLZ5 and DTLZ6.** (a) On DTLZ5(5,16), reduced formulations improve empirical IGD and, under the natural reduced population, shorten runtime. (b) On DTLZ6(5,16), reduction yields a speed–quality trade-off rather than uniform dominance. Points show individual seeds and larger symbols show means. (c) Although the prescribed partition is recovered in every run, all tested DTLZ6 variants remain away from the \(g=0\) manifold, identifying convergence of the distance variables as the dominant limitation.

### Figure C：解析真实 Pareto 前沿

文件：`figure_dtlz_true_front.svg/.pdf/.png/.tiff`。

- **a**：DTLZ5(2,3) 的解析四分之一圆、full NSGA-III 与 ORCA-reduced L2 解集；短线显示代表性点到真实前沿的距离。
- **b/c**：5 paired seeds 的 GD 和 IGD，以连线保留配对信息。
- 这张图直接对应已有 objective-reduction 文献中“obtained nondominated set versus true POF”的验证方式。

建议英文图注：

> **Geometric validation against the analytic DTLZ5 Pareto front.** (a) The prescribed correlated block \(\{f_1,f_2\}\) is collapsed by its Euclidean norm, under which the true DTLZ5(2,3) front becomes a unit quarter-circle. Markers show the nondominated solutions from the paired seed whose IGD improvement is closest to the median across seeds; short segments denote distances to the analytic front. (b,c) Paired GD and IGD results across five seeds. At approximately equal evaluation counts, the ORCA-reduced L2 formulation lowers the mean GD and IGD to 0.21 and 0.24 of the full-problem values, respectively.

## 4. 主文中如何取舍

如果 DTLZ5/6 只占 Results & Discussion 的约三分之一，建议：

1. **主文 Figure A：** 结构恢复图。
2. **主文 Figure C：** 解析真实前沿与 paired GD/IGD，这是最直观的正确性证据。
3. **主文 Figure B 或 Supplementary：** 如果版面允许放主文，用它解释 DTLZ5 与 DTLZ6 的差异；否则把完整三面板放 Supplementary，并在正文保留 DTLZ6 的一张 trade-off panel。

不建议只画 network：单独的 community graph 很容易受到 layout 和阈值影响。Network 与完整 affinity heatmap、known partition 对照同时出现，证据更完整。

## 5. 文献依据

- 原始 DTLZ test suite：[Deb et al., *Scalable Multi-Objective Optimization Test Problems*, CEC 2002](https://doi.org/10.1109/CEC.2002.1007032)（[作者版 PDF](https://www.sop.tik.ee.ethz.ch/publicationListFiles/dtlz2002a.pdf)）。其 Fig. 7 用三维解集与 DTLZ5 的预期一维退化曲线进行视觉比较，是本项目“解集–弧线”图最直接的先例；原文没有对应的 DTLZ6 解集图。
- Objective-reduction 文献中已有 obtained nondominated set 与 expected low-dimensional front 的叠加，以及 reduced-objective front 的可视化先例：[Saxena et al., *Objective Reduction in Many-Objective Optimization: Linear and Nonlinear Algorithms*, IEEE TEVC 2013](https://doi.org/10.1109/TEVC.2012.2185847)（[作者版 PDF](https://repository.ias.ac.in/82728/1/8-a.pdf)）。
- 将 objectives 作为 nodes、目标关系作为 edges，并用 community detection 做 objective reduction 的直接先例：[Maulana et al., *Reducing Complexity in Many Objective Optimization Using Community Detection*, CEC 2015](https://doi.org/10.1109/CEC.2015.7257281)。
- 对 DTLZ5/6 高维退化前沿的解释应谨慎；\(M>3\) 时不能不加限定地把整个 \(g=0\) intended manifold 称为完整真实 PF：[Ishibuchi et al., *Pareto Fronts of Many-Objective Degenerate Test Problems*, IEEE TEVC 2016](https://doi.org/10.1109/TEVC.2015.2505784)。因此当前解析真前沿图只使用无歧义的 \(M=3\) case。

## 6. 投稿前建议补做的正式实验

1. DTLZ5(2,3) 与 DTLZ6(2,3)，`k=10`，30 paired seeds。
2. Full 与 reduced 均使用 population=91；full 采用 3D Das–Dennis `H=12`，reduced 采用 2D `H=90`。
3. 500 generations，即每种方法每个 run 45,500 evaluations；保存 1、10、25、50、100、200、500 generations 的 checkpoints。
4. DTLZ6(2,3) 当前默认 ORCA 会得到错误分组 `[1,2,1]`（ARI=-0.5）。在修复 anchor tie-breaking 前，只能把已知 `[1,1,2]` 的实验标为 **known-group/oracle L2 ablation**，不能标成 ORCA-reduced。
5. 以 10,001 个解析点构建固定 reference front；在原始三目标空间计算 GD、IGD、radial gap 和组内 symmetry error。不要用各算法输出的 union 作为 true reference。
6. 以 run/seed 为统计单位，报告 paired effect、bootstrap 95% CI 和多重比较校正；不要把同一 run 内的多个解当成独立重复。
7. 对所有高维实验统一 reference set、归一化、HV reference point 和 Monte Carlo 样本数。当前不同 CSV 的 empirical IGD/HV 只能在同一文件、同一 protocol 内比较。

当前高维文件中的 HV 是参考点 1.1、50,000 samples 的 Monte Carlo approximate HV；IGD 使用 pooled empirical nondominated reference，不是解析真前沿 IGD。正式实验若采用 IGD+，可引用 [Ishibuchi et al., 2015](https://doi.org/10.1007/978-3-319-15892-1_8)，但仍需固定且可信的 reference front。

## 7. 复现

在项目的 Python 环境中运行：

```bash
python figures/dtlz_paper/generate_dtlz_source_data.py
python figures/dtlz_paper/make_dtlz_paper_figures.py
```

绘图脚本仅使用 Matplotlib；同时导出可编辑文字的 SVG、嵌入 TrueType 字体的 PDF、300-dpi PNG 和 600-dpi LZW-compressed TIFF。
