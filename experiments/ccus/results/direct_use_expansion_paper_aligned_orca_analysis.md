# Material Passport

- Origin Skill: academic-research-suite / experiment-agent；pdf；nature-figure
- Origin Mode: validate
- Origin Date: 2026-08-10
- Verification Status: VERIFIED FOR THE REPORTED LOCAL ORCA WORKFLOW
- Scenario: `direct_use_expansion`
- ORCA implementation: `/Users/hongxuan/Documents/ORCA_python`，Git commit
  `a0168f3c0d32c7120de5a863672c5acd1af3855b`，运行时工作树有未提交修改
- Fixed-point projection: Julia / JuMP / Ipopt
- Analysis and plotting: Python / local ORCA utilities / matplotlib
- Global-optimality status: not claimed

# `direct_use_expansion` 的完整非线性 ORCA 结果

## 1. 回答前一次结果不一致的问题

旧值 `0.5031, 0.5342, 0.7674` 确实调用了本地 ORCA package 中的相互作用函数，
但输入只有 TAC、总排放、ISI 三个单目标最优端点，并且采用线性局部算子。因此它是
“三端点适配结果”，不是 Wang–Allman (2025) 完整非线性 ORCA 工作流的最终结果。

本次重新计算包含论文算法的主要步骤：

1. 分别求三个单目标种子点；
2. 按论文式 (3) 从约束切空间中的目标下降方向生成随机 conic direction；
3. 以步长 `0.03` 前进一步，再按式 (4) 用 Ipopt 投影回可行域；
4. 在每个固定点计算非线性 modified-gradient objective interactions；
5. 使用本地 package 的权重函数（`alpha=0.9`, `beta=100`）和论文式 (5) 聚合；
6. 用 Leiden、`K=2` 分组，并用 average linkage 作独立审计。

CCUS 模型有显式等式约束，而原 package 的标准接口只直接处理不等式。本次主分析把
等式作为单位权重的切空间投影；同时给出“不处理等式、只处理不等式”的审计版本，
用来判断这个扩展是否改变结论。

## 2. 主实验与可行性

- 20 个随机重复，随机种子 `20250810`–`20250829`；
- 每个重复包含 3 个单目标种子，每个种子再生成 40 个固定点；
- 共 `20 × 3 × 41 = 2460` 个分析点，另有 2400 次投影；
- 投影失败 `0`，重复点拒绝 `0`；
- 最大等式残差 `4.78e-16`；
- 最大不等式违反 `9.98e-9`，低于 `1e-6` 接受阈值。

这些检查说明进入 ORCA 聚合的点在数值容差内可行。它们是由局部 Ipopt 投影得到的，
不是非凸模型全局最优性的证明。

## 3. 完整 ORCA 强度

下表的 signed correlation 越大，两个目标的局部方向越一致；package 输出的 adjacency
为 `(signed + 1)/2`。均值和标准差来自 20 个随机重复。

| 目标对 | Signed correlation，mean ± SD | 范围 | Adjacency mean | 排名 |
|---|---:|---:|---:|---:|
| 总排放 + ISI | **0.766282 ± 0.000019** | 0.766246–0.766319 | **0.883141** | 1 |
| TAC + 总排放 | **0.556256 ± 0.007978** | 0.539187–0.568456 | 0.778128 | 2 |
| TAC + ISI | **0.511422 ± 0.013878** | 0.481199–0.533612 | 0.755711 | 3 |

20/20 个重复都得到相同排名。Leiden 与 average linkage 在所有重复中也都得到相同
分组：`{TAC}` 与 `{总排放, ISI}`。

## 4. 为什么与旧值不同

| 目标对 | 三端点 signed | 完整 ORCA signed mean | 变化 | 端点排名 → 完整排名 |
|---|---:|---:|---:|---:|
| TAC + 总排放 | 0.503062 | 0.556256 | +0.053194 | 3 → 2 |
| TAC + ISI | 0.534153 | 0.511422 | −0.022731 | 2 → 3 |
| 总排放 + ISI | 0.767359 | 0.766282 | −0.001077 | 1 → 1 |

差异不是 package 身份变化，而是取样与算子变化。旧计算只评价三个极端点；新计算在
完整可行域附近的 2460 个固定点上应用 nonlinear modified-gradient operator。
`总排放–ISI` 对两种口径都极稳定；`TAC–总排放` 和 `TAC–ISI` 很接近，所以端点抽样
不足时会交换第二、第三名。

## 5. 稳健性分析

| 步长 / 每个种子新增点数 | TAC–排放 | TAC–ISI | 排放–ISI | 重复数 |
|---|---:|---:|---:|---:|
| 0.01 / 40 | 0.544087 | 0.492753 | **0.766360** | 5 |
| 0.03 / 40（主实验） | 0.556256 | 0.511422 | **0.766282** | 20 |
| 0.05 / 40 | 0.554751 | 0.511980 | **0.766356** | 5 |
| 0.03 / 20 | 0.565875 | 0.529147 | **0.766264** | 5 |
| 0.03 / 80 | 0.555522 | 0.509701 | **0.766287** | 5 |

所有 40 个主实验与敏感性重复都保持同一排序和同一二群分组。仅处理不等式的审计
也保持同一排序：排放–ISI `0.764401`，TAC–排放 `0.556526`，TAC–ISI `0.485974`。
因此，“排放–ISI 最强、TAC 单独成组”不依赖本次等式切空间扩展。

## 6. 与 Pareto information loss 的关系

将完整 ORCA 结果接入 Russell–Allman Section 4.1 的 Pareto information-loss 分析后：

| 目标 grouping | ORCA adjacency | Equal-bin mean loss | ORCA rank | Loss rank |
|---|---:|---:|---:|---:|
| TAC + 总排放 | 0.778128 | **0.308955** | 2 | 1 |
| 总排放 + ISI | **0.883141** | 0.438194 | 1 | 2 |
| TAC + ISI | 0.755711 | 0.907279 | 3 | 3 |

完整 ORCA 与负 information loss 的描述性 Spearman correlation 为 `+0.5`，Kendall
tau 为 `+0.333`。由于只有三个目标对，这些数值不能作显著性推断。最关键的是，ORCA
第一名仍不是 information loss 最小的一组：若目标降维的唯一判据是保留全局 Pareto
几何，`TAC + 总排放` 更合适；若关心活跃约束附近的局部方向耦合，ORCA 会选择
`总排放 + ISI`。

这不是算法矛盾。ORCA 测量固定点附近的局部 modified-gradient 关系，而 Section 4.1
information loss 衡量跨 sink、预处理比例和流程结构切换的全局 frontier 冗余。

## 7. 解释边界与 fallacy scan

- Correlation ≠ causation：ORCA strength 不证明工艺变量间的物理因果关系。
- Local-to-global fallacy：局部梯度相似不能自动等价为全局 Pareto 可合并性；本结果
  正好显示两者第一名不同。
- Garden of forking paths：主参数在计算前固定，步长与点数变化只作敏感性分析；不从
  多组结果中挑选单个有利重复。
- Look-elsewhere effect：报告全部 20 个主重复和全部 20 个敏感性重复，不使用多重尝试
  中的极值作为结论。
- Simpson、Berkson、collider、base-rate neglect、regression-to-mean、survivorship、
  reverse causality：在当前确定性优化比较中未发现直接适用证据，但不能据此排除真实
  工业数据中的这些偏差。

## 8. 可复现性限制

本次明确使用的是用户本地 ORCA source tree，而不是另一个同名第三方包。记录了当前
Git commit，但该工作树存在未提交修改，因此结果还不是完全 version-pinned artifact。
若用于论文归档，应先提交或导出这些修改，再锁定 Python/Julia environment。Ipopt
适合当前平滑连续 NLP 的可行投影，但 `LOCALLY_SOLVED` 只表示局部收敛；求解失败也
不能自动解释为 infeasible。
