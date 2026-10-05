# `direct_use_expansion` 的三目标 Pareto frontier 与 ORCA 验证

## Material Passport

- Origin Skill: academic-research-suite / experiment-agent；nature-figure
- Origin Mode: validate
- Origin Date: 2026-08-10
- Verification Status: VERIFIED AS A DETERMINISTIC LOCAL NLP APPROXIMATION
- Scenario: `direct_use_expansion`
- Optimization: augmented epsilon-constraint + Ipopt deterministic multi-start
- Plotting: Python / matplotlib only；SVG、PDF、TIFF 和 PNG 均已导出
- Global-optimality status: **未证明**

> **更新（2026-08-10）：** 本报告第 3 节的等宽分箱/最近邻结果现在仅作为早期近似
> 敏感性分析。采用论文 Eqs. 10–11 的实际保留目标值和条件 Pareto 切片重新计算后，
> 三组排序与 ORCA 完全一致。主结论和可复现文件见
> `direct_use_expansion_exact_information_loss_analysis.md`；下文原分析保留用于说明
> 方法口径为何会改变结果。

## 1. 结论

在 `direct_use_expansion` 情景中，ORCA 与 Pareto information-loss 验证**不相符**：

- 按 Wang–Allman (2025) 完整非线性固定点流程、20 个随机重复计算，ORCA 最强的
  目标对是 **总排放 + ISI**，mean strength = `0.8831`；
- Pareto frontier 上 information loss 最小的目标对却是
  **TAC + 总排放**；
- 三种候选点来源、两类局部切片方法和全部 14 个分辨率设定都得到同一排序：
  `TAC + 总排放` < `总排放 + ISI` < `TAC + ISI`。

因此，这个情景不支持“ORCA strength 越高，Pareto information loss 就越低”的
直接定量解释。更合理的解释是：ORCA 描述给定最优点附近、受活跃约束限制的
**局部方向关系**；Pareto information loss 测量跨越 Urea、Saline Storage 和
Greenhouse 三种系统结构的**全局 trade-off 冗余**。直接利用路线引起的拓扑切换
使两者出现分歧。

## 2. Pareto frontier 的严格计算口径

没有使用遗传算法、粒子群或随机 population heuristic。计算采用增广
epsilon-constraint：分别选择一个目标作为主目标，把另外两个目标设为显式上界，
并加入 `1e-3` 的归一化增广项以排除弱有效解。

- 主网格：TAC 为主目标，排放和 ISI 使用 `31 × 31` ε 网格；
- 边界审计：总排放和 ISI 分别作为主目标，各使用 `7 × 7` 正交网格；
- 三条降维 frontier：每种二目标 grouping 使用 61 个 ε 水平；
- 每个目标单元使用 3 个确定性起点；均失败时继续尝试其余 anchor；
- 接受阈值：原等式、原不等式和 ε 约束残差均不超过 `1e-6`。

共接受 834 个三目标网格解和 181 个降维解。合并后有 1015 个候选点；在按
`1e-8` 去重并按三个最小化目标剔除 dominated points 后，得到 **225 个唯一
非支配点**。

| Objective | Pareto minimum | Pareto maximum |
|---|---:|---:|
| TAC | 2.541 million | 35.149 million |
| Total emissions | 32.500 | 125.451 |
| ISI | 4.169 | 20.357 |

所有写入候选点的最大原等式残差为 `7.96e-16`，最大原不等式违反为
`9.99e-9`，最大 ε 违反为 `1.30e-8`。

### 网格收敛

把 31 × 31 主网格与其中嵌入的 16 × 16 子网格比较，细网格到粗网格 frontier
的归一化最近距离为：均值 `0.0175`、95 分位 `0.0804`、最大值 `0.1099`。
95 分位低于预先采用的 `0.10` 判据，说明当前 ranking 已达到可接受的网格稳定性；
它不是全局最优证明。

### Ipopt 局部失败审计

ε 可行域随约束收紧必须嵌套。首轮日志中发现 36 个二维网格单元出现“更严格单元
可行、较宽松单元却未收敛”的现象；三条降维曲线最初有 3 个同类单元。每个此类
单元都能由更严格单元的解提供显式可行见证，因此不能解释为模型不可行。

定向热启动重新优化找回了 1 个降维单元；其余 36 个二维单元和 2 个降维单元仍未
满足 Ipopt 的终止状态、残差和 ε 约束三重接受标准。它们保留在日志中，没有被静默
改写为可行或不可行。由于见证点本身已存在于候选集中，并且候选来源剔除分析不改变
三个 grouping 的排序，这一局部求解缺口不改变本报告的定性结论，但阻止我们声称
全局完备 frontier。

## 3. 按 Russell–Allman Section 4.1 的 information loss

三个目标先按本次计算 frontier 的最小值和最大值归一化：

`f̄_i = (f_i - f_i,min) / (f_i,max - f_i,min)`。

对于每个待检验的二目标 grouping，沿未被 grouping 的第三个目标切片，并在每个
有效切片内计算两个 grouped objectives 的归一化 range 之和，再对切片平均。为避免
连续 frontier 的离散化选择主导结论，使用了两套定义：

1. 8、10、12、15、18、20、25、30 个等宽切片；
2. 8、12、16、20、24、30 个最近邻局部切片。

同时计算降维 frontier 到完整 frontier 的归一化三维最近距离（IGD-like coverage）。

| Grouped objectives | ORCA strength | Mean loss: equal bins | Mean loss: neighbors | Mean distance to reduced frontier |
|---|---:|---:|---:|---:|
| TAC + total emissions | 0.7781 | **0.3090** | **0.3580** | **0.2276** |
| Total emissions + ISI | **0.8831** | 0.4382 | 0.5006 | 0.5201 |
| TAC + ISI | 0.7557 | 0.9073 | 0.9320 | 0.6979 |

全部 14 个分辨率下，最低 information loss 都是 **TAC + total emissions**。
完整 ORCA 排名为 `排放 + ISI` > `TAC + 排放` > `TAC + ISI`；旧的三端点近似则把
后两者次序颠倒。完整 ORCA 与负 information loss 排名的描述性 Spearman
correlation 为 `+0.5`，Kendall tau 为 `+0.333`；只有三个目标对，因此不作显著性
推断，也不能把这种正相关解释成两个指标等价。

### 候选来源敏感性

| Candidate pool | Frontier points | Lowest-loss pair | Second | Third |
|---|---:|---|---|---|
| TAC-primary grid only | 161 | TAC + emissions | Emissions + ISI | TAC + ISI |
| All three full grids | 185 | TAC + emissions | Emissions + ISI | TAC + ISI |
| Full grids + reduced fronts | 225 | TAC + emissions | Emissions + ISI | TAC + ISI |

这排除了“结论仅由某一组正交审计点或降维点造成”的解释。

## 4. Frontier 上的 CCUS 系统变化

225 个非支配点跨越三种主要 sink，说明本情景确实产生了丰富的系统选择：

| Main sink | Frontier points | Main objective region | Pretreated fraction |
|---|---:|---|---:|
| Urea | 45 | 低 TAC、较高 ISI | 58.7%–61.8% |
| Saline Storage | 73 | 低排放过渡区 | 11.3%–59.8% |
| Greenhouse | 107 | 低 ISI、直接利用区 | 约 0%–65.1%；中位数 30.2% |

Urea 分支具有最低成本；Saline Storage 分支把排放降至约 32.50；Greenhouse
分支把 ISI 降至约 4.17，并在极端点达到近 100% 直接利用。frontier 不是一个固定
流程的小扰动曲面，而是包含明显的 sink 和预处理结构切换。这解释了为什么完整非线性
ORCA 仍给出最强的排放–ISI 一致性，而全局 Pareto 几何却认为 TAC–排放更适合合并。

## 5. Ipopt 能否用于这个 optimization？

**可以，但结论必须限定为局部 Pareto approximation。** 当前模型的 82 个变量均为
连续变量，约束和目标经过平滑化，因而适合 Ipopt。Ipopt 也明显快于对这个网格使用
通用全局空间分支方法。

但是 Ipopt 是局部 NLP solver：

- `LOCALLY_SOLVED` 不是全局最优证书；
- 求解失败不能自动解释为模型 infeasible；
- 多起点、ε 可行域嵌套审计、残差审计和网格收敛检查是必要的；
- 若论文需要“严格全局 Pareto frontier”而不是“严格 scalarization 下的局部
  Pareto approximation”，必须进一步使用能给出全局上下界的确定性全局非线性
  求解方法，并报告 optimality gap。

所以，本次方法在“没有使用 heuristic Pareto search”这一点上是严格的；在“每个
非凸 NLP 子问题都被全局求解”这一点上并不严格，也没有如此宣称。

## 6. Fallacy scan 与解释边界

- Correlation ≠ causation：ORCA 和 Pareto 几何都不能证明现实工艺中的因果关系。
- Garden of forking paths：本情景是机制导向构造的探索参数组；应与基线和其他候选
  情景一起报告，不能只展示分歧最大的结果。
- Look-elsewhere effect：14 个切片设置属于稳健性分析，不应挑选其中一个最有利数值。
- Local-to-global fallacy：局部梯度相关不能自动推断跨拓扑 frontier 的全局冗余；
  本情景正好展示了这一点。
- 其余 Simpson、Berkson、collider、base-rate neglect、regression-to-mean、
  survivorship 和 reverse-causality 风险，在本确定性模型比较中未发现直接适用证据。
