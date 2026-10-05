# Material Passport

- Origin Skill: academic-research-suite / experiment-agent；pdf；nature-figure
- Origin Mode: validate
- Origin Date: 2026-08-10
- Verification Status: **VERIFIED FOR THE FINITE, DETERMINISTIC LOCAL-NLP WORKFLOW**
- Scenario: `direct_use_expansion`
- Information-loss definition: Russell–Allman Eqs. 10–11, evaluated on actual retained-objective values
- Optimization: augmented epsilon-constraint + deterministic Ipopt multi-start
- Primary discretization: 11 retained-objective levels × 21 epsilon levels per grouping
- Plotting: Python / matplotlib；SVG、PDF、TIFF、PNG
- Global-optimality status: **未证明**

# `direct_use_expansion`：论文原式 information loss 重算

## 1. 主要结论

这次重算改变了此前基于等宽分箱/最近邻近似得到的判断。按照论文定义，在完整三目标
Pareto 样本中选取第三个（保留）目标的**实际取值**作为条件值 \(p\)，再求相应条件
Pareto 切片，得到：

| Grouped objectives | Exact conditional loss | Slice SD | ORCA strength | Loss rank | ORCA rank |
|---|---:|---:|---:|---:|---:|
| Total emissions + ISI | **0.2900** | 0.3309 | **0.8831** | **1** | **1** |
| TAC + total emissions | 0.3453 | 0.3787 | 0.7781 | 2 | 2 |
| TAC + ISI | 0.7429 | 0.5725 | 0.7557 | 3 | 3 |

信息损失越小，说明在固定剩余目标后，被合并的两个目标之间可失去的 Pareto 信息越少；
ORCA strength 越大，说明 ORCA 判断它们越适合合并。两者的三个名次完全一致：

`Total emissions + ISI` > `TAC + total emissions` > `TAC + ISI`

因此，**当前论文原式的 Pareto 验证支持 ORCA 的相对排序**。三个样本的描述性
Spearman rank correlation 和 Kendall rank correlation 都为 `1.0`。由于只有三个
目标对，这只是排序一致性，不是统计显著性证据，也不能证明两个指标在数值上等价。

## 2. 为什么与上一版结果不同

上一版在连续 Pareto front 上采用两种替代办法：

1. 把保留目标分成若干等宽区间；
2. 围绕每个点取固定数量的最近邻。

它们把一段不同 \(p\) 值的 Pareto 点混进同一个条件集。尤其在 frontier 点密度很不均匀、
并且 sink/预处理结构发生切换时，区间宽度和邻居数量会改变条件内的目标跨度。旧方法得到
`TAC + total emissions` 最小，只能作为连续离散数据的近似敏感性结果，不能称为论文
Eqs. 10–11 的直接实现。

本次也测试了“只加 \(R=p\) 等式，然后直接优化两个 grouped objectives”的方案，发现
它会产生被 \(R<p\) 点支配的条件可行点，因而不属于论文要求的完整三目标 Pareto 子集
\(J_p\)。该诊断结果已舍弃，没有进入主统计。

## 3. 论文原式的实现

令 \(C\) 为准备合并的两个目标，\(R\) 为保留的第三个目标。先用完整 Pareto 样本的上下界
归一化每个目标：

\[
\bar f_i=(f_i-l_i)/(u_i-l_i).
\]

对每个实际保留目标值 \(p\)，从完整三目标 Pareto 集中取出满足 \(R=p\) 的条件切片
\(J_p\)，然后计算：

\[
B_p=\sum_{i\in C}\left(\max_{j\in J_p}\bar f_{i,j,p}
-\min_{j\in J_p}\bar f_{i,j,p}\right),
\qquad
B=\frac{1}{P}\sum_{p=1}^{P}B_p.
\]

本次 \(P=11\)。这 11 个 \(p\) 不是等宽区间中心，而是从已验证的 225 点完整 Pareto
样本中按分位位置选出的**真实目标值**，因此每个切片至少有一个原始 Pareto 锚点。
对每个 \(p\)，用保留目标上界 \(R\le p\) 和另一个 grouped objective 的 21 个
epsilon 上界求解增广 epsilon-constraint 子问题；只保留 \(R\le p\) 在归一化误差
`1e-6` 内活跃的解。生成点与原始 225 点 frontier 合并后，再做一次三目标全局非支配
筛选，最后才计算 \(B_p\)。

这种做法允许非凸 frontier，不依赖加权和的凸性，也没有使用遗传算法、粒子群或随机
population heuristic。

## 4. 求解与数值审计

主计算包括 `693` 个目标子问题，接受 `556` 个满足终止状态和残差标准的解，其中
`361` 个满足保留目标活跃条件。最大残差为：

| Audit quantity | Maximum |
|---|---:|
| Original equality residual | `2.03e-9` |
| Original inequality violation | `1.21e-8` |
| Epsilon-constraint violation | `1.37e-8` |
| Active retained-objective normalized error | `7.31e-8` |

11×21 主网格中，每个 grouping 都表示了 11/11 个条件值。含至少两个唯一点的切片数量
分别为：排放+ISI `10/11`、TAC+排放 `9/11`、TAC+ISI `11/11`。单点切片按论文定义
给出 \(B_p=0\)，但它们也意味着该处条件 frontier 的几何信息较少，所以报告中没有只给
均值而隐去切片点数。

### 嵌套网格敏感性

| Retained × epsilon | Emissions + ISI | TAC + emissions | TAC + ISI | Ranking |
|---|---:|---:|---:|---|
| 6×11 | 0.2456 | 0.2772 | 0.7180 | 1–2–3 |
| 6×21 | 0.2612 | 0.2772 | 0.7366 | 1–2–3 |
| 11×11 | 0.2166 | 0.3453 | 0.7327 | 1–2–3 |
| 11×21 | 0.2900 | 0.3453 | 0.7429 | 1–2–3 |

四个嵌套设置的排序完全相同，因此“ORCA 与 information-loss 排序一致”不是单一网格
选择造成的。但是总排放+ISI 的绝对值在 11×11 到 11×21 间变化约 `0.0733`，所以
绝对 loss 值尚不能解释为高精度极限；当前最稳健的结论是**相对排序**。

## 5. 与旧近似的对照

| Grouped objectives | Exact conditional | Equal-bin approximation | Nearest-neighbor approximation |
|---|---:|---:|---:|
| Total emissions + ISI | **0.2900** | 0.4382 | 0.5006 |
| TAC + total emissions | 0.3453 | **0.3090** | **0.3580** |
| TAC + ISI | 0.7429 | 0.9073 | 0.9320 |

三种算法都把 TAC+ISI 判为最不适合合并，但只有论文原式实现把排放+ISI判为最适合
合并，并与 ORCA 的第一名相符。这说明该案例对“如何构造 (J_p)”高度敏感；旧近似可
用于数据稀疏时的探索，却不能替代论文定义作最终验证。

## 6. 对 ORCA 合理性的解释

ORCA strength 为 `0.8831, 0.7781, 0.7557` 时，exact conditional loss 依次为
`0.2900, 0.3453, 0.7429`。方向和排序都符合假设：ORCA 越强，对应分组的 Pareto
信息损失越小。特别是 TAC+ISI 的 ORCA 只比 TAC+排放略弱，但 information loss
明显更大，说明 ORCA 更适合被解释为**候选分组排序指标**，而不是信息损失的线性定量
预测器。

这里使用的是完整非线性固定点、多重复的 paper-aligned ORCA 结果。更早报告的
`0.5031 / 0.5342 / 0.7674` 来自不同的端点/简化采样口径，不能与本次 11×21 条件
Pareto 切片直接比较。

## 7. Ipopt 与“严格 Pareto”的边界

Ipopt 可以求这个连续、平滑化的 CCUS NLP，并且本次所有保留点都通过了原始模型残差
审计。不过 Ipopt 只提供局部 NLP 解：

- `LOCALLY_SOLVED` 不是全局最优证明；
- 某个 epsilon 子问题未收敛不能自动解释为 infeasible；
- 多起点、已知 Pareto 可行见证、非支配筛选和嵌套网格审计只能增强可信度，不能替代
  确定性全局求解器给出的上下界与 optimality gap。

所以本次是“严格按 epsilon-constraint 标量化、严格按论文公式统计”的确定性局部
Pareto approximation，不是带全局证书的连续 Pareto frontier 全集。

## 8. Fallacy scan

- **Simpson's paradox**：没有分层观测数据被汇总，未见直接适用证据。
- **Ecological fallacy**：结论针对模型中的目标对，不外推到单个真实设施。
- **Berkson's bias**：Pareto 筛选确实是条件选择；因此结论限定于 Pareto-efficient
  设计，不能外推到全部可行设计。
- **Collider bias**：多个目标共同决定是否进入 frontier，可能诱导条件相关；ORCA–loss
  一致性只作设计空间验证，不作因果解释。
- **Base-rate neglect**：没有把某种 sink 的 frontier 频率解释为现实采用概率。
- **Regression to the mean**：无重复随机观测均值回归问题；ORCA 重复仅估计数值采样稳定性。
- **Survivorship bias**：未收敛子问题被记录而非删除后当作 infeasible，但局部失败仍可能
  造成 frontier 缺口。
- **Look-elsewhere effect**：报告全部四档嵌套网格，不挑选最有利设置。
- **Garden of forking paths**：新旧切片口径均公开；主方法由论文公式预先限定。
- **Correlation ≠ causation**：排序一致不能证明工艺目标之间存在现实因果关系。
- **Reverse causality**：两个指标都由同一模型输出构造，不赋予因果方向。
- **Local-to-global risk（额外）**：ORCA 的局部/约束几何与全局 Pareto loss 不是同一量；
  本案例支持相对排序，但不能据此宣称普遍等价。

## 9. 可复现输出

- 条件切片逐点数据：`direct_use_expansion_exact_info_loss_quantile21_points_audited.csv`
- 每个 p 的 B_p：`direct_use_expansion_exact_info_loss_quantile21_slices.csv`
- 四档网格汇总：`direct_use_expansion_exact_info_loss_quantile21_summary.csv`
- ORCA 与旧近似对照：`direct_use_expansion_exact_info_loss_quantile21_comparison.csv`
- 求解日志：`direct_use_expansion_exact_info_loss_quantile21_solve_log.csv`
- 验证图：`direct_use_expansion_exact_information_loss_quantile21_validation.*`
