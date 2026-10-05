# DTLZ9–ORCA Phase 3：Grouping recovery 与鲁棒性报告

## 最重要的结论

现有 ORCA 在已知 `K=2` 时能输出 DTLZ9 的正确标签，但这个结果主要由强制聚成两组和 tie-breaking 产生，不能证明它识别出了 correlation strengths。它在精确 PF 上给第一组内部边强度 0，并在 unknown-K 设置下把 M 个 objectives 全部分成单独组。联合活跃切空间版本则恢复了理论上的组内 `+1`、跨组 `-1`，并在 `M=3,5,10,20` 上以 unknown K 100% 恢复两组。

## 实验矩阵

- Objectives：`M={3,5,10,20}`。
- Decision dimension：`n=M` 与 `n=10M`。
- 精确 PF：每个配置 25 个解析点。
- 随机实验：30 seeds，每个 case 12 个点。
- 部分活跃比例：`q={0,0.25,0.5,0.75,1}`。
- 异质污染：`q={0.25,0.5,0.75}`，inactive objectives 使用相互独立的正偏移。
- Gradient noise：相对尺度 `0.01` 与 `0.05`。
- Point-selection diagnostic：现有 ORCA fixed-point generator，`M={3,5,10}`。
- 方法：Raw-Cosine、Single-Constraint-Mean、ORCA-current、ORCA-joint、Pearson、Spearman。
- Grouping：fixed `K=2` 与 unknown-K positive components；同时报告阈值 `-0.05/0/+0.05` 敏感性。

## 1. 精确 PF：强度恢复与 unknown-K grouping

下表为 `n=10M`；`n=M` 得到同样的结构性结论。

| M | 方法 | 组内均值 | 跨组均值 | unknown-K 估计组数 | 精确恢复率 |
|---:|---|---:|---:|---:|---:|
| 3 | ORCA-current | 0.000 | -0.645 | 3 | 0% |
| 3 | ORCA-joint | 1.000 | -1.000 | 2 | 100% |
| 5 | ORCA-current | 0.000 | -0.377 | 5 | 0% |
| 5 | ORCA-joint | 1.000 | -1.000 | 2 | 100% |
| 10 | ORCA-current | 0.000 | -0.185 | 10 | 0% |
| 10 | ORCA-joint | 1.000 | -1.000 | 2 | 100% |
| 20 | ORCA-current | 0.000 | -0.0917 | 20 | 0% |
| 20 | ORCA-joint | 1.000 | -1.000 | 2 | 100% |

ORCA-joint 的 signed-matrix MAE 为 `7.4×10^-17` 到 `3.24×10^-16`。ORCA-current 的 MAE 随 M 增大，从 0.380 上升到 0.941。

Pearson 和 Spearman 在精确 PF 上也能恢复两组，因为所有第一组 objective values 都是同一个 `cos(theta)`，而 `fM=sin(theta)` 单调反向；这属于理想 sampling baseline，不代表它们在一般非线性 Pareto set 上具有同样的几何解释。

## 2. 为什么 fixed K 的 100% 不能单独作为成功证据

在精确 PF 上，Raw-Cosine、ORCA-current 和 ORCA-joint 的 fixed-`K=2` recovery 都是 100%。但 Raw-Cosine 的全部非对角边为 0，ORCA-current 的第一组内部边也为 0。这两个结果只因 average linkage 被要求最终保留两组，并按确定性 tie 顺序合并第一批 objectives。

因此：

- fixed-K label accuracy 回答“知道答案是两组时能否产生这组标签”；
- signed strengths 和 unknown-K recovery 才回答“算法是否从关系证据中发现了这两组”。

阈值敏感性进一步支持这一点。ORCA-current 在阈值 0 时估计 M 组；把阈值改为 `-0.05` 后才得到 2 组，但这是依赖已知答案选择阈值。ORCA-joint 在 `-0.05`、0 和 `+0.05` 三个阈值下均保持 2 组。

## 3. 部分交集与异质污染

当一个点只激活部分结构约束时，joint estimator 只能使用该点上可见的联合几何。以 `M=10,n=100` 为例：

- shared-offset sampling：q=0 时 recovery 0%；q=0.25 时 73.3%；q≥0.5 时 100%。
- heterogeneous contamination：q=0.25 时 63.3%；q≥0.5 时 100%。
- ORCA-current：所有上述 unknown-K cases 均为 0%。

Pearson/Spearman 在这些合成样本上总体仍较强，因为各 objectives 仍共享 theta 这一潜变量；加入异质偏移后，部分配置出现下降，例如 `M=20,q=0.25` 时 Pearson 为 86.7%、Spearman 为 80.0%。因此关于 value correlation 对 sample quality 的敏感性只能判为“部分支持”，不能据此声称梯度方法全面优于值相关方法。

## 4. Gradient noise

在精确 PF 上加入 1%–5% gradient noise 后，ORCA-joint 的 unknown-K recovery 基本保持在 96.7%–100%。ORCA-current 仍大多为 0%；个别高 M case 因零阈值附近的微小随机正边偶然形成连通分量，不具有稳定结构解释。

## 5. 现有 ORCA selected-point generation 是主要瓶颈

ORCA fixed-point generator 本身没有程序失败，但它生成的点很少达到所有 `M-1` 结构约束同时活跃的交集。

| M | n/M | 全约束交集点比例 | ORCA-joint unknown-K recovery | ARI |
|---:|---:|---:|---:|---:|
| 3 | 1 | 23.8% | 63.3% | 0.633 |
| 5 | 1 | 25.4% | 76.7% | 0.850 |
| 10 | 1 | 22.5% | 60.0% | 0.707 |
| 3 | 10 | 0% | 0% | 0.000 |
| 5 | 10 | 0% | 0% | 0.117 |
| 10 | 10 | 0% | 0% | 0.071 |

这说明即使 correlation estimator 改为正确的联合切空间版本，若 point selection 没有覆盖多约束交集，仍无法稳定恢复 DTLZ9 grouping。下一阶段最重要的算法工作不是调聚类器，而是增加面向 active-set intersection 的采样机制。

## Hypothesis verdicts

| 假设 | 结论 | 证据 |
|---|---|---|
| H1：联合切空间能恢复解析 signed strengths | 支持 | exact PF matrix error ≤ `3.24×10^-16` |
| H2：current per-constraint ORCA 会遗漏组内 alignment | 支持 | 组内边恒为 0；cross edge 随 M 衰减 |
| H3：正确几何支持 unknown-K grouping | 支持 | ORCA-joint 在 M=3–20 均为 100% |
| H4：sample quality/active-set coverage 决定恢复率 | 支持，但 value-baseline 部分仍理想化 | q 和 selected-point experiments 显著影响 joint recovery |
| H5：几何结果可扩展到 M=20,n=200 | 支持到本实验规模 | 无失败，精确 PF 仍达机器精度；尚未做 Phase 4 大规模 runtime study |

## 最终判断

对最初问题“现有 ORCA 能否把 DTLZ9 objectives 分组并看清 correlation/competition”应作分层回答：

1. **仅要求固定 K=2 标签：表面上可以，但证据不可靠。** Raw 和 current ORCA 都会因 tie/强制合并得到正确标签。
2. **要求可解释的 correlation strengths：current ORCA 不可以。** 它看不到第一组内部 `+1`，且竞争强度随 objective 数量衰减。
3. **采用 joint-active-tangent：几何上可以。** 在精确/高覆盖样本上能正确恢复 `+1/-1` 和 unknown-K grouping。
4. **端到端现有 point selection：还不可以。** 尤其 `n=10M` 时没有生成全约束交集点，是进入 Phase 4 前必须解决的瓶颈。

## Phase 3 状态与下一步边界

**Phase 3 已完成并通过预设执行标准。** 结果支持进入 Phase 4，但本次没有修改 ORCA core、没有实现 intersection-aware point selection，也没有做下游 multiobjective optimizer 的性能比较。建议 Phase 4 按以下顺序推进：

1. 在 ORCA 中新增 joint-active-set projection 选项；
2. 新增 intersection-targeted point generation；
3. 用 unknown-K 和 fixed-K 双轨验证；
4. 最后再做 reduction/downstream optimizer 性能实验。
