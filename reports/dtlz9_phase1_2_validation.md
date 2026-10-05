# DTLZ9–ORCA Phase 1–2：实现与几何验证报告

## 结论先行

DTLZ9 analytical adapter、精确 Pareto-front generator、受控相邻表面 generator、解析 Jacobians 和 joint-active-tangent estimator 均已实现。6 项自动测试全部通过；在 `M={3,5,10}`、`n/M={1,10}`、150 个精确 PF 点上，联合 signed matrix 与理论矩阵的最大误差为 `6.66×10^-16`。

## 实现定义

对 objective block `B_j`，使用

\[
f_j(x)=\sum_{k\in B_j}x_k^{0.1}.
\]

结构约束为

\[
g_j(x)=1-f_j(x)^2-f_M(x)^2\le 0,\qquad j=1,\ldots,M-1.
\]

精确 PF 使用一维参数化

\[
f_1=\cdots=f_{M-1}=\cos\theta,\qquad f_M=\sin\theta,
\]

所以 oracle grouping 为

\[
\{f_1,\ldots,f_{M-1}\},\quad\{f_M\}.
\]

联合活跃切空间上的理论 signed matrix 为：第一组内部 `+1`，第一组与 `fM` 之间 `-1`，对角线为 `+1`。

## 自动验证结果

| 检查 | 最大观测误差/残差 | 结论 |
|---|---:|---|
| PF structural constraint | `5.69×10^-16` | 通过 |
| Objective Jacobian vs finite difference | `1.02×10^-10` relative | 通过 |
| Constraint Jacobian vs finite difference | `1.43×10^-10` relative | 通过 |
| Raw-gradient off-diagonal cosine | `0` | 与 block 正交理论一致 |
| Joint signed matrix vs oracle | `6.66×10^-16` | 通过 |
| Projector symmetry residual | `0` | 通过 |
| Projector idempotence residual | `2.76×10^-15` | 通过 |
| Tangency residual | `1.32×10^-15` | 通过 |

测试命令的最终结果为 `6/6 tests passed`。

## 不变性验证

### Objective normal extension

将每个 objective gradient 替换为

\[
\nabla f_i + \alpha_i^T J_g
\]

不会改变约束流形上的 objective restriction。对 `alpha scale={0.1,1,10}`，joint matrix 最大变化为 `5.55×10^-16`，而 raw-gradient cosine 可发生明显变化。这验证了 joint tangent estimator 对等价 normal extension 的不变性。

### Constraint row mixing

用可逆矩阵左乘 active constraint Jacobian 不改变联合切空间。对正交混合和 condition number 10 的混合：

- objective value 最大误差：`1.91×10^-14`；
- joint matrix 最大变化：`5.55×10^-16`；
- projector 最大变化：`1.22×10^-15`。

condition number `10^8` 的极端混合作为 report-only stress test：joint matrix 仍保持在 `5.55×10^-16` 内，但 projector elementwise 变化达到 `1.18×10^-9`，反映了病态矩阵下的数值敏感性。

## Current ORCA 的解析预期也被数值复现

在精确 PF 上，current ORCA 的第一组内部 signed strength 为 0；第一组与 `fM` 的竞争强度为

\[
-\frac{1}{1+0.55(M-2)}.
\]

因此 cross-group strength 随 M 增大而趋近 0：`M=3` 为 `-0.645`，`M=5` 为 `-0.377`，`M=10` 为 `-0.185`，`M=20` 为 `-0.0917`。这说明问题不是实现 bug，而是逐约束聚合对 DTLZ9 联合几何的结构性衰减。

## 已处理的异常

最初 smoke run 在 `n/M=1` 的 off-front 构造中遇到 `cos(theta)+delta>1`，超过单变量 block 的最大 objective value。修复方式是将受控 objective target 限制在 `nextafter(block_size, 0)`；随后完整运行无 point-generation exception。该修复只保证构造落在 decision domain 内，不改变精确 PF 几何。

## Phase 2 验收

**全部通过。** analytical derivatives、精确 PF、联合切空间 ground truth、normal-extension invariance 和 constraint-row-mixing invariance 均满足预设标准。Phase 3 可将 joint estimator 作为几何对照，但它仍属于研究原型，不是对 ORCA 核心包的正式修改。
