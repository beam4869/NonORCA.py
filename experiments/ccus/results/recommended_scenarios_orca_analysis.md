# 四个推荐 CCUS 情景的详细 ORCA 分析

## Material Passport

- Origin Skill: experiment-agent
- Origin Mode: validate
- Origin Date: 2026-08-10
- Verification Status: VERIFIED
- Version Label: recommended_orca_validation_v1
- Source: `run_diversity_experiments.jl` + `run_diversity_orca.py`
- Scope: `direct_use_expansion`、`three_way_competition`、`capacity_limited_hubs`、`high_capture_limited`

## 1. ORCA 的计算口径

每个参数情景先分别求 TAC、TotEmiss 和 ISI 的单目标最优解，再在这三个点计算目标梯度和约束 Jacobian。ORCA 使用：

- 全部 34 个等式约束；
- `g(x) ≥ -1e-6` 的活跃不等式；
- `alpha = 0.9`、`beta = 100` 的 competition-sensitive 权重；
- 三个单目标点上的局部 interactions 合并为情景级邻接矩阵。

ORCA signed correlation 的范围为 `[-1, 1]`。它表示活跃约束所定义的局部可行方向中，两个目标下降方向的一致或竞争程度；它不是样本统计相关系数，也不等于 Pareto frontier 上的全局 trade-off。

本报告同时给出：

1. **pointwise ORCA**：每个目标最优拓扑附近的局部关系；
2. **aggregated ORCA**：把同一情景的三个点按 ORCA interaction 权重汇总后的结果。

## 2. 基线参考

三个单目标点口径下的 `baseline_forced` 为：

| 目标对 | Aggregated ORCA signed correlation |
|---|---:|
| TAC–TotEmiss | 0.5660 |
| TAC–ISI | 0.2739 |
| TotEmiss–ISI | 0.6590 |

基线 TAC 主选 Urea，而 TotEmiss 和 ISI 都主选 Saline Storage，因此它没有真正把三个目标分配到三个不同的系统拓扑。

## 3. `three_way_competition`

### 参数逻辑

这个情景人为构造三种功能明确的 sink archetype：

- **Urea：经济路线。** 容量从 543 提高到 900，平均距离变为基线的 30%，压缩功率变为 18%。
- **Saline Storage：低排放路线。** 有效碳损失保持 0，但平均距离变为 3 倍、安全指数从 8 提高到 18，使其不再同时支配 TAC 和 ISI。
- **Algae：低 ISI/直接利用路线。** 容量从 126 扩展到 1000，安全指数从 7 降到 4，最低 CO₂ 比例保持 0.06，因此允许直接使用低纯度源气。

### 三个最优系统

| 目标 | Sink flow | 预处理比例 | TAC | TotEmiss | ISI |
|---|---|---:|---:|---:|---:|
| TAC | Urea = 586.0 | 61.81% | 3.539 M | 88.32 | 20.36 |
| TotEmiss | Saline = 530.8 | 56.12% | 12.962 M | 32.50 | 9.76 |
| ISI | Algae = 1000.0；Saline = 40.0 | ≈0% | 20.104 M | 160.51 | 6.77 |

Algae 的 24% 有效碳损失使 ISI 系统需要更大的直接气体流量才能满足相同的净捕集目标。因此它获得最低 ISI，却付出明显的成本和排放代价。

### ORCA 结果

| 目标对 | Aggregated ORCA | 相对基线变化 |
|---|---:|---:|
| TAC–TotEmiss | **-0.0298** | **-0.5958** |
| TAC–ISI | 0.0631 | -0.2107 |
| TotEmiss–ISI | 0.6752 | +0.0162 |

这里最重要的是局部符号反转：

| ORCA 评价点 | TAC–TotEmiss | TAC–ISI | TotEmiss–ISI |
|---|---:|---:|---:|
| TAC optimum | +0.7432 | +0.7113 | +0.6753 |
| TotEmiss optimum | +0.5374 | +0.2631 | +0.6752 |
| ISI optimum | **-0.5155** | **-0.2447** | +0.6752 |

ISI 最优点的负 interaction 获得约 116 的总权重，而另外两个点各约 42。原因是当前 ORCA 的 logistic 权重刻意提高竞争性、负 interaction 的贡献。因此聚合 TAC–TotEmiss 不是三个 pointwise 值的简单平均，而被推到接近零的轻微负值。

**解释：** 这是四个情景中最适合检验 ORCA 是否能够识别 topology-dependent conflict 的情景。它表明“成本–排放关系”会随系统从高纯度 Urea/Saline 路线切换到低纯度 Algae 路线而改变符号。

## 4. `direct_use_expansion`

### 参数逻辑

这个情景把 Greenhouse 设计成大容量、较低纯度、低安全指数的直接利用路线：

- Greenhouse 最低 CO₂ 比例从 0.94 降到 0.20；
- 容量从 508 增加到 1400；
- 有效碳损失从 0.50 降到 0.16；
- 安全指数从 7 降到 3，距离变为基线的 35%；
- Urea 仍保留低距离、低压缩功率的经济优势；Saline Storage 保留零有效碳损失。

Greenhouse 的 20% 门槛是机制辨识参数，不是已经校准的通用工业规格。

### 三个最优系统

| 目标 | Sink flow | 预处理比例 | TAC | TotEmiss | ISI |
|---|---|---:|---:|---:|---:|
| TAC | Urea = 586.0 | 61.81% | 2.541 M | 88.32 | 20.36 |
| TotEmiss | Saline = 530.8 | 56.10% | 11.963 M | 32.50 | 10.21 |
| ISI | Greenhouse = 958.9 | ≈0% | 33.247 M | 124.46 | 4.17 |

这是预处理结构最鲜明的情景：高纯度经济/封存路线需要约 56–62% 预处理，而 ISI 路线几乎全部直接利用。

### ORCA 结果

| 目标对 | Aggregated ORCA | 相对基线变化 |
|---|---:|---:|
| TAC–TotEmiss | 0.5031 | -0.0629 |
| TAC–ISI | **0.5342** | **+0.2603** |
| TotEmiss–ISI | **0.7674** | +0.1083 |

| ORCA 评价点 | TAC–TotEmiss | TAC–ISI | TotEmiss–ISI |
|---|---:|---:|---:|
| TAC optimum | 0.6216 | 0.6642 | 0.7674 |
| TotEmiss optimum | 0.5908 | 0.4622 | 0.7673 |
| ISI optimum | 0.3049 | 0.4765 | 0.7674 |

**解释：** 三个端点的系统选择差别很大，但 ORCA 仍为正。这不是矛盾：端点比较显示“选择哪条路线”的全局代价，而 ORCA 描述固定最优点附近的局部下降方向。结构不同并不自动意味着局部目标方向互相竞争。这个情景最适合说明 ORCA 与 endpoint/Pareto trade-off 的互补性。

## 5. `capacity_limited_hubs`

### 参数逻辑

该情景不允许任一优势 sink 单独承担全部捕集任务：

- Algae、Greenhouse、Saline、Methanol、Urea、Acetic Acid 容量分别设为 `220/500/360/500/360/500`；
- Greenhouse 允许 30% CO₂，Saline 要求 95%，化学转化 sink 保持 99.9%；
- Urea 继续保持低运输成本，但容量只有 360；
- Saline 保持零有效损失，但容量也只有 360。

### 三个最优系统

| 目标 | Sink flow | 预处理比例 | 活跃容量约束 |
|---|---|---:|---|
| TAC | Urea = 360.0；Acetic Acid = 267.0 | 65.61% | Urea 满容量 |
| TotEmiss | Saline = 360.0；Urea = 188.5 | 89.12% | Saline 满容量 |
| ISI | Algae = 136.2；Greenhouse = 500.0 | 66.37% | Greenhouse 满容量 |

### ORCA 结果

| 目标对 | Aggregated ORCA | 相对基线变化 |
|---|---:|---:|
| TAC–TotEmiss | 0.5429 | -0.0231 |
| TAC–ISI | **0.5413** | **+0.2674** |
| TotEmiss–ISI | 0.6976 | +0.0386 |

| ORCA 评价点 | TAC–TotEmiss | TAC–ISI | TotEmiss–ISI |
|---|---:|---:|---:|
| TAC optimum | 0.6314 | 0.5505 | 0.6979 |
| TotEmiss optimum | 0.5776 | 0.3088 | 0.6974 |
| ISI optimum | 0.4229 | **0.7628** | 0.6975 |

**解释：** 当各自最有利的 sink 达到容量上限后，目标改善只能通过相似的源流量重新分配完成，局部可行方向被容量约束收窄。因此 TAC–ISI 在 ISI 点表现出很强的正一致性。该情景最适合研究 ORCA 对 active capacity constraints 和多-sink网络的响应。

## 6. `high_capture_limited`

### 参数逻辑

该情景沿用容量受限的 sink 性质，但：

- 净捕集目标从 40% 提高到 60%；
- sink 容量扩展到 `300/700/500/650/450/650`，避免目标直接不可行；
- 仍保持 Urea 的经济优势、Saline 的零损失优势和 Greenhouse 的低安全指数优势。

### 三个最优系统

| 目标 | Sink flow | 预处理比例 | 活跃容量约束 |
|---|---|---:|---|
| TAC | Urea = 450.0；Acetic Acid = 515.5 | 89.33% | Urea 满容量 |
| TotEmiss | Saline = 500.0；Urea = 334.3 | 92.11% | Saline 满容量 |
| ISI | Algae = 204.3；Greenhouse = 700.0；Saline = 49.2 | 82.29% | Greenhouse 满容量 |

三个解的 NetCapture 均约为 755.37，说明 60% 捕集约束在三种最优解中都处于活跃状态。

### ORCA 结果

| 目标对 | Aggregated ORCA | 相对基线变化 |
|---|---:|---:|
| TAC–TotEmiss | 0.5921 | +0.0261 |
| TAC–ISI | **0.5450** | **+0.2712** |
| TotEmiss–ISI | 0.7002 | +0.0412 |

| ORCA 评价点 | TAC–TotEmiss | TAC–ISI | TotEmiss–ISI |
|---|---:|---:|---:|
| TAC optimum | 0.6837 | 0.6217 | 0.7006 |
| TotEmiss optimum | 0.5775 | 0.3036 | 0.7001 |
| ISI optimum | 0.5128 | 0.7121 | 0.6999 |

**解释：** 更高的捕集目标迫使更多高纯度、容量受限路线同时工作，三个目标共享更多活跃资源约束，因此总体保持中高强度的正相关。它没有产生 `three_way_competition` 那样的符号反转，但更适合做高捕集率下的可行性和网络拥塞压力测试。

## 7. 横向结论

| 研究目的 | 最合适情景 | 原因 |
|---|---|---|
| 检验 ORCA 能否识别局部冲突和符号变化 | `three_way_competition` | ISI 拓扑附近 TAC–TotEmiss 与 TAC–ISI 都转为负值 |
| 分析“是否预处理”的极端差异 | `direct_use_expansion` | ISI 解近 0% 预处理，另外两解约 56–62% |
| 分析多-sink、容量饱和与活跃约束 | `capacity_limited_hubs` | 三个目标分别激活 Urea、Saline、Greenhouse 容量上限 |
| 分析较高捕集目标和网络拥塞 | `high_capture_limited` | 60% 捕集约束活跃，所有最优系统均使用多条路线 |

四个情景在固定 `K=2` 的 average-linkage 聚类中仍然把 TotEmiss 与 ISI 分为一组、TAC 单独成组。说明参数改变显著影响相关强度和局部符号，但当前粗粒度 objective grouping 对这些变化较稳定。

## 8. 验证与注意事项

### 数值验证

- 四个推荐情景的 12 个单目标解全部通过 `1e-6` 残差检查。
- 全部 8 个参数情景重新运行后，24 个主结果、情景汇总、aggregated ORCA 和 pointwise ORCA 的最大数值差均为 `0.0`；仅求解时间未作比较。
- `test/runtests.jl`、原 ORCA/Pareto 检查和新增 diversity 检查均通过。

### Fallacy scan

- Coverage: 11/11 checked.
- `Correlation ≠ causation`：**CAUTION**。ORCA 数值只能描述给定模型和参数下的局部目标方向，不能证明现实工艺中一个目标会导致另一个目标变化。
- `Look-elsewhere effect`：**NOTE**。这些情景是为了产生结构差异而筛选的探索情景，不能把最大相关性变化当作预先设定的统计检验结果。
- `Garden of forking paths`：**CAUTION**。参数组合经过机制导向筛选，因此后续论文应同时报告基线、全部候选情景和筛选规则，而不能只展示最戏剧化的 `three_way_competition`。
- Simpson、ecological、Berkson、collider、base-rate neglect、regression-to-mean、survivorship、reverse causality：在本确定性优化实验中未发现适用证据或不适用。

### 模型边界

- Ipopt 结果是确定性交叉起点下的局部最优，不是全局最优证书。
- 探索情景的容量、距离倍数和安全指数没有被解释为实测参数。
- 零流量路线的纯度约束和变量下界在数学上属于活跃约束；ORCA 因而也包含“打开一条新路线”的局部约束信息。这是模型 active-set 的一部分，但解释时不能只把它理解成正在运行设备的约束。
- `direct_use_expansion` 的 Greenhouse 20% CO₂ 门槛需要进一步用具体作物、污染物容限和输送方式校准。

