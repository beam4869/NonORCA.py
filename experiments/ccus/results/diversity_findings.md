# CCUS 参数多样性试验与 ORCA 结果

## 结论

在不改动基线参数的前提下，本轮新增了 6 组探索性参数情景和 2 组对照情景。全部 8 组情景的可行性问题及 TAC、总排放、ISI 三个单目标问题均得到可接受的局部解；24 个单目标解的质量守恒、独立碳守恒诊断和不等式违反均不超过 `1e-6`。

六个探索情景都实现了三个目标分别选择三个不同的主导 sink。最适合作为后续 ORCA/Pareto 分析主情景的是：

1. `direct_use_expansion`：最大化“是否预处理”的结构反差；
2. `three_way_competition`：得到最清晰的 Urea–Saline Storage–Algae 三路竞争；
3. `capacity_limited_hubs`：通过有限容量产生不同的多-sink组合；
4. `high_capture_limited`：用于检验 60% 捕集目标下的网络拥塞和可行性。

## 为什么调整这些参数

文献支持把 sink 容量/注入率、时序可用性作为 source–sink matching 的核心约束，而不是只做一个统一容量倍数。[Tan (2012)](https://doi.org/10.1021/ie202821r) 与 [Tan et al. (2013)](https://doi.org/10.1002/ep.11630) 都显式建模了 sink 容量或注入率限制。

[NETL 的 CO₂ 运输与封存成本指南](https://www.netl.doe.gov/projects/VueConnection/download.aspx?filename=QualityGuidelinesforEnergySystemStudiesCarbonDioxideTransportandStorageCostsinNETLStudies_073124.pdf&id=c6081462-66d8-4cf2-a8e8-cd3424dce3d9)把源端压缩、管输距离/压降和符合管道规范的纯度分开处理；[NETL 杂质设计指南](https://doi.org/10.2172/1556902)也强调管输、EOR 和盐水层的杂质限制不同，实际项目应采用项目特定的 CO₂ 规格。因此，本轮分别改变 `Zmin_k`、运输距离、压缩功率和 sink 容量，而没有把它们压成一个综合系数。

预处理存在真实的纯度—能耗权衡。流程模拟研究发现，物理吸收达到接近 99% 纯度时能耗会明显上升，而约 95% 是较合理的能耗折中；全链条研究也建议管输 CO₂ 的最低纯度约为 95 vol%。参见 [Goto et al. (2013)](https://doi.org/10.1016/j.egypro.2013.05.171) 和 [Porter et al. (2016)](https://doi.org/10.1016/j.ijggc.2016.08.019)。因此高纯度 sink 使用 `0.95–0.999`，低纯度直用情景则被明确标为探索性，而非通用管输标准。

低纯度生物利用路线有实验依据：微藻研究曾直接使用 8–10 vol% CO₂ 的烟气，较新的双菌株系统也测试了含 CO₂、NO 和 SO₂ 的模拟烟气。[Kaštánek et al. (2010)](https://doi.org/10.1177/0734242X10375866)，[Cho et al. (2024)](https://doi.org/10.1016/j.biortech.2023.130051)。但本模型中 `direct_use_expansion` 对 Greenhouse 的 20% 门槛仍是机制辨识假设，需要污染物毒性、输送方式和具体温室作物数据才能校准。

`Mu_k` 在本模型中应解释为“利用/处理后未被有效保留的碳比例”，不能直接解释成地质封存泄漏率。IPCC 对选址、设计、运行和监测良好的地质储层评估为 100 年与 1000 年尺度保留率均超过 99%。因此旧模型中 11–50% 的值若称为地质泄漏会明显失真。[IPCC SRCCS Chapter 5](https://archive.ipcc.ch/pdf/special-reports/srccs/srccs_chapter5.pdf)

## 三个重点情景的系统结构

| 情景 | 目标 | 主导 sink | 活跃 sink（流量 > 1） | 预处理比例 |
|---|---|---|---|---:|
| `direct_use_expansion` | TAC | Urea | Urea | 61.81% |
|  | TotEmiss | Saline Storage | Saline Storage | 56.10% |
|  | ISI | Greenhouse | Greenhouse | 0.00% |
| `three_way_competition` | TAC | Urea | Urea | 61.81% |
|  | TotEmiss | Saline Storage | Saline Storage | 56.12% |
|  | ISI | Algae | Algae + Saline Storage | 0.00% |
| `capacity_limited_hubs` | TAC | Urea | Urea + Acetic Acid | 65.61% |
|  | TotEmiss | Saline Storage | Saline Storage + Urea | 89.12% |
|  | ISI | Greenhouse | Algae + Greenhouse | 66.37% |

基线中 TAC 主选 Urea，而 TotEmiss 与 ISI 都主选 Saline Storage。也就是说，新情景真正增加的是 ISI 路线和多-sink网络结构，不只是数值扰动。

## ORCA 的响应

为了保持比较一致，每个情景的 ORCA 都只聚合 TAC、TotEmiss 和 ISI 三个单目标最优点；因此这里的基线值与之前包含 MaxCapture 点的 ORCA 表不能直接混用。

| 情景 | TAC–TotEmiss | TAC–ISI | TotEmiss–ISI |
|---|---:|---:|---:|
| `baseline_forced` | 0.5660 | 0.2739 | 0.6590 |
| `three_way_competition` | -0.0298 | 0.0631 | 0.6752 |
| `direct_use_expansion` | 0.5031 | 0.5342 | 0.7674 |
| `capacity_limited_hubs` | 0.5429 | 0.5413 | 0.6976 |
| `high_capture_limited` | 0.5921 | 0.5450 | 0.7002 |

最明显的结果是 `three_way_competition` 把 TAC–TotEmiss 的 ORCA signed correlation 从 `+0.5660` 推到 `-0.0298`；`direct_use_expansion` 则把 TAC–ISI 从 `0.2739` 提高到 `0.5342`。这说明旧的一次一参数缩放没有充分激发系统拓扑变化，而 sink-specific 的纯度、容量、损失和运输组合能够明显改变 ORCA 的局部相关强度。

不过所有情景在固定 `K=2` 的描述性聚类中仍然得到“TotEmiss + ISI”与“TAC”两组。更丰富的系统结构改变了相关强度，但没有推翻这一粗粒度分组。

## 参数性质和限制

- `baseline_forced` 完全保留旧参数；`baseline_unforced` 只取消 `F₁ ≥ 100`，用于识别强制 Algae 流量的影响。
- 其余 6 组是确定性的机制辨识情景，不是从文献逐项拟合得到的真实地点参数。
- 高纯度门槛和需要区分容量、距离、压缩功率的做法有文献依据；具体容量扩张倍数、路线距离倍数和安全指数仍需项目数据校准。
- Ipopt 给出的是多起点筛选后的局部 NLP 最优解，不是全局最优证书。
- `diversity_score` 只用于筛选具有不同 sink 和预处理结构的参数情景，不进入 CCUS 优化目标。

## 输出文件

- `diversity_parameters.csv`：每个情景、每个 sink 的完整参数。
- `diversity_objective_solutions.csv`：24 个单目标解、sink 流量/份额、预处理比例和全部 82 个归一化变量。
- `diversity_candidate_summary.csv`：可行性与结构多样性汇总。
- `diversity_*_gradients.csv`：ORCA 所需目标和约束梯度。
- `diversity_orca_pairwise.csv`：各情景三对目标的 ORCA 结果。
- `diversity_orca_pointwise_pairwise.csv`：每个单目标最优点上的局部 ORCA 结果。
- `recommended_scenarios_orca_analysis.md`：四个推荐情景的参数机制、系统结构和详细 ORCA 解释。
