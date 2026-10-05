# TAC 与原始 emissions–ISI 组合目标的 Pareto–Sankey 试算

## 本次定义

按照“不归一化、直接使用原始目标值”的要求，组合目标定义为

\[
J_{raw}=0.20\,\mathrm{TotEmiss}+0.80\,\mathrm{ISI}.
\]

其中 `TotEmiss` 的单位为 kt CO₂ yr⁻¹，ISI 为模型的原始安全指标，两个目标均未使用
payoff table 归一化。TAC 也以原始 cost units yr⁻¹ 参与支配筛选；图中仅除以
\(10^6\) 作为坐标单位转换，不是 payoff normalization。

需要注意，0.20 与 0.80 是代数系数，不等于最终组合目标中 20% 与 80% 的实际贡献。
由于 emissions 的原始数值较大，在五个关键点上，weighted emissions contribution
仍占组合目标的约 44.6–52.0%，weighted ISI contribution 约占 48.0–55.4%。

## Pareto 筛选

候选集为全部 225 个已验证的确定性三目标 Pareto 候选解。对候选解在
\((TAC,J_{raw})\) 平面执行精确两两支配筛选，再以仅用于识别数值重复的相对容差
\(10^{-5}\) 合并近重复解，最终得到 57 个不同的非支配点。没有进行插值，也没有使用
遗传算法或其他启发式 Pareto 方法。

底层非凸模型仍由多起点 Ipopt 求解，因此这里是相对于现有候选集的精确前沿，而不是
全局最优性的数学证书。

## 关键点

为避免在未归一化坐标上使用受单位支配的几何 knee definition，本次不计算 knee。
关键点改为由 sink allocation 直接定义的五个结构阶段。

| 点 | 角色 | TAC | Emissions | ISI | 0.20×Emissions | 0.80×ISI | \(J_{raw}\) | Saline 占比 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| A | Minimum TAC | 2.541×10⁶ | 88.319 | 20.357 | 17.664 | 16.285 | 33.949 | 0.0% |
| B | 10% saline allocation | 5.687×10⁶ | 82.440 | 19.278 | 16.488 | 15.422 | 31.910 | 9.6% |
| C | Sink-switch point | 9.521×10⁶ | 58.921 | 14.961 | 11.784 | 11.969 | 23.753 | 50.2% |
| D | 90% saline allocation | 1.155×10⁷ | 38.354 | 11.184 | 7.671 | 8.947 | 16.618 | 88.5% |
| E | Minimum raw emissions + ISI | 1.193×10⁷ | 32.500 | 10.108 | 6.500 | 8.087 | 14.587 | 100.0% |

## 网络结构变化

- A：所有送达流量进入 Urea；Ammonia 为 direct stream，Steel 经预处理进入 Urea。
- B：Ammonia 开始由 Urea 转向 Saline storage，Saline allocation 为 9.6%。
- C：Saline storage 与 Urea 分别承接约 50.2% 和 49.8% 的送达流量，是主要 sink
  发生切换的结构点。
- D：约 88.5% 的送达流量进入 Saline storage，Urea 仅保留约 11.5%。
- E：所有送达流量进入 Saline storage，达到最低 \(J_{raw}\)。

预处理比例从 A 的 61.8% 逐步降低到 E 的 55.1%，主要变化仍来自 Ammonia direct
flow 在 Urea 与 Saline storage 之间的重新分配，以及 Steel treated flow 最终转入
Saline storage。

## Greenhouse 为什么仍未出现

三维 Pareto frontier 中主 sink 为 Greenhouse 的点有 107 个，但在本次原始值组合下，
57 个二维非支配点中 Greenhouse 仍为 0。原因是即使给 ISI 系数 0.80，emissions 的
原始数值范围仍远大于 ISI；Greenhouse 带来的 ISI 改善不足以抵消其较高 emissions
与 TAC，因而在 \((TAC,J_{raw})\) 平面仍被 Saline Storage 解支配。

这说明“不归一化 + 0.20/0.80”并没有让 ISI 在实际组合目标中占 80%。若目标是明确
给予 ISI 四倍的无量纲偏好，应使用归一化后的 0.20/0.80；若坚持使用原始值但希望
Greenhouse 进入前沿，则需要进一步降低 emissions 的代数系数。

## Emissions 单位

当前 emissions 的统一单位是 kt CO₂ yr⁻¹。运输排放使用

\[
E_{transport}=TransportPow\times24\times\epsilon_p/10^6,
\]

并在 source emissions 中乘入 source CO₂ fraction。旧代码中缺少一致单位换算的
较大 emissions 数值不能直接与本结果比较。
