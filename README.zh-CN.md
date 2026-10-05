# ORCA：非线性多目标优化中的目标降维

[English](README.md) · [API 使用说明](docs/api.md) ·
[论文复现](docs/reproduction.md) · [图表对应清单](docs/reproduction_status.md)

ORCA 根据采样点处的目标梯度与约束信息，估计目标之间的关系并进行分组，
用于多目标优化中的目标降维。本仓库同时提供可安装的 Python 核心和论文中的
椭圆、DTLZ、CCUS 实验。

对应论文：Hongxuan Wang、Andrew Allman，
*ORCA: The Objective Reduction Community Algorithm for nonlinear many-objective
optimization problems*。

## 安装与快速运行

建议使用 Python 3.12。在仓库根目录执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e .
python examples/quickstart.py
```

Windows PowerShell 的激活命令为 `.venv\Scripts\Activate.ps1`。
最小示例仅依赖 NumPy、SciPy，使用确定性的 average-linkage 分组，
不需要 Julia 或外部优化求解器。包从当前仓库安装，未声称已发布到 PyPI。

需要 Leiden 社区检测或 Pyomo 接口时：

```bash
python -m pip install -e ".[leiden,pyomo]"
```

需要复现论文和运行检查时，安装记录的依赖版本：

```bash
python -m pip install -r requirements.txt
python scripts/check_release.py
```

CCUS 的优化求解还需要 `experiments/ccus/` 中的 Julia 1.10.0 环境。

## 使用方式

```python
from orca import NonlinearORCAConfig, nonlinear_orca
from orca.benchmarks import DTLZ5Problem

problem = DTLZ5Problem(
    intrinsic_dimension=2,
    num_objectives=3,
    k_tail=2,
    initial_point_strategy="deterministic",
)
result = nonlinear_orca(problem, NonlinearORCAConfig(
    num_groups=2,
    num_points_per_seed=8,
    step_size=0.03,
    random_seed=0,
    grouping_method="average_linkage",
))
print(result.adj_matrix)
print(result.groups)
```

这段代码用于演示接口，不代表论文的完整实验设置。
自己的模型可以实现 `NonlinearORCAProblem`；已有采样点、梯度及约束 Jacobian
时，可调用 `nonlinear_orca_from_sampled_gradients`。
分组结果本身不等于完成降维后的优化求解，后者由实验脚本实现。

## 目录结构

| 目录 | 用途 |
|---|---|
| `src/orca/` | 核心算法及可复用的 benchmark、实验辅助函数 |
| `examples/` | 最小示例和论文椭圆示例 |
| `experiments/dtlz/` | DTLZ5/6 分组、NSGA-III 和补充诊断 |
| `experiments/dtlz9/` | DTLZ9 几何、不同算法变体和实验 |
| `experiments/ccus/` | CCUS Julia 模型、Python 分析及原始结果快照 |
| `data/`、`results/` | 论文数据、逐 seed 记录及已保存结果 |
| `figures/` | 图形生成代码及绘图输入 |
| `tests/`、`scripts/` | 测试、数值核对和绘图入口 |
| `docs/` | API、复现命令、图表映射和迁移说明 |
| `environments/` | 验证环境依赖及历史环境记录 |
| `validation/` | 带日期的检查记录和数值差异说明 |
| `archive/` | 历史代码与原始元数据，不作为当前运行入口 |
| `outputs/` | 新生成的检查与绘图结果，Git 默认忽略 |

## 常用命令

在仓库根目录运行：

```bash
python -m pytest -q
python scripts/reproduce_available.py
python scripts/replot_ccus.py
python -m experiments.dtlz.run_dtlz5_5_16_orca_nsga3
python -m experiments.dtlz9.run_phase1_3 --help
```

完整参数与 CCUS 两种工况命令见 [复现说明](docs/reproduction.md)。
部分原始椭圆、绘图和 CCUS 驱动仍在各自目录写结果；运行完整实验前请按说明使用工作副本。

## 验证状态与已知限制

上一版已通过 41 项 Python 测试和 30 项数值核对。
本次结构调整后的实际验证见 [检查记录](validation/reorganization.md)。
整理没有改动算法公式、参数默认值或已保存的数值数据。

- Table 2 表注写样本标准差，原代码及数值使用 `ddof=0`；已保留两种口径比较。
- 此前 NSGA-III 单 seed 能运行，但 HV/IGD 与历史记录有差异，尚未确认精确重现。
- Fig. 4 的完整 DTLZ5(5,20) 敏感性网格仍缺失。
- Julia 优化未在本次整理中重跑；CCUS 检查基于已保存梯度与 NLP 解重算分析。

ORCA 的图权重不是 Pearson 相关系数；DTLZ9 的 `ORCA-current` 和 `ORCA-joint`
也不是同一实现。详细说明见 [已知问题](docs/known_issues.md)。

引用信息见 [CITATION.cff](CITATION.cff)。原有 MIT 许可证及历史署名核对记录保留，
不一致的旧作者邮箱未写入新安装配置。

## 新增：DTLZ9 单目标最优起点测试

已完成 180 次初始化对照和 40 次尺度诊断，记录见
[实验说明](results/dtlz9_single_objective_starts/README.md)。
原 n=10M 设置下最优端点采样退化为一个点；在 M=5、n=50 同时缩小步长与去重阈值后，
固定 K=2 的分组恢复。该调整只作为案例诊断，未修改核心算法或原论文结果。
