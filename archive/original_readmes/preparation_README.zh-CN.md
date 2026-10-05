# ORCA 论文代码整理说明

这次整理以你上传的 **Nonlinear Paper.zip（Mac 本地文件夹）** 为主，
对照此前上传的 **NLMaOP_OptE (5).pdf**，替代上一版以云盘历史文件为主的整理包。

## 已整理的内容

- ORCA Python 核心、DTLZ5/6/9 实验、椭圆示例，以及对应的原始数值结果。
- CCUS 两种条件的模型、采样、ORCA、Pareto 和条件信息损失脚本。
- CCUS 原始 `Project.toml` 和 `Manifest.toml`、Python 历史锁文件与本次验证环境。
- 可运行的检查入口、图表对应表、版本来源、英文 README、引用文件和 GitHub CI 配置。
- 清理机器绝对路径；排除虚拟环境、缓存、NAS 数据、重复大型图片和论文草稿。

原始计算方法没有重写。主代码放在原有模块目录中，旧 Julia 探索脚本放在 `archive/`。

## 快速检查

在解压后的目录中执行：

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python scripts/check_release.py
```

本次已通过 **41 项原有 Python 测试**和 **30 项新增数值核对**。
重新生成的椭圆 68 个点、相关矩阵和分组均吻合；CCUS 两种条件的
ORCA 及分组结果、从保存的 NLP 解重算的条件信息损失也吻合。
Julia 求解环境未在这里运行，因此这些检查不等于重做了 CCUS 优化实验。

## 需要你了解的剩余问题

1. **Table 2 的标准差口径不一致。** 表注写 sample SD，但原脚本及表中数字
   用的是 `ddof=0`。`validation/table2_standard_deviation_audit.csv` 同时列出原值
   和 `ddof=1` 重算值；没有擅自改论文或覆盖原结果。
2. **NSGA-III 历史数值还不能保证精确重现。** Seed 0 的三种优化设置均跑通，
   评价次数及可行性正确，但 HV/IGD 与原记录不同。需要确认生成 Table 2 时
   的完整 Python 环境和代码版本，才能确定差异原因。
3. **Fig. 4 的完整敏感性网格尚未找到。** 找到了相关旧 Julia 代码和部分数据，
   但不能据此补造 DTLZ5(5,20) 每格十次实验的原始记录。

完整命令见 `docs/reproduction.md`，逐图逐表对应关系见
`docs/reproduction_status.md`，详细问题见 `docs/known_issues.md`。
本包尚未上传或推送到 GitHub。
