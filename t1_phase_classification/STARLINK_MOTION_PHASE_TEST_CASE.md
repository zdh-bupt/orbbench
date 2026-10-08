# 基于 TLE 数据特征的目标运动状态测试标准

## 1. 测试目标

本标准检验模型能否仅根据连续 TLE 及其可复现派生量，把空间目标轨迹片段划分为可观测运动状态。

标签描述的是轨迹形状，不是运营方任务意图。因此测试结果可以表述为“持续升高”“短时偏移后回归”，不能表述为“部署”“碰撞规避”或“离轨指令”。外部事件和论文标注只保留为来源审计字段，不参与状态判定，也不能输入待测模型。

## 2. 评测对象

测试对象分为四层，避免把单条 TLE、预测窗口和完整事件混为一谈。

| 层级 | 定义 | 用途 |
|---|---|---|
| TLE 观测 | 一个 epoch 的两行 TLE 及派生轨道量 | 最小原始数据单元 |
| `5→1` 窗口 | 5 条历史 TLE 预测紧随其后的 1 条 TLE | 自回归表征输入 |
| 连续片段 | 同一 NORAD 目标的一组连续窗口；相邻目标 epoch 间隔不超过 36 小时 | 计算持续时间和轨迹形状 |
| 状态段 | 相邻且状态相同的片段合并后的结果 | 分段指标和可视化 |

片段边界可以来自独立规则管线或变化点算法，但不能来自待评模型的最终聚类标签。当前演示包使用既有规则产生候选边界，并在大于 36 小时的数据缺口处再次切分；本标准只负责根据片段自身数据特征命名状态。

## 3. 状态体系

| ID | 状态 | 中文含义 | 评测层级 |
|---:|---|---|---|
| 0 | `STABLE_ORBIT` | 稳定轨道 | 主定量类 |
| 1 | `SUSTAINED_RAISE` | 持续升高 | 主定量类 |
| 2 | `SUSTAINED_DECAY` | 持续降低 | 主定量类 |
| 3 | `OFFSET_PLATEAU` | 长时偏移平台/长周期往返 | 案例类 |
| 4 | `SHORT_EXCURSION_RETURN` | 短时显著偏移后回归 | 案例类 |
| 5 | `UNKNOWN_TRANSITION` | 数据不足、边界不清或规则冲突 | 拒判类 |

主指标只使用前三类。`OFFSET_PLATEAU` 和 `SHORT_EXCURSION_RETURN` 的独立目标数太少，只报告案例结果；`UNKNOWN_TRANSITION` 不进入聚类标签匹配。

## 4. 片段特征

对每个连续片段计算以下统计量。时间一律使用真实 UTC，不把样本序号当作等间隔时间。

| 符号 | 字段 | 定义 | 判别作用 |
|---|---|---|---|
| `N` | `SAMPLE_COUNT` | 片段内有效 `5→1` 窗口数 | 控制最小证据量 |
| `T` | `DURATION_DAYS` | 首末目标 epoch 的时间差 | 区分短时事件和长时平台 |
| `Δa` | `NET_A_CHANGE_KM` | 末点半长轴减首点半长轴 | 判断升高、降低或回归 |
| `R_a` | `A_RANGE_KM` | 片段内半长轴最大值减最小值 | 判断是否存在显著偏移 |
| `p+` | `POSITIVE_DA_FRACTION` | `da/dt > 0` 的窗口比例 | 判断升高方向的一致性 |
| `p-` | `NEGATIVE_DA_FRACTION` | `da/dt < 0` 的窗口比例 | 判断降低方向的一致性 |
| `G_max` | `MAX_TARGET_GAP_HOURS` | 相邻目标 epoch 最大间隔 | 数据连续性门控 |

其他 18 个 TLE 派生特征仍可作为模型输入，包括偏心率、倾角、角度的 `sin/cos`、平均运动、`BSTAR`、近远地点高度和真实时间间隔。状态银标只使用上表的片段统计量，避免把数据源特有的 SGP4 残差可用性泄漏给标签。

## 5. 互斥判定规则

### 5.1 质量门控

1. 按 `NORAD_CAT_ID, EPOCH` 排序并去重。
2. 任一相邻目标 epoch 间隔大于 36 小时，必须切成两个片段。
3. 半长轴、时间和方向统计量出现非有限值时，标为 `UNKNOWN_TRANSITION`。
4. 除短时回归类允许至少 3 个目标窗口外，其余确定状态至少需要 5 个目标窗口。

每个目标窗口本身已有 5 条有效历史 TLE，因此“3 个目标窗口”不等于只有 3 条原始 TLE。

### 5.2 决策顺序

按下列顺序执行，命中第一条后停止。顺序用于消除边界附近的规则重叠。

| 优先级 | 输出状态 | 数据特征条件 |
|---:|---|---|
| 1 | `SHORT_EXCURSION_RETURN` | `N ≥ 3`，`T < 10 d`，`R_a ≥ 4 km`，`|Δa| ≤ 1 km`，`p+ ≥ 0.25` 且 `p- ≥ 0.25` |
| 2 | `OFFSET_PLATEAU` | `N ≥ 5`，`T ≥ 10 d`，`R_a ≥ 5 km`，`|Δa| ≤ 1 km`，`p+ ≥ 0.30` 且 `p- ≥ 0.30` |
| 3 | `SUSTAINED_RAISE` | `N ≥ 5`，`T ≥ 2 d`，`Δa ≥ 4 km`，`p+ ≥ 0.55` |
| 4 | `SUSTAINED_DECAY` | `N ≥ 5`，`T ≥ 3 d`，`Δa ≤ -0.9 km`，`p- ≥ 0.70` |
| 5 | `STABLE_ORBIT` | `N ≥ 5`，`R_a ≤ 2.5 km`，`|Δa| < 0.9 km` |
| 6 | `UNKNOWN_TRANSITION` | 未命中以上任何规则 |

这组阈值是当前 Starlink 演示数据的可复现工程判据，不是跨星座物理常数。迁移到不同轨道高度、TLE 密度或目标类型时，必须只在训练/验证目标上重新校准，不能查看最终测试目标。

## 6. 标签生成与输出

标准分类命令为：

```powershell
python scripts/plot_tle_observable_state_division.py
```

脚本按上述规则生成：

- `reports/observable_regime_demo/tle_fragment_state_catalog.csv`：每个 TLE 片段的特征、命中规则和状态；
- `reports/observable_regime_demo/observable_state_division.png`：状态划分图；
- `reports/observable_regime_demo/observable_state_division.svg`：同一图的可编辑矢量版本。

图中左侧用持续时间、半长轴净变化和变化范围展示所有片段的分布；右侧给出六类状态的真实数据片段。图表使用 2026-09-08 的当前演示包生成：共 2,638 个连续片段、69,764 个 `5→1` 窗口。

![TLE 数据片段状态划分](reports/observable_regime_demo/observable_state_division.png)

## 7. 数据划分与防泄漏

1. 以 NORAD ID 为最小隔离单元，稳定哈希划分为 70% 训练、15% 验证、15% 最终测试。
2. 同一 NORAD ID 不得跨集合；当前数据包的三个集合交集均为 0。
3. 标准化、表征训练、PCA、聚类中心、阈值校准和簇到状态映射只能使用训练/验证集合。
4. 最终测试只运行一次；自监督训练也不得使用最终测试目标的无标签窗口。
5. 外部论文事件可以用于案例解释，但不得用于状态特征、模型输入或主指标筛选。

## 8. 评测标准

### 8.1 数据完整性门

- `SAMPLE_ID` 在索引和 NPZ 中一一对应；
- 所有输入张量为有限值；
- 每个 `5→1` 窗口的相邻 TLE 间隔均在 `(0, 36 h]`；
- NORAD 级训练、验证、测试集合无交集；
- 所有未命中状态规则的片段进入 `UNKNOWN_TRANSITION`，不得强制归类。

任一项失败，整次测试无效。

### 8.2 主定量指标

- 聚类一致性：ARI、NMI；
- 匈牙利映射后：macro-F1 及逐类 precision/recall；
- 分段：对称 segment IoU、过分割率、漏段率；
- 物理一致性：类内稳健离散度、相邻状态效应量、非法碎片率；
- 鲁棒性：随机丢弃 TLE、合理噪声和不同随机种子下的分段一致性。

主指标只在 `STABLE_ORBIT`、`SUSTAINED_RAISE`、`SUSTAINED_DECAY` 上计算，并同时报告每类窗口数、独立 NORAD 数和片段数。不能只报告窗口加权总分，因为 `STABLE_ORBIT` 占比远高于其他状态。

### 8.3 边界与案例指标

- Boundary-F1@12h/24h/48h 仅作为诊断，直到最终测试集中至少有 20 个独立主状态边界；v2 回溯测试当前只有 8 个。
- `OFFSET_PLATEAU` 与 `SHORT_EXCURSION_RETURN` 分别报告案例时间线、持续时间误差和 segment IoU，不并入主 macro-F1。
- `UNKNOWN_TRANSITION` 报告拒判率和其前后状态，不参与簇编号到语义状态的匹配。

## 9. 当前标签迁移说明

旧标签仅用于追溯候选片段来源，不参与第 5 节的状态判定。

| 旧语义标签 | 可观测状态候选 | 迁移含义 |
|---|---|---|
| `ON_STATION`、`LOW_ORBIT_PHASING` | `STABLE_ORBIT` | 只保留局部稳定形状，不保留业务/相位语义 |
| `DEPLOYMENT_RAISE` | `SUSTAINED_RAISE` | 只保留持续升高形状 |
| `DEORBIT` | `SUSTAINED_DECAY` | 只保留持续降低形状 |
| `SHELL_TRANSFER` | `OFFSET_PLATEAU` | 只保留长时偏移/往返形状 |
| `COLLISION_AVOIDANCE` | `SHORT_EXCURSION_RETURN` | 只保留短时偏移后回归形状 |
| 候选类与 `UNKNOWN` | 重新判定 | 命中特征规则才归类，否则拒判 |

现有冻结实验结果仍对应 `output/tle_observable_regime_testset` 中的旧版代理标签；本次生成的片段目录用于重新组织测试标准和数据特征审计，不静默改写既有最终测试结果。

### 9.1 v2 标签回溯复测

在 TLE 输入、自监督下一 TLE 预测目标、NORAD 划分和聚类数均未改变的前提下，复用既有无监督模型，并仅用训练集 v2 标签重新拟合簇到状态的 Hungarian 映射。测试主样本为 10,209 个窗口、376 个目标；三种子结果如下：

| 方法 | ARI | NMI | Macro-F1 | 对称 segment IoU | Boundary-F1@24h |
|---|---:|---:|---:|---:|---:|
| Raw PCA + KMeans | 0.7938 ± 0.0003 | 0.6668 ± 0.0006 | 0.7010 ± 0.0003 | 0.8826 ± 0.0002 | 0.2927 ± 0.0000 |
| PELT + KMeans | 0.6598 ± 0.0000 | 0.4951 ± 0.0000 | 0.4708 ± 0.0000 | 0.8905 ± 0.0000 | 0.1905 ± 0.0000 |
| Transformer embedding + KMeans | -0.0168 ± 0.0042 | 0.0026 ± 0.0016 | 0.3244 ± 0.0043 | 0.1455 ± 0.0053 | 0.0000 ± 0.0000 |

完整逐类结果、原始指标和 v1 对照见 `reports/observable_regime_feature_labels_v2/retest/RESULTS.md`。这是一项标签重组后的回溯性评测，不构成新的未见冻结测试。

## 10. 结论边界

允许的结论是：模型能否从 TLE 中恢复稳定、持续升高、持续降低、长时偏移和短时回归等可观测轨迹形态。

不允许的结论是：模型识别了运营方真实生命周期、燃烧原因、碰撞规避意图或精确控制指令。小于阈值的轨道维持、短于采样间隔的机动和跨数据缺口的变化仍不可可靠判定。

数据来源为 Li et al., *A Networking Perspective on Starlink's Self-Driving LEO Mega-Constellation*, ACM MobiCom 2023 及作者公开 artifact；当前核对的 artifact 提交为 `0188c05d487c5614f700ad4894e77bb9881f791d`。
