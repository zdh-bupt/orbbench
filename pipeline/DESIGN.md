# OrbBench Unified STM Task Pipeline — 设计与实验方案

生成时间：2026-09-13。本目录是自动科研流水线的工作区，所有代码、数据切片、结果、图和报告都在这里。
目标：把 stm场景汇总 里的 7 类任务连成一条上下游明确的 STM pipeline，用同一份 OrbBench 数据、同一套划分和指标跑通，
并完成《OrbBench_两组核心实验设计.md》里的两组实验（任务连接实验、质量与不确定度实验）。

## 0. 约束与数据口径

- 计算：仅 CPU（7 核，可用内存约 19 GB），无 GPU。深度模型不在此重训；已有的 8 模型预测表、Chronos-2、体素网络结果作为 "reported" 行导入对比表。
- 时间轴：全部可读轨道数据止于 2025-12-31。锁定评测窗 W = 2025-07-01 ~ 2025-12-31（与既有轨道预测测试窗一致）；训练/校准只用 W 之前的数据。
- 对象：
  - 核心队列 C：3,597 个在轨目标（Pruned 2,985 + Multi 612），1 min TEME 连续轨迹（L2），用于阶段划分、演化、交会筛查。
  - 真值队列 T：Sentinel-1A/2A/2B/3A/3B，POD 精密轨道（L4）+ S1A/S2A/S3A 机动日志，用于轨道预测真值、不确定度校准、机动检测标签、质量实验。
- 坐标：轨道状态 TEME→GCRS（astropy），误差与不确定度 RTN；时间 UTC。
- 数据源路径见 `orbpipe/config.py`。原始数据只读，不修改。

## 1. 架构（参考 TFB 四层）

```
Data layer      objects(type/regime/status) · TLE archive · PAF pool · L2 trajectories · POD · maneuver logs · space weather · DISCOS · CDM
Task layer      T1 phase → T2 maneuver → T3 weather → T4 forecast+UQ → T5 evolution → T6 warning → T7 avoidance   (统一 Task 接口: fit/predict/metrics)
Chain layer     Independent (A0) / Chained (A1)：上游输出按 chain.yaml 注入下游输入；逐边断开消融
Evaluation      固定划分、逐目标先算再取舰队中位数、按 type×regime 分层、bootstrap CI；覆盖率与精度分开报
Reporting       results/*.csv · figures/*.pdf · REPORT.md（自动生成）· status.json（巡检）
```

## 2. 七任务定义与上下游

| # | 任务 | 输入 | 输出 | 标签/真值 | 指标 | 下游用途 |
|---|---|---|---|---|---|---|
| T1 | 轨道阶段划分 | TLE 派生序列 (a, e, i, n, B*) | 每窗状态 {STABLE, RAISE, DECAY, …} | 规则银标（STARLINK_MOTION_PHASE 标准阈值） | Macro-F1, ARI, Boundary-F1@24h | T4：机动期用短历史/重置；T6：非稳定段膨胀 σ |
| T2 | 机动识别 | TLE 序列残差 (Δa, Δn, ΔB*) | 逐历元机动概率、事件区间 | Sentinel .man 真实日志 | F1/F2（±1 d 容差）、延迟 | T4：机动后只用机动后 TLE；T6：机动附近膨胀 σ |
| T3 | 空间天气预测 | F10.7/Kp/Ap 日序列（ctx 90 d） | 未来 3/7 d 点预测+分位 | 观测值 | MAE/RMSE、区间覆盖 | T4：预测 Ap/F10.7 进入 LEO 误差模型与 σ 膨胀 |
| T4 | 轨道预测 + UQ | 24 h 历史状态 + TLE + (T1,T2,T3 输出) | 0.5/1/4/8/12/24 h 位置速度 + RTN σ | Sentinel POD（真）；其余用 MTF 轨迹（伪真值，标明） | RTN 误差 P50/P95、σ 覆盖率、NEES | T6：预测状态与协方差 |
| T5 | 目标/星座演化 | 队列 t0 状态（小时级） | +1/6/24 h 高度×纬度密度、拥堵壳层排序 | 实际状态 | 密度重叠%、TV、热点 F1 | T6：只筛同/邻壳层候选对 |
| T6 | 碰撞预警 | 轨迹对 (+σ)；ESA CDM 序列 | TCA、miss、Pc、风险等级 | 参考轨迹 miss；CDM true_risk | Recall/假警、Recall@K、Brier | T7：高风险事件 |
| T7 | 自主避碰 | 高风险事件相对状态 | 推力时序 Δv | 安全距离 | 安全率、Δv、动作后 Pc | 回到 T6 复核 |

## 3. 实验

- E0 基准表：每任务 baselines × metrics，按 type×regime 分层（正文表）。
- E1 任务连接：A0 Independent vs A1 Chained（T4/T6/T7 指标 + 候选对数量/筛查时间）；逐边断开 phase→T4、maneuver→T4、weather→T4、evolution→T6、UQ→T6。
- E2 数据消融：B0 Full / B1 no-maneuver / B2 no-weather / B3 no-physical / B4 no-precise / B5 TLE-only。
- E3 轨迹质量：C0 raw-TLE nearest / C1 PAF nearest / C2 PAF+MTF，对 Sentinel POD 的 RTN P50/P90/P95、覆盖损失、真实机动保留率。
- E4 不确定度：σ 与真实误差相关性、分箱误差、50/90/95% 覆盖、NEES；D0 point / D1 fixed-σ / D2 OrbBench-σ 对预警与避碰的影响。

## 4. 运行与巡检

- `bash run_all.sh` 按阶段执行（幂等，已完成阶段跳过）；每阶段写 `status.json` 与 `logs/*.log`。
- `python -m orbpipe.monitor` 打印进度、最近错误、已产出结果；每 15 min 由会话定时任务调用并汇报。
- 阶段：S0 staging → S1 T3 weather → S2 T1 phase → S3 T2 maneuver → S4 T4 forecast+UQ → S5 T5 evolution → S6 T6 warning → S7 T7 avoidance → S8 experiments E1–E4 → S9 report。

## Task-level chain ablations (S8g, `orbpipe/chain_ablation.py`)
Each example runs the full chain a→b→c→d and the same chain with one upstream *task output* removed; the metric is always on the last task, against truth.
- X1 `x1_cdm_chain.csv` / `x1_cdm_ranking.csv`: T3 weather + T5 object attributes → T4 chaser-σ model (5-fold CV) → T6 Pc alert (≥1e-4) → T7 Pc-gated avoidance (down to 1e-5). 199 ESA CDMs, operator covariances = truth.
- X2 `x2_starlink_triage.csv`: T1 phase/SMA-rate + T2 score + T3 Ap/F10.7 → T4 forecast-reliability flag → T6 triage. 17 Starlinks, MEME truth, train ≤2026-07-05.
- X3 `x3_leo_chain.csv`: T5 shell-density ordering → T6 screening under a 20 % pair budget (σ-age Pc) → T7. 600 LEO objects, 3 d, 139 reference approaches.
- X4 `x4_sentinel_chain.csv`: T2 detector score → T4 σ-age × g(T2) → T6 Pc alert → T7, on Sentinel POD truth with 4 synthetic chasers per forecast (miss log-uniform 0.05–10 km, chaser σ 0.2 km).
Controller (all examples): thrust 0.3 m/s² along the encounter-plane miss direction in [TCA−150 s, TCA]; alert when predicted Pc ≥ 1e-4, thrust until predicted Pc < 1e-5 (or predicted miss ≥ 2 km for the distance rule); safety judged with the true covariance.
