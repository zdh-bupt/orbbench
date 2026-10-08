# 闭集 LEO 三维密度预测

本项目从逐星 TEME 状态 CSV 构建可审计的闭集训练集，并预测起点 `t0` 已知成员在未来 `1 h / 6 h / 24 h` 的全球三维体素分布。

项目目前实现：

- 递归发现 CSV，并通过卫星清单选择训练对象；
- 由首个有效 TEME 位置速度计算瞬时轨道根数，整颗跳过非 LEO 文件；
- 流式读取多年 CSV，只选择每个 UTC 整点之前且龄期不超过 90 秒的记录，并因果传播到统一整点；
- TEME 转 ITRS/WGS84，并生成 ENU 速度；
- 按卫星写入带月级 row group 的 Parquet 状态缓存；
- 构造固定 `t0` 成员的闭集窗口、CIC 标签、速度矩和上下域外桶；
- 从 `t0` 状态做因果两体 Kepler 传播，生成三个时效的物理先验；
- 多尺度 3D U-Net + bottleneck ConvGRU 学习有限物理残差；
- 对域内体素和两个域外桶做联合 softmax，保证非负与总数守恒；
- 严格时间切分、训练、验收、自动回退物理基线和无未来标签的在线推理。

## 输入格式

每个文件只包含一颗卫星，默认文件名为 `<satellite_id>_pruned.csv`，必需列为：

```text
Epoch_UTC,X_m,Y_m,Z_m,VX_m_s,VY_m_s,VZ_m_s
```

位置单位是米，速度单位是米每秒，坐标系是 TEME。目录可以包含任意层级的子目录。

LEO 判定默认要求瞬时轨道近地点不低于 100 km、远地点不高于 2000 km。建模网格按研究指导覆盖 300 至 1500 km；属于 LEO 但状态落在网格外时，质量进入 `outside_below` 或 `outside_above`，不会丢失。

## 正式工作流

先创建 `configs\satellite_ids.txt`，每行一个卫星 ID，然后执行：

```powershell
python scripts\audit_raw_data.py --config configs\default.yaml `
  --selection-file configs\satellite_ids.txt

python scripts\prepare_states.py --config configs\default.yaml `
  --selection-file configs\satellite_ids.txt

python scripts\prepare_closed_cohort.py --config configs\default.yaml
python scripts\train.py --config configs\default.yaml

python scripts\evaluate.py --config configs\default.yaml `
  --checkpoint artifacts\model-best.pt --split test

python scripts\predict.py --config configs\default.yaml
```

## 输出

```text
data/processed/states/             逐星小时 Parquet 和 state_manifest.json
data/closed_cohort_shards/         月度闭集训练分片和 manifest.json
artifacts/model-best.pt            验证集最优候选模型
artifacts/deployment_decision.json 锁定测试后的部署选择
reports/evaluation.json            指标、bootstrap 和验收结果
predictions/latest_forecast.npz    三时效稠密预测及域外计数
predictions/latest_forecast.json   可读预测摘要
```

## 项目边界

- 当前是闭集任务，不预测未知的未来发射、退役或新接入对象。
- 两体 Kepler 是可由现有 Cartesian TEME 状态直接构造的因果基线，不包含 J2、阻力、机动或空间天气。学习模型必须在锁定测试中证明稳定增益，否则回退物理基线。
- 若 CSV 标签由某个传播器长期生成，低误差首先表示模型复现该传播器，不能自动等同于真实轨道预报精度。
