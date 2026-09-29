from __future__ import annotations

"""生成最新一期闭合集群预测产物（用于部署/消费）。

脚本从因果缓存状态构建推理样本，根据部署决策选择“学习残差模型”
或“物理先验模型”，保存 NPZ 预测张量，并输出可读的 JSON 摘要。
"""

import argparse
from datetime import datetime, timezone
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
# 支持脚本直跑，同时可从 src/ 导入项目模块。
sys.path.insert(0, str(ROOT / "src"))

from starlink_forecast.config import grid_from_config, load_config, resolve_project_path
from starlink_forecast.inference import build_inference_sample, normalized_physics_prediction


def parse_args() -> argparse.Namespace:
    # 提供运行时覆盖参数，避免直接改默认配置文件。
    parser = argparse.ArgumentParser(description="Forecast a closed LEO cohort from the latest cached causal state")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--artifact-dir", help="Directory containing an experiment checkpoint and decision")
    parser.add_argument("--checkpoint", help="Defaults to <artifact_dir>/model-best.pt")
    parser.add_argument("--decision", help="Defaults to <artifact_dir>/deployment_decision.json")
    parser.add_argument("--origin-time", help="UTC anchor; defaults to the latest usable cache time")
    parser.add_argument("--force-learned", action="store_true", help="Research only: ignore a missing/failed acceptance decision")
    parser.add_argument("--output", default="predictions/latest_forecast.npz")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 加载运行配置，并在给定起报时刻构建推理样本。
    config = load_config(args.config)
    sample = build_inference_sample(config, origin_time=args.origin_time)
    artifact_dir = resolve_project_path(
        config,
        args.artifact_dir if args.artifact_dir is not None else config["paths"]["artifact_dir"],
    )
    decision_path = (
        resolve_project_path(config, args.decision)
        if args.decision
        else artifact_dir / "deployment_decision.json"
    )
    decision = (
        json.loads(decision_path.read_text(encoding="utf-8"))
        if decision_path.is_file()
        else {"accepted": False, "selected_model": "causal_physics_prior"}
    )
    # 仅在评估通过（或研究模式强制）时启用学习模型。
    use_learned = bool(args.force_learned or decision.get("accepted", False))
    target = resolve_project_path(config, args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    if use_learned:
        # 学习模型推理放在子进程中执行，隔离 Torch 运行时风险。
        checkpoint_path = (
            resolve_project_path(config, args.checkpoint)
            if args.checkpoint
            else artifact_dir / "model-best.pt"
        )
        input_path = target.parent / f".{target.stem}.{os.getpid()}.inference-input.npz"
        output_path = target.parent / f".{target.stem}.{os.getpid()}.inference-output.npz"
        try:
            # 写入单样本推理输入，供 worker 脚本读取。
            np.savez_compressed(
                input_path,
                features=sample["features"],
                physics_prior=sample["physics_prior"],
                physics_prior_outside=sample["physics_prior_outside"],
                future_total=sample["future_total"],
                temperatures=np.asarray(
                    [
                        float(decision.get("temperatures", {}).get(str(int(horizon)), 1.0))
                        for horizon in sample["horizons"]
                    ],
                    dtype=np.float32,
                ),
            )
            completed = subprocess.run(
                [
                    sys.executable,
                    str(ROOT / "scripts" / "run_learned_inference.py"),
                    "--config",
                    str(Path(args.config).resolve()),
                    "--checkpoint",
                    str(checkpoint_path),
                    "--input",
                    str(input_path),
                    "--output",
                    str(output_path),
                ],
                check=False,
            )
            if completed.returncode:
                raise SystemExit(f"Learned inference subprocess failed with exit code {completed.returncode}")
            # 读取学习残差模型输出的期望计数。
            with np.load(output_path, allow_pickle=False) as learned:
                expected_count = learned["expected_count"].copy()
                outside_expected_count = learned["outside_expected_count"].copy()
        finally:
            # 无论成功失败，都清理临时输入输出文件。
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)
        selected_model = "learned_physics_residual_forced" if args.force_learned and not decision.get("accepted") else "learned_physics_residual"
    else:
        # 回退路径：仅使用归一化物理先验计算期望。
        expected_count, outside_expected_count = normalized_physics_prediction(sample)
        selected_model = "causal_physics_prior"

    grid = grid_from_config(config)
    np.savez_compressed(
        target,
        schema_version=np.int16(config["dataset"]["schema_version"]),
        selected_model=np.str_(selected_model),
        origin_time_ns=sample["origin_time_ns"],
        horizons_hours=sample["horizons"],
        satellite_ids=sample["satellite_ids"],
        expected_count=expected_count.astype(np.float32),
        outside_expected_count=outside_expected_count.astype(np.float32),
        grid_shape=np.asarray(grid.shape, dtype=np.int16),
        height_min_km=np.float32(grid.height_min_km),
        height_step_km=np.float32(grid.height_step_km),
        lat_step_deg=np.float32(grid.lat_step_deg),
        lon_step_deg=np.float32(grid.lon_step_deg),
    )

    # 生成简洁 JSON 摘要，便于观测与下游校验。
    origin = np.datetime_as_string(np.datetime64(int(sample["origin_time_ns"]), "ns"), unit="s") + "Z"
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "origin_time_utc": origin,
        "selected_model": selected_model,
        "member_count": len(sample["satellite_ids"]),
        "satellite_ids": sample["satellite_ids"].tolist(),
        "horizons": [
            {
                "hours": int(horizon),
                "in_grid_expected_count": float(expected_count[index].sum()),
                "outside_below_expected_count": float(outside_expected_count[index, 0]),
                "outside_above_expected_count": float(outside_expected_count[index, 1]),
                "total_expected_count": float(expected_count[index].sum() + outside_expected_count[index].sum()),
            }
            for index, horizon in enumerate(sample["horizons"])
        ],
        "npz": str(target),
    }
    summary_path = target.with_suffix(".json")
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
