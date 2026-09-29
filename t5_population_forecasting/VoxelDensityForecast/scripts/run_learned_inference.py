from __future__ import annotations

"""单样本学习推理的 Torch 子进程 worker。

该辅助脚本由 `predict.py` 通过子进程调用，用于隔离模型运行时影响。
脚本读取序列化 NPZ 样本，加载 checkpoint 执行推理，按需做温度标定，
最终写出期望计数 NPZ。
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
# 支持脚本直跑时导入项目包。
sys.path.insert(0, str(ROOT / "src"))

from starlink_forecast.config import load_config
from starlink_forecast.training import checkpoint_run_config, model_from_config


def parse_args() -> argparse.Namespace:
    # 该脚本总是被程序调用，因此关键参数均强制必填。
    parser = argparse.ArgumentParser(description="Internal Torch-only inference worker")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # 分片语义仍由原始配置校验；模型结构从 checkpoint 的运行配置恢复。
    config = load_config(args.config)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    try:
        run_config = checkpoint_run_config(checkpoint, config)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model_from_config(run_config).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    # 读取序列化单样本输入并执行前向推理。
    with np.load(args.input, allow_pickle=False) as sample:
        with torch.no_grad():
            output = model(
                torch.from_numpy(sample["features"]).unsqueeze(0).to(device),
                torch.from_numpy(sample["physics_prior"]).unsqueeze(0).to(device),
                torch.from_numpy(sample["physics_prior_outside"]).unsqueeze(0).to(device),
                torch.from_numpy(sample["future_total"]).unsqueeze(0).to(device),
            )
        temperatures = (
            sample["temperatures"]
            if "temperatures" in sample.files
            else np.ones(output.voxel_probability.shape[1], dtype=np.float32)
        )
        # 在联合类别空间逐时效重标定后，换算回计数。
        voxel_counts: list[torch.Tensor] = []
        outside_counts: list[torch.Tensor] = []
        for lead_index, temperature in enumerate(temperatures):
            categories = torch.cat(
                (
                    output.voxel_probability[:, lead_index].flatten(start_dim=1),
                    output.outside_probability[:, lead_index],
                ),
                dim=1,
            )
            calibrated = torch.softmax(
                torch.log(categories.clamp_min(1.0e-12)) / float(temperature), dim=1
            )
            total = torch.from_numpy(sample["future_total"][[lead_index]]).to(device)
            voxel_counts.append(
                calibrated[:, :-2].reshape_as(output.voxel_probability[:, lead_index])
                * total[:, None, None, None]
            )
            outside_counts.append(calibrated[:, -2:] * total[:, None])

    # 为 `predict.py` 落盘栅格内/外部桶期望计数。
    np.savez_compressed(
        args.output,
        expected_count=torch.stack(voxel_counts, dim=1)[0].cpu().numpy().astype(np.float32),
        outside_expected_count=torch.stack(outside_counts, dim=1)[0].cpu().numpy().astype(np.float32),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
