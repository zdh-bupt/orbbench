from __future__ import annotations

"""基于状态缓存构建闭合集群的月度训练分片。

脚本按配置时间窗切分分片产物，并写出供训练/评估脚本使用的清单。
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 允许从仓库根目录直接调用脚本。
sys.path.insert(0, str(ROOT / "src"))

from starlink_forecast.closed_cohort import build_closed_cohort_shards
from starlink_forecast.config import load_config


def parse_args() -> argparse.Namespace:
    # 可选时间窗与月份上限，便于 smoke/本地调试。
    parser = argparse.ArgumentParser(description="Build monthly closed-cohort LEO training shards")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--start", help="Inclusive UTC origin interval")
    parser.add_argument("--end", help="Exclusive UTC origin interval")
    parser.add_argument("--max-months", type=int)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable stderr progress bar")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # 对局部运行上限做严格正数校验。
    if args.max_months is not None and args.max_months <= 0:
        raise SystemExit("--max-months must be positive")

    # 构建分片清单，供后续按时间切分数据集。
    manifest = build_closed_cohort_shards(
        load_config(args.config),
        start=args.start,
        end=args.end,
        overwrite=args.overwrite,
        max_months=args.max_months,
        progress=not args.no_progress,
    )

    # 输出精简摘要到标准输出。
    summary = {
        "usable_shard_count": manifest["usable_shard_count"],
        "total_origin_count": manifest["total_origin_count"],
        "satellite_count": len(manifest["satellite_ids"]),
        "manifest": manifest["manifest_path"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    # “执行成功但产物为空”使用退出码 2 区分。
    return 0 if manifest["usable_shard_count"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
