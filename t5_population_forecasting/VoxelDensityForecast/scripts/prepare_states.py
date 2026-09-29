from __future__ import annotations

"""从原始卫星状态向量文件构建小时级状态缓存。

该脚本是预处理入口：把已审计的原始 CSV 转换为因果 TEME/ITRS 缓存产物，
并输出简要执行摘要。
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 支持独立运行脚本时正确导入项目包。
sys.path.insert(0, str(ROOT / "src"))

from starlink_forecast.config import load_config
from starlink_forecast.state_cache import prepare_state_caches


def parse_args() -> argparse.Namespace:
    # CLI 参数与 prepare_state_caches 保持对应，便于复现流水线。
    parser = argparse.ArgumentParser(description="Build causal hourly TEME/ITRS state caches for selected LEO satellites")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--data-root", help="Override paths.raw_state_dir")
    parser.add_argument("--selection-file", help="Optional TXT/CSV/JSON satellite ID list")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--no-source-hash", action="store_true", help="Skip SHA-256 for a quick smoke run")
    parser.add_argument("--no-progress", action="store_true", help="Disable stderr progress bars")
    parser.add_argument(
        "--chunksize-rows",
        type=int,
        help="Runtime CSV chunk size; lower values reduce memory without changing cache compatibility",
    )
    parser.add_argument(
        "--coordinate-batch-rows",
        type=int,
        help="Runtime coordinate-transform batch size; lower values reduce memory without changing cache compatibility",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # 显式做命令行校验，保证失败行为可预期。
    for name, value in (
        ("--max-files", args.max_files),
        ("--chunksize-rows", args.chunksize_rows),
        ("--coordinate-batch-rows", args.coordinate_batch_rows),
    ):
        if value is not None and value <= 0:
            raise SystemExit(f"{name} must be positive")

    # 执行缓存构建（配置 + 可选本地覆盖）。
    manifest = prepare_state_caches(
        load_config(args.config),
        data_root=args.data_root,
        selection_file=args.selection_file,
        overwrite=args.overwrite,
        max_files=args.max_files,
        hash_sources=not args.no_source_hash,
        progress=not args.no_progress,
        chunksize_rows=args.chunksize_rows,
        coordinate_batch_rows=args.coordinate_batch_rows,
    )

    # 输出高价值摘要，便于终端与 CI 日志查看。
    summary = {
        "processed_satellite_count": manifest["processed_satellite_count"],
        "status_counts": manifest["audit"]["status_counts"],
        "manifest": manifest["manifest_path"],
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # 任一卫星校验/处理失败时返回非零退出码。
    errors = [entry for entry in manifest["satellites"] if entry["status"] in {"invalid", "processing_error"}]
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
