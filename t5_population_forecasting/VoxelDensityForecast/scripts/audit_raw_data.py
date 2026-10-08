from __future__ import annotations

"""在下游预处理前，对原始 TEME 状态 CSV 做快速审计。

脚本会扫描配置中的原始状态目录，调用
`starlink_forecast.raw.audit_state_directory` 完成文件体检与 LEO 分类，
输出完整 JSON 报告；当出现无效文件或缺失目标卫星 ID 时返回非零退出码。
"""

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# 允许从仓库根目录直接运行脚本。
sys.path.insert(0, str(ROOT / "src"))

from starlink_forecast.config import leo_from_config, load_config, resolve_project_path
from starlink_forecast.raw import audit_state_directory, read_selection_file


def parse_args() -> argparse.Namespace:
    # CLI 参数与配置键保持一致，便于本地覆盖。
    parser = argparse.ArgumentParser(description="Quickly audit pruned TEME CSV files and classify LEO satellites")
    parser.add_argument("--config", default=str(ROOT / "configs" / "default.yaml"))
    parser.add_argument("--data-root", help="Override paths.raw_state_dir")
    parser.add_argument("--selection-file", help="Optional TXT/CSV/JSON satellite ID list")
    parser.add_argument("--max-files", type=int)
    parser.add_argument("--output", default="reports/raw_data_audit.json") 
    parser.add_argument("--no-progress", action="store_true", help="Disable stderr progress bar")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    # 显式校验参数，便于自动化场景快速失败。
    if args.max_files is not None and args.max_files <= 0:
        raise SystemExit("--max-files must be positive")

    # 解析配置与可选覆盖参数，并统一成项目绝对路径。
    config = load_config(args.config)
    selection = config["selection"]
    selection_path = args.selection_file or selection.get("selection_file")
    if selection_path and not Path(selection_path).is_absolute():
        selection_path = resolve_project_path(config, selection_path)
    raw_root = Path(args.data_root or config["paths"]["raw_state_dir"])
    if not raw_root.is_absolute():
        raw_root = resolve_project_path(config, str(raw_root))

    # 执行原始数据审计与 LEO 分类。
    audit = audit_state_directory(
        raw_root,
        pattern=str(selection["file_pattern"]),
        recursive=bool(selection.get("recursive", True)),
        satellite_id_regex=str(selection["satellite_id_regex"]),
        leo=leo_from_config(config),
        selected_ids=read_selection_file(selection_path),
        max_files=args.max_files,
        progress=not args.no_progress,
    )

    # 落盘完整审计结果，便于追溯与排障。
    target = resolve_project_path(config, args.output)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")

    # 打印精简摘要，方便人工查看和 CI 日志检索。
    summary = {
        "discovered_file_count": audit["discovered_file_count"],
        "inspected_file_count": audit["inspected_file_count"],
        "status_counts": audit["status_counts"],
        "missing_selected_ids": audit["missing_selected_ids"],
        "output": str(target),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))

    # CI 闸门：存在无效文件或缺失目标 ID 时返回失败。
    return 1 if audit["status_counts"].get("invalid", 0) or audit["missing_selected_ids"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
