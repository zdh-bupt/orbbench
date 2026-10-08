"""Central configuration: data paths, cohort, evaluation window, task defaults."""
from __future__ import annotations
from pathlib import Path
import os

ROOT = Path(__file__).resolve().parents[1]          # .../pipeline
DATA = ROOT / "data"
RESULTS = ROOT / "results"
FIGURES = ROOT / "figures"
LOGS = ROOT / "logs"
for _d in (DATA, RESULTS, FIGURES, LOGS):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------- read-only sources ----------------
# Root of the raw OrbBench data tree (see ../../DATA.md). Override with ORBBENCH_DATA_ROOT when the raw data live elsewhere.
MNT = Path(os.environ.get("ORBBENCH_DATA_ROOT", "/mnt/data"))
# Root of this release (the directory that contains pipeline/, t1_phase_classification/, t3_space_weather/, ...).
RELEASE_ROOT = Path(os.environ.get("ORBBENCH_RELEASE_ROOT", str(ROOT.parent)))
PRUNED_DIR = MNT / "trajectory_data/Pruned/Pruned_results"              # <dir>/<norad>_pruned.csv, 1-min TEME (nearest-TLE SGP4)
MULTI_DIR = MNT / "trajectory_data/Multi/Multi_results"                  # <dir>/<norad>multi.csv,   1-min TEME (MTF)
TLE_ZIP = MNT / "TLEs/TLEs_NORAD.zip"                            # raw TLE archive, one txt per NORAD
STEP6_PATCH_TLE = MNT / "processed_data/step6_patch_1164_ref20251231/tle"   # PAF-filtered TLEs for 1164 objects
LABELS_CSV = MNT / "processed_data/tle_analysis_bundle/07_type_orbit_onorbit_time_violin/outputs/type_orbit_onorbit_time_per_norad.csv"
GAPS_CSV = MNT / "processed_data/tle_analysis_bundle/08_type_orbit_update_interval_violin/outputs/type_orbit_update_interval_per_norad.csv"
DISCOS_CSV = MNT / "Discos/discos_all_complete.csv"
POD_ROOT = MNT / "ai4ssa_datasets/sentinel系列pod已整理数据"
MAN_DIR = POD_ROOT / "公开机动数据"
SW_PARQUET = MNT / "ai4ssa_datasets/数据集格式样例/space_weather/space_weather_daily.parquet"
SW_TXT_TAR = MNT / "空间天气数据.tar.gz"
CDM_CSV = RELEASE_ROOT / "t6_t7_cdm_avoidance/cdm_data.csv"      # 199 ESA Kelvins CDM events (shipped with the release)
STARLINK_EPH_DIR = MNT / "Starlink_Ephemeris"
SUMMARY_ROOT = RELEASE_ROOT                                       # task sub-projects live next to pipeline/ in the release

# reported (external) result files to import into benchmark tables
REPORTED = {
    "forecast_8models": SUMMARY_ROOT / "t4_deep_models/result.csv",
    "weather_chronos2_backtest": SUMMARY_ROOT / "t3_space_weather/metrics/backtest_metrics_summary.json",
    "weather_chronos2_formal": SUMMARY_ROOT / "t3_space_weather/metrics/formal_forecast_metrics.csv",
    "phase_clustering": SUMMARY_ROOT / "t1_phase_classification/reports/observable_regime_feature_labels_v2/retest/summary_by_system.csv",
    "evolution_voxel": SUMMARY_ROOT / "t5_population_forecasting/results_1h/一小时预测结果表格.csv",
}

# ---------------- evaluation window & cohort ----------------
WINDOW_START = "2025-01-01T00:00:00Z"   # space-weather table ends 2025-06-24 -> W = 2025 H1 (175 d)
WINDOW_END = "2025-06-24T00:00:00Z"
TRAIN_END = WINDOW_START                     # everything before the window may be used for fitting/calibration
STAGE_CADENCE_MIN = 10                       # cohort trajectory cadence staged for the window
FORECAST_HORIZONS_H = [0.5, 1, 4, 8, 12, 24]

# Sentinel truth cohort: name -> (norad, pod subdir, maneuver file or None)
SENTINEL_START = "2019-01-01T00:00:00Z"      # long truth window for Sentinel-based experiments (train 2019-23, test 2024-25H1)
SENTINEL_TEST_START = "2024-01-01T00:00:00Z"
SENTINELS = {
    "S1A": (39634, "s1a(2014-2024)", "s1a.man"),
    "S1B": (41456, "s1b(2016-2022)", None),
    "S2A": (40697, "s2a(2023-2025)", "s2a.man"),
    "S2B": (42063, "s2b(2023-2026)", None),
    "S3A": (41335, "s3a（2022-2024）", "s3a.man"),
    "S3B": (43437, "s3b（2022-2024）", None),
}

# physical constants
MU_KM3_S2 = 398600.4418
RE_KM = 6378.137

N_WORKERS = int(os.environ.get("ORBPIPE_WORKERS", "6"))
SEED = 20260913
