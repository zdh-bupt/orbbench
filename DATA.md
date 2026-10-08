# Data: what is shipped, what is not, and where the code expects it

The pipeline reads raw sources through the paths in `pipeline/orbpipe/config.py`; all of them are resolved under
`ORBBENCH_DATA_ROOT` (default `/mnt/data`). Sizes are those of the copies used for the paper.

## 1. Raw sources (not shipped unless marked)

| `config.py` name | Expected path under `ORBBENCH_DATA_ROOT` | Content | Size | Shipped | How to obtain |
|---|---|---|---|---|---|
| `TLE_ZIP` | `TLEs/TLEs_NORAD.zip` | raw TLE archive, one `<norad>.txt` per object | 9.1 GB | no | Space-Track (space-track.org) TLE history per NORAD ID |
| `PRUNED_DIR` | `trajectory_data/Pruned/Pruned_results/<dir>/<norad>_pruned.csv` | OrbBench L2 trajectories, 1-min TEME, nearest-TLE SGP4 (`Epoch_UTC,X_m,Y_m,Z_m,VX_m_s,VY_m_s,VZ_m_s`) | 5.5 TB | no | OrbBench data release (L2) |
| `MULTI_DIR` | `trajectory_data/Multi/Multi_results/<dir>/<norad>multi.csv` | OrbBench L2 trajectories, multi-TLE fusion (MTF) | 4.9 TB | no | OrbBench data release (L2) |
| `STEP6_PATCH_TLE` | `processed_data/step6_patch_1164_ref20251231/tle/<norad>.txt` | PAF-filtered ("step6") TLEs of 1,164 objects | 64 MB | no | OrbBench data release (filtered TLE pool) |
| `LABELS_CSV`, `GAPS_CSV` | `processed_data/tle_analysis_bundle/07_…/type_orbit_onorbit_time_per_norad.csv`, `08_…/type_orbit_update_interval_per_norad.csv` | per-NORAD object type / orbit regime / on-orbit status and TLE update-interval statistics | 9 MB | no (merged into `pipeline/data/objects.csv`) | OrbBench data release (catalogue tables) |
| `DISCOS_CSV` | `Discos/discos_all_complete.csv` | ESA DISCOS object attributes (mass, dimensions, shape) | 9 MB | no (mass/area of the cohort are in `objects.csv`) | ESA DISCOS web service (discosweb.esoc.esa.int, registration required) |
| `POD_ROOT`, `MAN_DIR` | `ai4ssa_datasets/sentinel系列pod已整理数据/<sat>/…csv`, `…/公开机动数据/<sat>.man` | Sentinel-1A/1B/2A/2B/3A/3B precise orbit determination (POD) products and public maneuver logs | 12 GB | maneuver logs yes (`pipeline/data/sentinel_man_*.csv`); POD no | Copernicus POD Data Hub (ESA); maneuver files from the Sentinel operations pages |
| `SW_PARQUET` | `ai4ssa_datasets/数据集格式样例/space_weather/space_weather_daily.parquet` | daily F10.7 / Kp / Ap table (observed + predicted sections) | 0.8 MB | **yes**: `data/space_weather/` (plus `solar_flux_measurements.parquet`) | GFZ Potsdam / NOAA SWPC daily indices |
| `CDM_CSV` | — | 199 ESA Kelvins CDM events used by T6/T7 (`event_id, time_to_tca, true_risk, miss_distance, …`) | 0.25 MB | **yes**: `t6_t7_cdm_avoidance/cdm_data.csv` | subset of the Kelvins Collision Avoidance Challenge dataset (below) |
| `STARLINK_EPH_DIR` | `Starlink_Ephemeris/*.zip` | SpaceX public Starlink ephemerides (MEME J2000 with 6×6 covariance), daily zips 2026-06-10 → 2026-09-10 | 372 GB | no (`pipeline/data/starlink/summary.csv` only) | Space-Track public ephemeris files published by SpaceX |
| smoke package | `pipeline/data/smoke/orbbench_server_smoke/` | raw + step6 TLEs of the 20 Starlink truth satellites to 2026-08-24 | 32 MB | no | OrbBench data release |
| external: Kelvins CDM | `pipeline/data/external/kelvins_cdm/` | full Collision Avoidance Challenge dataset (E54b) | 221 MB zip | no | Zenodo, doi:10.5281/zenodo.4463683 (CC BY 4.0) |
| external: Shorten TLE benchmark | `pipeline/data/external/shorten_tle_benchmark/` | TLE observation benchmark with curated maneuver timestamps (E54a) | 6 MB | no | github.com/dpshorten/TLE_observation_benchmark_dataset |
| GFZ index table | `pipeline/data/Kp_ap_Ap_SN_F107_since_1932.txt` | definitive Kp/ap/Ap/SN/F10.7 since 1932 | 5 MB | **yes** | kp.gfz-potsdam.de (CC BY 4.0; sunspot numbers CC BY-NC 4.0) |

## 2. Staged intermediates (`pipeline/data/`)

`python -m orbpipe.stage0_staging` and `python -m orbpipe.stage_starlink` turn the raw sources into the staged
files. The small ones are shipped (see `pipeline/data/README.md`); the large ones are not:

| File | Size | Needed by |
|---|---|---|
| `traj_window.npz` | 1.7 GB | T1, T5, T6 (cohort trajectories at 10-min cadence over the evaluation window) |
| `c4_leo_600_3d.npz`, `c4_leo_1500_3d.npz` | 56 MB, 123 MB | `chain_proof` C4 / X3 (600 objects) and `e2e` (1,500 objects; Table 4 T6/T7 rows), built by `chain_proof.load_leo_subset` from the L2 CSVs |
| `t6_shell_700_1000_450_3d.npz` | 42 MB | `chain_proof` C2 |
| `sentinel_pod_<S>.parquet` | 41–347 MB each | T4 forecast / UQ, E3, P1/P2, X4 |
| `starlink/meme_<norad>.parquet` | 468 MB | `starlink_chain` S1–S4, X2, T4 Starlink columns |
| `deep/train/*.npy`, `deep/windows/*.npz` | 2.7 GB, 208 MB | `deep_forecast` training arrays and inference windows |

## 3. Per-sample result files above the size limit (`pipeline/results/`)

Every CSV/JSON result file is shipped, plus every parquet file below 20 MB. Three per-sample parquet files are not
shipped: `p1_thinning_samples.parquet` (159 MB), `p2_injection_samples.parquet` (46 MB) and
`t2_cohort_flags.parquet` (30 MB). None of them feeds Table 4; they back the P1/P2 proof experiments and the
cohort-wide T2 flags in `REPORT.md`.

## 4. Sub-project inputs

| Sub-project | Inputs required to re-run | Shipped |
|---|---|---|
| `t1_phase_classification/` | v2 observable-state feature-label dataset, fitted baseline models and transformer embeddings (`--dataset-dir`, `--baseline-root`, `--transformer-root` of `scripts/retest_feature_label_clustering_v2.py`) | no (the sub-project package intentionally excludes label data and model weights); aggregate outputs in `reports/` are shipped |
| `t3_space_weather/` | `data/space_weather/*.parquet`, `GFZ_true_20250625_20250701.txt`, Hugging Face weights `amazon/chronos-2` | yes (weights downloaded on first run) |
| `t4_deep_models/` | Pruned L2 CSVs (`PRUNED_DIR`) for `prepare_dataset.py`; the pipeline's `deep_forecast.py` uses the same source | no |
| `t5_population_forecasting/VoxelDensityForecast/` | Pruned L2 CSVs (`paths.raw_state_dir` in `configs/default.yaml`), `configs/satellite_ids.txt` | no; the +1 h evaluation slice is shipped in `results_1h/` |
| `t6_t7_cdm_avoidance/` | `cdm_data.csv` | yes |
