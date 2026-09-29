# OrbBench Table 4 — reproducibility package

Code, result files, environment pins and data notes behind **Table 4: Task-level comparisons and independent
versus unified pipeline execution** of the OrbBench paper (seven space-traffic-management tasks, T1–T7, plus the
independent-vs-unified pipeline comparison).

The package has two layers:

* `pipeline/` — the unified STM pipeline (`orbpipe` Python package). It runs all seven tasks on the same data, in
  independent (A0) and chained/unified (A1) mode, and produces the T2, T4 (PICP and deep-model position errors),
  T6 and T7 blocks of Table 4 together with the results reported in the paper's Sections 5.2–5.3.
* `t1_…`, `t3_…`, `t4_…`, `t5_…`, `t6_t7_…` — the task sub-projects whose numbers enter Table 4 either directly
  (T1, T3, T5) or as inputs to the pipeline (T4 model code, CDM events).

`scripts/make_table4.py --check` rebuilds every cell of Table 4 from the shipped result files and compares it with
the manuscript; on this release it reports `ALL CELLS MATCH`.

## 1. Where every number in Table 4 comes from

| Block | Rows | Source file in this package | Producing code | Rerun |
|---|---|---|---|---|
| **T1 Phase classification** | Constant stable, PCA + KMeans, PELT + KMeans, Transformer + KMeans (Macro-F1, ARI) | `t1_phase_classification/reports/observable_regime_feature_labels_v2/retest/summary_by_system.csv` (split = test) | `t1_phase_classification/scripts/retest_feature_label_clustering_v2.py` (labels: `build_tle_observable_feature_labels_v2.py`; metrics: `satlife_observable/metrics.py`) | needs the feature-label dataset and fitted models, which the sub-project does not ship (see `DATA.md` §4) |
| **T2 Maneuver detection** | F1 and false alarms, Sentinel-1A/2A/3A, independent vs unified | `pipeline/results/detail_table.csv` (block `T2 Maneuver`), from `pipeline/results/uc_t2_sentinel.csv` | `orbpipe/t2_maneuver.py` (detectors), `orbpipe/upstream_chain.py` (A0 = SGP4-residual detector; A1 = threshold conditioned on T1 phase and T3 Ap), `orbpipe/detail_table.py` | `python -m orbpipe.upstream_chain --force && python -m orbpipe.detail_table` (needs TLE archive + maneuver logs) |
| **T3 Space weather forecasting** | 3-day / 7-day MAE of F10.7, Kp, Ap for 30/60/90-day history | `t3_space_weather/metrics/backtest_metrics_summary.json` (`ctx{30,60,90}_pred{3,7}` → `MAE_mean`) | `t3_space_weather/run.py` (zero-shot amazon/chronos-2, 10 rolling backtest windows per configuration) | `cd t3_space_weather && python run.py` (CPU, inputs shipped) |
| **T4 Orbit forecasting — PICP95** | 95 % coverage of the 24 h along-track interval, per satellite and pooled, independent (σ-age or fixed σ) vs unified (σ-age × g(T2) / g(T1,T2,T3)) | `pipeline/results/detail_table.csv` (block `T4 Forecast`) | `orbpipe/detail_table.py` on `t4_forecast_cases.parquet` (Sentinel, POD truth), `s1_starlink_cases.parquet` (Starlink, MEME truth) and the e2e cohort | `python -m orbpipe.detail_table` (needs staged POD / MEME / cohort data) |
| **T4 Orbit forecasting — position error 1 h / 24 h** | LSTM, GRU, DLinear, TCN, Informer, FEDformer, Chronos, Chronos-2 (3D P50, km) | `pipeline/results/deep_forecast_table_1h.csv`, `deep_forecast_table_24h.csv` (identical copies in `tables/table4_position_error_p50_*.csv`) | `orbpipe/deep_forecast.py`, which imports the model definitions from `t4_deep_models/src/models.py` and the RTN representation from `t4_deep_models/src/representation.py`; protocol in `t4_deep_models/PROTOCOL_zh.md` | `.venv_cuda/bin/python -m orbpipe.deep_forecast` (GPU; checkpoints and per-window predictions shipped in `pipeline/data/deep/`) |
| **T5 Population forecasting** | Last-frame persistence, Kepler + CIC, Physics + U-Net (TV, regional relative MAE, +1 h) | `pipeline/results/t5_evolution_reported.csv` = `t5_population_forecasting/results_1h/一小时预测结果表格.csv` | `t5_population_forecasting/VoxelDensityForecast/scripts/{prepare_states,prepare_closed_cohort,train,evaluate,summarize_1h_evaluation}.py` | needs the Pruned L2 trajectories (5.5 TB) |
| **T6 Collision warning** | ACR, DCR, DAC by primary object type and altitude shell, independent vs unified | `pipeline/results/detail_t6_found_per_origin.csv` (ACR), `pipeline/results/detail_table.csv` block `T6 Warning` (DCR = `In budget (%)`, DAC = `Alerted`) | `orbpipe/e2e.py` (end-to-end T1→T7 on 1,500 LEO objects, 3 days, 5 decision epochs, 20 % screening budget, 30 injected dangerous conjunctions per epoch), `orbpipe/detail_table.py` | `E2E_N_OBJ=1500 python -m orbpipe.e2e --force && python -m orbpipe.detail_table` (needs the L2 cohort trajectories) |
| **T7 Collision avoidance** | DRC and total Δv by type and shell | `pipeline/results/detail_table.csv` block `T7 Avoidance` (DRC = `Resolved`, Δv = `dv`) | same run as T6 (`orbpipe/e2e.py`; controller shared with `orbpipe/chain_ablation.py`) | same as T6 |

Metric definitions used in the T6/T7 blocks (from `orbpipe/e2e.py` and `detail_table.py`):

* **ACR** — approach coverage rate: share of reference close approaches (< 10 km) of that group that are reached
  by the screening under a 20 % pair budget (A0: random pair order; A1: order from the T5 shell-density forecast).
* **DCR** — dangerous-conjunction coverage rate: share of the injected dangerous conjunctions (reference miss 20–1000 m)
  that fall inside the screening budget.
* **DAC** — dangerous conjunctions alerted (A0: predicted miss < 5 km; A1: predicted Pc ≥ 1e-4 from the T4 covariance).
* **DRC** — dangerous conjunctions resolved, i.e. made safe after the T7 maneuver (A0: distance rule to 2 km;
  A1: Pc-gated thrust down to Pc < 1e-5), judged with the reference geometry.
* **Δv_total** — summed Δv of all T7 maneuvers in the group, m/s.

Group sizes (`n` column of `detail_table.csv`): 40 payload, 15 rocket-body and 68 debris injected dangerous
conjunctions; 20 / 73 / 30 in the 300–800 / 800–1200 / 1200–2000 km shells; 123 in total.

## 2. Layout

```
.
├── README.md                 this file
├── ENVIRONMENT.md            hardware, Python/package versions, environment variables, stage runtimes
├── DATA.md                   raw sources, staged files, what is / is not shipped and where to get the rest
├── environment.yml           conda file for the CPU environment
├── requirements/             pip pins per component (pipeline-cpu, pipeline-cuda, t1, t3, t4, t5)
├── scripts/make_table4.py    rebuilds Table 4 from the result files; --check compares with the paper
├── tables/                   paper tables (CSV + LaTeX) and the assembled Table 4 (table4_assembled.md, *_assembled.csv)
├── data/space_weather/       daily F10.7/Kp/Ap table and flux measurements (input of T3 and of pipeline stage S0)
├── pipeline/                 unified STM pipeline
│   ├── orbpipe/              Python package: config, util, stage0_staging, t1_phase … t7_avoidance, experiments,
│   │                         chain_proof, chain_ablation, upstream_chain, starlink_chain, e2e, detail_table,
│   │                         deep_forecast, external, exp_summary, table5, report, monitor
│   ├── run_all.sh            stage runner (S0 … S9), scripts_*.sh: the exact invocations of the later stages
│   ├── DESIGN.md             pipeline design (tasks, inputs/outputs, chain edges, experiments)
│   ├── EXPERIMENT_DESIGN_zh.md  the two core experiment designs of the paper (Chinese)
│   ├── REPORT.md             auto-generated report of every experiment (all tables of Sections 5.2–5.3)
│   ├── status.json           stage status and wall times of the recorded run
│   ├── data/                 small staged inputs, trained deep-model checkpoints and predictions (README inside)
│   ├── results/              all result CSV/JSON files and the per-sample parquet files < 20 MB
│   ├── figures/              figures produced by orbpipe.report
│   └── logs/                 stage logs of the recorded run
├── t1_phase_classification/  T1 sub-project (rules, retest scripts, metrics, aggregate results, figures)
├── t3_space_weather/         T3 sub-project (run.py, GFZ truth rows, forecasts, backtests, metrics, figures)
├── t4_deep_models/           T4 eight-model study (src/, protocol, original result.csv, legacy checkpoints)
├── t5_population_forecasting/ T5 sub-project (VoxelDensityForecast package, +1 h evaluation slice, task summary)
└── t6_t7_cdm_avoidance/      199 ESA CDM events + the original physics / PPO avoidance use-case scripts
```

## 3. Quick start

```bash
python -m venv .venv && . .venv/bin/activate
pip install -r requirements/pipeline-cpu.txt

# 1. rebuild Table 4 from the shipped results and check it against the paper (seconds, no raw data needed)
python scripts/make_table4.py --check

# 2. re-run the T3 block (CPU, downloads amazon/chronos-2 on first use; ~1 h on 7 cores)
cd t3_space_weather && python run.py && cd ..

# 3. regenerate the report from the shipped results (no raw data needed)
cd pipeline && ../.venv/bin/python -m orbpipe.report
```

## 4. Re-running the pipeline

The pipeline is staged: `bash pipeline/run_all.sh` runs S0 → S9 in order and skips stages marked `done` in
`status.json`; individual stages take stage names (`bash run_all.sh S4 S8`), and every module also accepts
`--force` when run as `python -m orbpipe.<module>`. Stage S0 needs the raw data tree (`ORBBENCH_DATA_ROOT`,
default `/mnt/data`; see `DATA.md`). The later stages that produced Table 4 were launched with the shell
scripts kept next to `run_all.sh`:

| Table 4 block | Command (from `pipeline/`) | Data needed |
|---|---|---|
| T2 | `scripts_run_e2e.sh` first line: `python -m orbpipe.upstream_chain --force`, then `python -m orbpipe.detail_table` | TLE archive, Sentinel maneuver logs, `results/t3_weather_forecasts.parquet` |
| T4 PICP | `python -m orbpipe.t4_forecast`, `python -m orbpipe.chain_proof` (C1), `python -m orbpipe.starlink_chain` (S1), then `detail_table` | Sentinel POD, Starlink MEME, cohort L2 trajectories |
| T4 position error | `scripts_run_deep.sh` → `.venv_cuda/bin/python -m orbpipe.deep_forecast` | Pruned L2 CSVs for training (120 objects), POD / MEME / L2 truth for the test windows |
| T6, T7 | `scripts_run_e2e.sh` second line: `E2E_N_OBJ=1500 python -m orbpipe.e2e --force`, then `detail_table` | `data/c4_leo_1500_3d.npz` (built from the L2 CSVs), `data/e2e_tles_1500.parquet` (shipped) |

Evaluation protocol (from `orbpipe/config.py` and the module docstrings): cohort window 2025-01-01 → 2025-06-24
(fits and calibrations only on data before the window); Sentinel truth experiments train on 2019–2023 and test on
2024-01 → 2025-06; Starlink experiments train 2026-06-10 → 07-05 and test 07-05 → 08-23; the end-to-end run uses
1,500 LEO objects (300–2000 km), 3 days, decision epochs every 12 h, 24 h horizon. Random seeds are fixed in
`config.py` (`SEED = 20260913`), `deep_forecast.py` (42) and the sub-project configs.

## 5. Sub-projects

* **T1** `t1_phase_classification/` — TLE observable-state (phase) labels and the retrospective K=3 clustering
  comparison; `README.md`, `STARLINK_MOTION_PHASE_TEST_CASE.md` (label rules) and `figures/tle_state_use_case/RESULTS_TABLE.md`
  document the protocol. Row-level predictions, label data and weights are excluded by design.
* **T3** `t3_space_weather/` — Chronos-2 zero-shot forecasts of F10.7, Kp and Ap with 30/60/90-day context and
  3/7-day horizon; `README.md` documents the F10.7 observed/adjusted label swap in the source table and how the
  code corrects it. Outputs (forecasts, backtests, metrics, figures) are shipped.
* **T4** `t4_deep_models/` — the eight-model study: `PROTOCOL_zh.md` (data, RTN-relative representation, model
  and training configuration), `src/` (models, representation, data, train, evaluate, render), `result.csv`
  (original per-regime table) and `legacy_checkpoints/` (older-protocol weights; not the ones behind Table 4).
  The Table 4 rows were produced by `pipeline/orbpipe/deep_forecast.py`, which retrains the six supervised models
  with the code in `src/` and evaluates all eight models on the same windows as the SGP4 baselines.
* **T5** `t5_population_forecasting/` — closed-cohort LEO voxel density forecasting (`VoxelDensityForecast/README.md`,
  `TASK_SUMMARY_zh.md`); `results_1h/` holds the +1 h evaluation slice, the comparison figures and their data.
* **T6/T7** `t6_t7_cdm_avoidance/` — `cdm_data.csv` (199 ESA Kelvins events used by `chain_ablation.py` X1,
  `t6_warning.py` CDM risk and `t7_avoidance.py`) and the original use-case scripts: `jisuan.py` (conservative Pc,
  Patera method), `wulibipeng.py` (physics avoidance, 0.3 m/s² thrust), `zhinengbipeng.py` (PPO avoidance,
  stable-baselines3).

## 6. Notes for reviewers

* All numbers in Table 4 are read from files in this package; `scripts/make_table4.py --check` is the audit trail.
  ACR is compared at ±0.06 because `detail_t6_found_per_origin.csv` stores it rounded to one decimal, while the
  manuscript prints the unrounded value from the same run.
* Two `REPORT.md` sections mention `/mnt/data`; those are log lines of the recorded run and do not affect the code,
  which now reads `ORBBENCH_DATA_ROOT` / `ORBBENCH_RELEASE_ROOT` (see `ENVIRONMENT.md` §4).
* The GFZ Kp/Ap/F10.7 table is redistributed under CC BY 4.0 (sunspot numbers CC BY-NC 4.0); the CDM subset comes
  from the ESA Kelvins dataset (CC BY 4.0, doi:10.5281/zenodo.4463683). No licence for the OrbBench code itself is
  included in this package; add one before publishing the repository.
