# Environment configuration

This file lists, per component, the hardware and software that produced the numbers in Table 4, the package
pins, and the environment variables the code reads. Exact package sets were exported with `pip freeze` from the
virtual environments that ran the experiments; where a sub-project did not record versions this is stated.

## 1. Hardware and operating system of the recorded runs

| Component | Machine / accelerator | Notes |
|---|---|---|
| `pipeline/` CPU stages (S0–S9, S8b–S8l) | Linux x86_64 (kernel 7.0), 7 CPU cores, 61 GB RAM host (about 19 GB were available to the pipeline) | `ORBPIPE_WORKERS=6`, `OMP_NUM_THREADS=1` (`run_all.sh`); chain ablation / e2e used `OMP_NUM_THREADS=8` |
| `pipeline/` stage S8m (`orbpipe.deep_forecast`) | same host, NVIDIA GeForce RTX 5090 (32 GB), CUDA 12.8 | AMP (float16) training, bfloat16 zero-shot Chronos / Chronos-2, `HF_ENDPOINT=https://hf-mirror.com` |
| `t1_phase_classification/` | GPU host with CUDA 12.8 (torch 2.11.0+cu128) | recorded in the sub-project README |
| `t3_space_weather/` | CPU only (`DEVICE = "cpu"` in `run.py`) | amazon/chronos-2 weights from the Hugging Face hub |
| `t4_deep_models/` (original eight-model study) | trained on another machine, not recorded | the Table 4 rows were re-trained on the RTX 5090 by `orbpipe.deep_forecast` |
| `t5_population_forecasting/` | Windows host (`configs/default.yaml` uses `F:\...` paths, `run_local_prepare.ps1`) | GPU not recorded |

## 2. Python and package versions

| Component | Python | Pin file | Key packages |
|---|---|---|---|
| pipeline, CPU | 3.14.4 | `requirements/pipeline-cpu.txt` | numpy 2.5.2, pandas 3.0.5, scipy 1.18.1, scikit-learn 1.9.0, astropy 8.0.1, sgp4 2.27, pyarrow 25.0.1, torch 2.14.0+cpu, stable-baselines3 2.9.0, gymnasium 1.3.0, matplotlib 3.11.1 |
| pipeline, GPU stage | 3.14.4 | `requirements/pipeline-cuda.txt` | torch 2.11.0+cu128, chronos-forecasting 2.3.2, transformers 5.17.0, accelerate 1.15.0, numpy 2.5.3, pandas 3.0.6, scikit-learn 1.9.1 |
| T1 | 3.13.11 | `requirements/t1_phase.txt` | torch 2.11.0+cu128, numpy 2.5.2, pandas 3.0.5, scikit-learn 1.9.0; scipy, joblib, ruptures, matplotlib unpinned |
| T3 | not recorded | `requirements/t3_space_weather.txt` | chronos-forecasting (Chronos2Pipeline API), pandas, numpy, pyarrow, scipy, matplotlib |
| T4 eight-model study | not recorded | `requirements/t4_deep_models.txt` | torch, numpy, pandas, matplotlib; protocol fixes chronos-forecasting 2.2.2 |
| T5 | >= 3.11 (`pyproject.toml`) | `requirements/t5_population.txt` | numpy, pandas, pyarrow, PyYAML, torch, astropy, scipy, matplotlib |

`scripts/make_table4.py` needs only pandas (any 2.x or 3.x).

## 3. Creating the environments

```bash
# CPU pipeline + table assembly
python -m venv .venv && . .venv/bin/activate
pip install -r requirements/pipeline-cpu.txt
python scripts/make_table4.py --check

# GPU stage (deep forecasters and zero-shot Chronos); a separate venv is recommended because torch differs
python -m venv .venv_cuda && .venv_cuda/bin/pip install -r requirements/pipeline-cuda.txt
```
`environment.yml` builds the CPU environment with conda (`conda env create -f environment.yml`).
The Chronos weights (`amazon/chronos-t5-large`, `amazon/chronos-2`) are downloaded from the Hugging Face hub on first
use; set `HF_ENDPOINT` to a mirror if the hub is not reachable.

## 4. Environment variables read by the code

| Variable | Read by | Default | Purpose |
|---|---|---|---|
| `ORBBENCH_DATA_ROOT` | `pipeline/orbpipe/config.py` | `/mnt/data` | root of the raw data tree (TLE archive, L2 trajectories, POD, DISCOS, Starlink ephemerides); see `DATA.md` |
| `ORBBENCH_RELEASE_ROOT` | `config.py`, `t3_space_weather/run.py` | parent of `pipeline/` | location of the sub-project directories (`t1_…`, `t3_…`, `t4_…`, `t5_…`, `t6_t7_…`, `data/`) |
| `ORBBENCH_DEEP_SRC` | `orbpipe/deep_forecast.py` | `<release>/t4_deep_models/src` | directory with `models.py` and `representation.py` of the eight-model study |
| `ORBBENCH_SW_PARQUET`, `ORBBENCH_SW_FLUX_PARQUET` | `t3_space_weather/run.py` | `<release>/data/space_weather/*.parquet` | daily space-weather table and flux measurements |
| `ORBBENCH_GFZ_TRUTH` | `t3_space_weather/run.py` | `<release>/t3_space_weather/GFZ_true_20250625_20250701.txt` | GFZ definitive rows for the formal-forecast validation |
| `ORBPIPE_WORKERS` | `config.py` | 6 | process-pool size of the pipeline stages |
| `E2E_N_OBJ` | `orbpipe/e2e.py` | 1500 | cohort size of the end-to-end run (Table 4 used 1500) |
| `DEEP_N_OBJ`, `DEEP_EPOCHS`, `CHRONOS_T5_BATCH` | `orbpipe/deep_forecast.py` | 120, 30, 2 | training objects, max epochs, Chronos-T5 batch size |
| `HF_ENDPOINT` | `deep_forecast.py`, Hugging Face hub | `https://hf-mirror.com` (set in `deep_forecast.py`) | hub mirror for Chronos weights |
| `OMP_NUM_THREADS`, `MKL_NUM_THREADS`, `OPENBLAS_NUM_THREADS` | shell scripts | 1 (run_all) / 8 (chain, e2e, deep) | BLAS thread caps |

## 5. Recorded stage runtimes (`pipeline/status.json`)

| Stage | Module | Wall time |
|---|---|---|
| S0 staging | `stage0_staging` | 54 min |
| S0b Starlink staging | `stage_starlink` | 41 s |
| S1 T3 weather baselines | `t3_weather` | 2 s |
| S2 T1 phase | `t1_phase` | 108 s |
| S3 T2 maneuver | `t2_maneuver` | 99 s |
| S4 T4 forecast + UQ | `t4_forecast` | 95 s |
| S5 T5 evolution | `t5_evolution` | 30 min |
| S6 T6 warning | `t6_warning` | 25 min |
| S7 T7 avoidance | `t7_avoidance` | 82 s |
| S8 experiments E0–E4 | `experiments` | 8 s |
| S8b E3 leave-one-out | `e3_loo` | 11 min |
| S8c challenge evidence | `challenge_evidence` | 74 s |
| S8d proof P1–P5 | `proof` | 164 s |
| S8e chain proof C1–C4 | `chain_proof` | 9 min |
| S8f Starlink chain S1–S4 | `starlink_chain` | 25 min |
| S8g chain ablation X1–X4 | `chain_ablation` | 9 min |
| S8h/S8i summaries, external datasets | `exp_summary`, `external` | 89 s, 7 s |
| S8j end-to-end T1→T7 (1,500 objects) | `e2e` | 27 min |
| S8k upstream connections | `upstream_chain` | 58 s |
| S8l detail table (Table 4 T2/T4/T6/T7 blocks) | `detail_table` | 242 s |
| S8m deep forecasters (GPU) | `deep_forecast` | 61 min |
| S9 report | `report` | 2 s |
