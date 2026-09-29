#!/usr/bin/env bash
# Serial runner for the 5.2/5.3 consolidation with capped threads (avoids OpenMP oversubscription).
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8
./.venv/bin/python -m orbpipe.chain_ablation --force --all > logs/nohup_chain_ablation.out 2>&1
./.venv/bin/python -m orbpipe.exp_summary --force > logs/nohup_exp_summary.out 2>&1
./.venv/bin/python -m orbpipe.external --force > logs/nohup_external.out 2>&1
