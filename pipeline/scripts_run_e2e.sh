#!/usr/bin/env bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 E2E_N_OBJ=1500
./.venv/bin/python -m orbpipe.upstream_chain --force > logs/nohup_upstream_chain.out 2>&1
./.venv/bin/python -m orbpipe.e2e --force > logs/nohup_e2e.out 2>&1
