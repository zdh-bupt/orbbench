#!/usr/bin/env bash
# Waits for the CUDA venv dependencies, then runs the deep-forecast stage on the GPU.
cd "$(dirname "$0")"
while ! grep -q "DEPS DONE" logs/venv_cuda_install.log 2>/dev/null; do sleep 30; done
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 HF_ENDPOINT=https://hf-mirror.com HF_HUB_ENABLE_HF_TRANSFER=0
echo "[$(date -u +%FT%TZ)] deep_forecast start"
.venv_cuda/bin/python -m orbpipe.deep_forecast > logs/nohup_deep_forecast.out 2>&1
echo "[$(date -u +%FT%TZ)] deep_forecast exit $?"
