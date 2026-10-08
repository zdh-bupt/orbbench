#!/usr/bin/env bash
cd "$(dirname "$0")"
while pgrep -f "pip install.*torch==2.11" >/dev/null; do sleep 20; done
sleep 5
.venv_cuda/bin/pip install -q sgp4 astropy pyarrow scipy scikit-learn "chronos-forecasting==2.3.2" >> logs/venv_cuda_install.log 2>&1; echo "deps exit $?" >> logs/venv_cuda_install.log
.venv_cuda/bin/python -c "import torch, sgp4, astropy, pyarrow, scipy, sklearn, chronos; print('deps ok; torch', torch.__version__, 'cuda', torch.cuda.is_available())" >> logs/venv_cuda_install.log 2>&1
echo "DEPS DONE" >> logs/venv_cuda_install.log
