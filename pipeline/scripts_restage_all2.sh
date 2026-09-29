#!/usr/bin/env bash
# Continuation runner: waits for the running e2e, then reruns the stages that depend on the re-staged trajectories.
cd "$(dirname "$0")"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 E2E_N_OBJ=1500 ORBPIPE_WORKERS=8
PY=./.venv/bin/python
while pgrep -f "orbpipe.e2[e] --forc[e]" >/dev/null; do sleep 30; done
echo "[$(date -u +%FT%TZ)] first e2e finished"
echo "[$(date -u +%FT%TZ)] restage traj_window (10-min, all objects)"; $PY -c "
from orbpipe.stage0_staging import build_objects, stage_trajectories
obj = build_objects(); stage_trajectories(obj)" > logs/nohup_restage.out 2>&1
echo "[$(date -u +%FT%TZ)] S5 t5"; $PY -m orbpipe.t5_evolution --force > logs/nohup_t5.out 2>&1
echo "[$(date -u +%FT%TZ)] S6 t6 (rebuilds shell cache)"; $PY -m orbpipe.t6_warning --force > logs/nohup_t6.out 2>&1
echo "[$(date -u +%FT%TZ)] chain_proof C4/C5 (rebuilds c4_leo_600)"; $PY -m orbpipe.chain_proof 45 --force > logs/nohup_chain_proof.out 2>&1
echo "[$(date -u +%FT%TZ)] chain_ablation X3/X4"; $PY -m orbpipe.chain_ablation --force > logs/nohup_chain_ablation.out 2>&1
echo "[$(date -u +%FT%TZ)] exp_summary"; $PY -m orbpipe.exp_summary --force > logs/nohup_exp_summary.out 2>&1
echo "[$(date -u +%FT%TZ)] S8 experiments"; $PY -m orbpipe.experiments --force > logs/nohup_experiments.out 2>&1
echo "[$(date -u +%FT%TZ)] e2e final (refreshed T5)"; $PY -m orbpipe.e2e --force > logs/nohup_e2e.out 2>&1
echo "[$(date -u +%FT%TZ)] report"; $PY -m orbpipe.report --force > logs/nohup_report.out 2>&1
echo "[$(date -u +%FT%TZ)] ALL DONE"
