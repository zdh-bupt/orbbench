#!/usr/bin/env bash
# wait for Sentinel long-window restage and E3-LOO, then rerun T2, T4 (long Sentinel window), experiments, report
cd "$(dirname "$0")"; export OMP_NUM_THREADS=2
while pgrep -f "scripts_restage_pod.py" >/dev/null; do sleep 30; done
echo "[$(date -u +%FT%TZ)] restage finished" >> logs/run_all.log
for m in "orbpipe.t2_maneuver --force" "orbpipe.t4_forecast --force"; do
  echo "[$(date -u +%FT%TZ)] === rerun $m ===" >> logs/run_all.log
  ./.venv/bin/python -m $m >> logs/run_all.log 2>&1; echo "[$(date -u +%FT%TZ)] === $m exit $? ===" >> logs/run_all.log
done
while pgrep -f "orbpipe.e3_loo" >/dev/null; do sleep 30; done
for m in "orbpipe.experiments --force" "orbpipe.report"; do
  echo "[$(date -u +%FT%TZ)] === rerun $m ===" >> logs/run_all.log
  ./.venv/bin/python -m $m >> logs/run_all.log 2>&1; echo "[$(date -u +%FT%TZ)] === $m exit $? ===" >> logs/run_all.log
done
echo "[$(date -u +%FT%TZ)] long-window chain done" >> logs/run_all.log
