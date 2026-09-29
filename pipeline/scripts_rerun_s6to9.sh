#!/usr/bin/env bash
cd "$(dirname "$0")"; export OMP_NUM_THREADS=2
for m in "orbpipe.t6_warning --force" "orbpipe.t7_avoidance --force" "orbpipe.experiments --force" "orbpipe.report"; do
  echo "[$(date -u +%FT%TZ)] === rerun $m ===" >> logs/run_all.log
  ./.venv/bin/python -m $m >> logs/run_all.log 2>&1; echo "[$(date -u +%FT%TZ)] === $m exit $? ===" >> logs/run_all.log
done
