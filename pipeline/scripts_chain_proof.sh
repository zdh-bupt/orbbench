#!/usr/bin/env bash
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1
run(){ echo "[$(date -u +%FT%TZ)] === rerun $1 ===" >> logs/run_all.log; ./.venv/bin/python -m $1 >> logs/run_all.log 2>&1; echo "[$(date -u +%FT%TZ)] === $1 exit $? ===" >> logs/run_all.log; }
run "orbpipe.chain_proof 12345 --force"
run "orbpipe.report"
echo "[$(date -u +%FT%TZ)] chain proof done" >> logs/run_all.log
