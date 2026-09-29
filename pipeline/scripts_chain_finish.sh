#!/usr/bin/env bash
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1
run(){ echo "[$(date -u +%FT%TZ)] === rerun $1 ===" >> logs/run_all.log; ./.venv/bin/python -m $1 >> logs/run_all.log 2>&1; echo "[$(date -u +%FT%TZ)] === $1 exit $? ===" >> logs/run_all.log; }
run "orbpipe.chain_proof 2 --force"
while pgrep -f "scripts_chain_proof.sh" >/dev/null; do sleep 20; done
run "orbpipe.chain_proof 5 --force"
run "orbpipe.report"
echo "[$(date -u +%FT%TZ)] chain finish done" >> logs/run_all.log
