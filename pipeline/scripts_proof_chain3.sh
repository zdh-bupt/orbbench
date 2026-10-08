#!/usr/bin/env bash
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1
while pgrep -f "scripts_proof_chain2.sh" >/dev/null; do sleep 20; done
run(){ echo "[$(date -u +%FT%TZ)] === rerun $1 ===" >> logs/run_all.log; ./.venv/bin/python -m $1 >> logs/run_all.log 2>&1; echo "[$(date -u +%FT%TZ)] === $1 exit $? ===" >> logs/run_all.log; }
run "orbpipe.proof 2 --force"
run "orbpipe.report"
echo "[$(date -u +%FT%TZ)] proof chain 3 done" >> logs/run_all.log
