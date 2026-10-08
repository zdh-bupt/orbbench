#!/usr/bin/env bash
# after E3-LOO finishes: run challenge-evidence experiments (E5–E8), then refresh experiments + report
cd "$(dirname "$0")"; export OMP_NUM_THREADS=1
while pgrep -f "orbpipe.e3_loo" >/dev/null; do sleep 30; done
echo "[$(date -u +%FT%TZ)] === rerun orbpipe.challenge_evidence ===" >> logs/run_all.log
./.venv/bin/python -m orbpipe.challenge_evidence --force >> logs/run_all.log 2>&1; echo "[$(date -u +%FT%TZ)] === challenge_evidence exit $? ===" >> logs/run_all.log
# wait for the T2/T4 long-window chain if still running, then refresh
while pgrep -f "scripts_after_restage.sh" >/dev/null; do sleep 30; done
./.venv/bin/python -m orbpipe.experiments --force >> logs/run_all.log 2>&1
./.venv/bin/python -m orbpipe.report >> logs/run_all.log 2>&1
echo "[$(date -u +%FT%TZ)] challenge-evidence chain done" >> logs/run_all.log
