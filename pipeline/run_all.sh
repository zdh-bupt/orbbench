#!/usr/bin/env bash
# Run the whole pipeline stage by stage (idempotent: stages marked done in status.json are skipped).
# Usage: bash run_all.sh [stage ...]   e.g. bash run_all.sh S0 S1
set -uo pipefail
cd "$(dirname "$0")"
PY=./.venv/bin/python
export ORBPIPE_WORKERS=${ORBPIPE_WORKERS:-6}
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
mkdir -p logs

declare -A MOD=(
  [S0]=orbpipe.stage0_staging
  [S1]=orbpipe.t3_weather
  [S2]=orbpipe.t1_phase
  [S3]=orbpipe.t2_maneuver
  [S4]=orbpipe.t4_forecast
  [S5]=orbpipe.t5_evolution
  [S6]=orbpipe.t6_warning
  [S7]=orbpipe.t7_avoidance
  [S8]=orbpipe.experiments
  [S8b]=orbpipe.e3_loo
  [S8c]=orbpipe.challenge_evidence
  [S8g]=orbpipe.chain_ablation
  [S8h]=orbpipe.exp_summary
  [S8i]=orbpipe.external
  [S8j]=orbpipe.e2e
  [S8k]=orbpipe.upstream_chain
  [S9]=orbpipe.report
)
ORDER=(S0 S1 S2 S3 S4 S5 S6 S7 S8 S8b S8c S8g S8h S8i S8k S8j S9)
STAGES=("$@"); [ ${#STAGES[@]} -eq 0 ] && STAGES=("${ORDER[@]}")

for s in "${STAGES[@]}"; do
  m=${MOD[$s]}
  echo "[$(date -u +%FT%TZ)] === $s: $m ===" | tee -a logs/run_all.log
  $PY -m "$m" >> "logs/run_all.log" 2>&1
  rc=$?
  echo "[$(date -u +%FT%TZ)] === $s exit $rc ===" | tee -a logs/run_all.log
  if [ $rc -ne 0 ]; then echo "stage $s failed, continuing with next stages that do not depend on it"; fi
done
