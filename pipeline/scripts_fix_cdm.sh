#!/usr/bin/env bash
# wait for S6..S9 chain to finish, then overwrite CDM metrics with the fixed regression and re-run S8/S9
cd "$(dirname "$0")"
while pgrep -f "run_all.sh S5 S6 S7 S8 S9" >/dev/null; do sleep 30; done
./.venv/bin/python - <<'PY' 2>/dev/null
from orbpipe.t6_warning import run_cdm
from orbpipe.util import save_df
save_df(run_cdm(), "t6_cdm_metrics.csv"); print("cdm metrics rewritten")
PY
./.venv/bin/python -m orbpipe.experiments --force >> logs/run_all.log 2>&1
./.venv/bin/python -m orbpipe.report >> logs/run_all.log 2>&1
echo "[$(date -u +%FT%TZ)] cdm fix + S8/S9 rerun done" >> logs/run_all.log
