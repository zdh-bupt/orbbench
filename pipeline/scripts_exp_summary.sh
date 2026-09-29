#!/usr/bin/env bash
cd "$(dirname "$0")"
while pgrep -f "chain_ablatio[n] --forc[e]" >/dev/null; do sleep 30; done
./.venv/bin/python -m orbpipe.exp_summary --force
