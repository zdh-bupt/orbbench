"""Print pipeline progress: stage states, running processes, recent log errors, produced results."""
from __future__ import annotations
import json, subprocess, sys, time
from pathlib import Path
from . import config as C

def main() -> None:
    st = json.loads((C.ROOT / "status.json").read_text()) if (C.ROOT / "status.json").exists() else {}
    print(f"# orbpipe status @ {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}")
    print(f"last_update: {st.get('last_update')}")
    for k, v in st.get("stages", {}).items():
        extra = {kk: vv for kk, vv in v.items() if kk not in ('state', 'updated')}
        print(f"  {k:22s} {v['state']:8s} {v['updated']}  {extra if extra else ''}")
    # running processes
    try:
        ps = subprocess.run(["bash", "-lc", "ps -eo pid,etime,pcpu,rss,args | grep -E 'orbpipe|run_all' | grep -v grep"], capture_output=True, text=True).stdout.strip()
        print("running:\n" + (ps if ps else "  (none)"))
    except Exception:
        pass
    # recent errors
    for lg in sorted(C.LOGS.glob("*.log")):
        lines = lg.read_text(errors="replace").splitlines()[-400:]
        errs = [l for l in lines if " ERROR " in l or "Traceback" in l or "failed" in l.lower()]
        if errs:
            print(f"errors in {lg.name}: {len(errs)} (last: {errs[-1][:200]})")
    # results inventory
    res = sorted(C.RESULTS.rglob("*"))
    files = [p for p in res if p.is_file()]
    print(f"results: {len(files)} files, figures: {len([p for p in C.FIGURES.rglob('*') if p.is_file()])}")
    for p in files[-15:]:
        print(f"  {p.relative_to(C.RESULTS)}  {p.stat().st_size//1024} KB")
    # tail of the most recently modified log
    logs = sorted(C.LOGS.glob("*.log"), key=lambda p: p.stat().st_mtime)
    if logs:
        print(f"--- tail {logs[-1].name}")
        print("\n".join(logs[-1].read_text(errors='replace').splitlines()[-8:]))

if __name__ == "__main__":
    main()
