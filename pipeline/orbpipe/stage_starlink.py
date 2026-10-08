"""Stage the Starlink truth cohort: 20 satellites with raw + step6 (official PAF) TLEs to 2026-08-24 (smoke package)
and operator MEME ephemerides with 6x6 covariance (UVW) from the daily zips 2026-06-10 .. 2026-09-10.

Output per satellite: data/starlink/meme_<norad>.parquet with
  time (UTC, 1 min), x,y,z,vx,vy,vz (km, km/s, MEME J2000 inertial), created (release time of the file used),
  c00..c55 lower-triangle covariance (21 values, UVW = radial/in-track/cross-track; km^2, km^2/s, km^2/s^2),
  sig_u_m, sig_v_m, sig_w_m (position 1-sigma in metres).
Duplicate timestamps across releases: keep the latest release (created).
"""
from __future__ import annotations
import re, sys, zipfile, glob
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
from . import config as C
from .util import get_logger, set_status, Timer

log = get_logger("stage_starlink")
SMOKE = C.DATA / "smoke/orbbench_server_smoke/data"
OUT = C.DATA / "starlink"; OUT.mkdir(parents=True, exist_ok=True)
RAW_TLE_DIR = SMOKE / "原始TLE/TLEs_NORAD"
STEP6_TLE_DIR = SMOKE / "原始TLE/TLEs_NORAD_onorbit_halfperiod_ge2006utc_step6_bstar_nonneg_pruned"

def starlink_ids() -> list[int]:
    return sorted(int(x) for x in (SMOKE / "satellite_list_21.txt").read_text().split() if x.strip().isdigit())

def parse_meme(text: str):
    L = text.splitlines()
    created = re.search(r"created:(.*?) UTC", L[0]).group(1).strip()
    rows = []; i = 4
    while i + 3 < len(L):
        p = L[i].split()
        if len(p) != 7: i += 1; continue
        ts = p[0]; yyyy, doy = int(ts[:4]), int(ts[4:7]); hh, mm, ss = int(ts[7:9]), int(ts[9:11]), float(ts[11:])
        t = pd.Timestamp(year=yyyy, month=1, day=1, tz="UTC") + pd.Timedelta(days=doy - 1, hours=hh, minutes=mm, seconds=ss)
        cov = [float(v) for v in (L[i + 1] + " " + L[i + 2] + " " + L[i + 3]).split()]
        if len(cov) != 21: i += 4; continue
        rows.append((t, *[float(v) for v in p[1:7]], *cov)); i += 4
    cols = ["time", "x", "y", "z", "vx", "vy", "vz"] + [f"c{k}" for k in range(21)]
    df = pd.DataFrame(rows, columns=cols); df["created"] = pd.Timestamp(created, tz="UTC")
    return df

def _one(norad: int):
    parts = []
    for zp in sorted(glob.glob(str(C.STARLINK_EPH_DIR / "*.zip"))):
        try:
            z = zipfile.ZipFile(zp)
            for n in z.namelist():
                if n.endswith(".txt") and f"/MEME_{norad}_" in n:
                    parts.append(parse_meme(z.read(n).decode("utf-8", "replace")))
        except Exception as ex:
            log.warning(f"{norad} {zp}: {ex}")
    if not parts: return norad, 0
    d = pd.concat(parts, ignore_index=True).sort_values(["time", "created"]).drop_duplicates("time", keep="last").reset_index(drop=True)
    # lower-triangle order: P00, P10,P11, P20,P21,P22, P30..P33, P40..P44, P50..P55 -> diag indices 0,2,5,9,14,20
    d["sig_u_m"] = 1000 * np.sqrt(d.c0.clip(lower=0)); d["sig_v_m"] = 1000 * np.sqrt(d.c2.clip(lower=0)); d["sig_w_m"] = 1000 * np.sqrt(d.c5.clip(lower=0))
    d.to_parquet(OUT / f"meme_{norad}.parquet", index=False)
    return norad, len(d)

def main():
    set_status("S0b_starlink", "running")
    try:
        with Timer() as T:
            ids = starlink_ids(); log.info(f"Starlink truth cohort: {len(ids)} satellites")
            tot = 0
            with ProcessPoolExecutor(min(C.N_WORKERS, len(ids))) as ex:
                for fu in as_completed([ex.submit(_one, n) for n in ids]):
                    n, k = fu.result(); tot += k; log.info(f"{n}: {k} 1-min MEME states")
            summ = []
            for n in ids:
                p = OUT / f"meme_{n}.parquet"
                if not p.exists(): continue
                d = pd.read_parquet(p)
                summ.append(dict(norad=n, n_states=len(d), t_start=d.time.min(), t_end=d.time.max(), days=(d.time.max() - d.time.min()).total_seconds() / 86400,
                                 sig_u_med_m=d.sig_u_m.median(), sig_v_med_m=d.sig_v_m.median(), sig_w_med_m=d.sig_w_m.median(), sig_v_p95_m=d.sig_v_m.quantile(.95),
                                 raw_tle=(RAW_TLE_DIR / f"{n}.txt").exists(), step6_tle=(STEP6_TLE_DIR / f"{n}.txt").exists()))
            S = pd.DataFrame(summ); S.to_csv(OUT / "summary.csv", index=False); log.info("\n" + S.round(2).to_string(index=False))
        set_status("S0b_starlink", "done", elapsed_s=round(T.elapsed), n_states=int(tot))
    except Exception as ex:
        log.exception("stage_starlink failed"); set_status("S0b_starlink", "failed", message=str(ex)); raise

if __name__ == "__main__":
    main()
