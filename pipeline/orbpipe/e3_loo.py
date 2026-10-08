"""E3-LOO: truth-free evaluation of TLE reconstruction on the whole cohort, plus validation of the OAD proxy.

Leave-one-TLE-out (LOO): for object o and held-out TLE k (epoch e_k, within the window), the reference state is
SGP4(TLE_k, e_k) — the most accurate point of that TLE (age 0). The state at e_k is reconstructed from the *other*
TLEs with four configurations and the RTN residual w.r.t. the reference is recorded:
  nearest-raw   nearest other TLE in the raw archive
  nearest-PAF   nearest other TLE in the PAF-lite pool (dedup, B*>=0, isolated-n-jump removed)
  MTF-raw       <=3 other raw TLEs within ±12 h, weights exp(-|Δt|/2 h)      (+ member spread)
  MTF-PAF       same on the PAF-lite pool                                     (+ member spread)
For objects whose archived OrbBench product exists (Multi_results = MTF on the step6/PAF pool), the product state at
e_k is also read (Hermite between 1-min rows). It is *in-sample* (TLE_k was a member) and is reported as product
self-consistency, not as a held-out error.
Held-out TLEs that PAF-lite removes are evaluated separately: their LOO residual should be much larger if PAF removed
genuine faults.

Sentinel objects are run with the same code and, in addition, the POD error of each reconstruction at e_k is
recorded so that the LOO residual can be validated as a proxy for true error.

E4 transfer: the Sentinel-calibrated σ = a + b·spread (from t4_uq_calibration.json) is applied to the cohort's MTF
spread; coverage of the LOO residual and Spearman(spread, |residual|) are reported per type × regime, and the
per-object OAD proxies delivered with Multi_results (overlap/spread medians) are correlated with per-object LOO error.

Outputs: results/e3_loo_samples.parquet, e3_loo_by_group.csv, e3_loo_paired.csv, e3_loo_removed.csv,
         e3_loo_sentinel_validation.csv, e4_cohort_transfer.csv, e4_object_proxy_corr.csv
"""
from __future__ import annotations
import sys, io, json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr

from . import config as C
from .util import (get_logger, set_status, stage_done, Timer, save_df, to_utc, read_tles, sgp4_states, to_rtn, quantiles, mad)
from .experiments import paf_lite

log = get_logger("e3_loo")
MAX_HOLDOUT = 40           # held-out TLEs per object (evenly spaced in the window)
K, TAU_H, WIN_H = 3, 2.0, 12.0
RTN = ["r", "t", "n"]

def _hermite_at(df: pd.DataFrame, epochs: pd.DatetimeIndex) -> np.ndarray:
    """Cubic-Hermite interpolate a 1-min trajectory (index time, cols x..vz km,km/s) at arbitrary epochs."""
    ts = df.index.values.astype("datetime64[ns]").astype("int64") / 1e9; te = epochs.values.astype("datetime64[ns]").astype("int64") / 1e9
    P = df[["x", "y", "z"]].to_numpy(float); V = df[["vx", "vy", "vz"]].to_numpy(float)
    i1 = np.searchsorted(ts, te, side="right"); i0 = i1 - 1
    ok = (i0 >= 0) & (i1 < len(ts)); i0c = np.clip(i0, 0, len(ts) - 1); i1c = np.clip(i1, 0, len(ts) - 1)
    h = ts[i1c] - ts[i0c]; ok &= (h > 0) & (h <= 120); h = np.where(h > 0, h, 1.0); s = np.clip((te - ts[i0c]) / h, 0, 1)
    h00 = 2 * s ** 3 - 3 * s ** 2 + 1; h10 = s ** 3 - 2 * s ** 2 + s; h01 = -2 * s ** 3 + 3 * s ** 2; h11 = s ** 3 - s ** 2
    pos = h00[:, None] * P[i0c] + (h10 * h)[:, None] * V[i0c] + h01[:, None] * P[i1c] + (h11 * h)[:, None] * V[i1c]
    pos[~ok] = np.nan
    return pos

def _read_product(path: str, source: str, t0, t1) -> pd.DataFrame | None:
    try:
        p = Path(path); size = p.stat().st_size; guess = 220_000_000 if source == "multi" else 130_000_000
        with open(p, "rb") as f:
            f.seek(max(0, size - guess)); buf = f.read()
        txt = buf.decode("utf-8", "replace"); txt = txt[txt.find("\n") + 1:]
        if source == "multi":
            d = pd.read_csv(io.StringIO(txt), header=None, usecols=[0, 3, 4, 5, 6, 7, 8]); d.columns = ["t", "x", "y", "z", "vx", "vy", "vz"]; scale = 1.0
        else:
            d = pd.read_csv(io.StringIO(txt), header=None).iloc[:, :7]; d.columns = ["t", "x", "y", "z", "vx", "vy", "vz"]; scale = 1e-3
        d["t"] = pd.to_datetime(d["t"], utc=True, errors="coerce", format="mixed"); d = d.dropna(subset=["t"])
        d = d[(d.t >= t0 - pd.Timedelta(minutes=2)) & (d.t <= t1 + pd.Timedelta(minutes=2))].set_index("t").sort_index()
        d = d[~d.index.duplicated(keep="last")]
        d[["x", "y", "z", "vx", "vy", "vz"]] = d[["x", "y", "z", "vx", "vy", "vz"]].astype(float) * scale
        return d
    except Exception:
        return None

def _recon(others: pd.DataFrame, epoch, mode: str):
    """Reconstruct state at epoch from `others` (DataFrame of TLEs). Returns (state6 | None, age_h, spread_km, n_members)."""
    if len(others) == 0: return None, np.nan, np.nan, 0
    dt_h = (others.epoch - epoch).dt.total_seconds().abs().values / 3600.0
    if mode == "nearest":
        j = int(np.argmin(dt_h)); r = others.iloc[j]
        st = sgp4_states(r.line1, r.line2, [epoch])[0]
        return (st if np.isfinite(st).all() else None), float(dt_h[j]), 0.0, 1
    sel = np.where(dt_h <= WIN_H)[0]
    if len(sel) == 0: sel = np.array([int(np.argmin(dt_h))])
    sel = sel[np.argsort(dt_h[sel])[:K]]
    S, W = [], []
    for j in sel:
        r = others.iloc[int(j)]; st = sgp4_states(r.line1, r.line2, [epoch])[0]
        if np.isfinite(st).all(): S.append(st); W.append(np.exp(-dt_h[j] / TAU_H))
    if not S: return None, np.nan, np.nan, 0
    S = np.array(S); W = np.array(W) / np.sum(W); fused = (W[:, None] * S).sum(0)
    spread = float(np.sqrt(np.sum(W * np.sum((S[:, :3] - fused[:3]) ** 2, axis=1)))) if len(S) > 1 else 0.0
    return fused, float(np.sum(W * dt_h[sel][: len(S)])), spread, len(S)

def _one(args):
    norad, source, path, t0, t1, pod_path = args
    try:
        raw = read_tles(norad, t0 - pd.Timedelta(days=2), t1 + pd.Timedelta(days=2))
        if len(raw) < 4: return norad, None
        paf = paf_lite(raw); kept = set(paf.epoch.values.astype("datetime64[ns]").astype("int64"))
        inwin = raw[(raw.epoch >= t0) & (raw.epoch < t1)]
        if len(inwin) == 0: return norad, None
        idx = np.unique(np.linspace(0, len(inwin) - 1, min(MAX_HOLDOUT, len(inwin))).astype(int))
        prod = _read_product(path, source, t0, t1) if source == "multi" else None
        pod = None
        if pod_path is not None and Path(pod_path).exists():
            pod = pd.read_parquet(pod_path); pod["time"] = pd.to_datetime(pod.time, utc=True)
            cols = [f"{c}_teme" for c in ("x", "y", "z", "vx", "vy", "vz")]
            pod = pod[["time"] + cols].rename(columns=dict(zip(cols, ["x", "y", "z", "vx", "vy", "vz"]))).set_index("time").sort_index()
        rows = []
        raw_ns = raw.epoch.values.astype("datetime64[ns]").astype("int64")
        for i in idx:
            k = inwin.iloc[i]; ek = k.epoch; ek_ns = np.int64(pd.Timestamp(ek).value)
            ref = sgp4_states(k.line1, k.line2, [ek])[0]
            if not np.isfinite(ref).all(): continue
            others_raw = raw[raw_ns != ek_ns]; others_paf = paf[paf.epoch.values.astype("datetime64[ns]").astype("int64") != ek_ns]
            base = dict(norad=norad, source=source, epoch=ek, k_removed_by_paf=bool(ek_ns not in kept),
                        gap_prev_h=float((ek - others_raw[others_raw.epoch < ek].epoch.max()).total_seconds() / 3600) if (others_raw.epoch < ek).any() else np.nan)
            truth_pod = None
            if pod is not None:
                pp = _hermite_at(pod, pd.DatetimeIndex([ek]))[0]
                if np.isfinite(pp).all():
                    # POD velocity for RTN frame: nearest row
                    j = pod.index.get_indexer([ek], method="nearest")[0]; truth_pod = np.concatenate([pp, pod.iloc[j][["vx", "vy", "vz"]].to_numpy(float)])
            for mname, oth, mode in (("nearest-raw", others_raw, "nearest"), ("nearest-PAF", others_paf, "nearest"), ("MTF-raw", others_raw, "mtf"), ("MTF-PAF", others_paf, "mtf")):
                st, age, sp, nm = _recon(oth, ek, mode)
                if st is None: continue
                res = to_rtn((st[:3] - ref[:3])[None], ref[:3][None], ref[3:][None])[0]
                row = dict(base, method=mname, age_h=age, spread_km=sp, n_members=nm, res_r=res[0], res_t=res[1], res_n=res[2], res_3d=float(np.linalg.norm(res)))
                if truth_pod is not None:
                    e = to_rtn((st[:3] - truth_pod[:3])[None], truth_pod[:3][None], truth_pod[3:][None])[0]
                    row.update(pod_err_3d=float(np.linalg.norm(e)), pod_err_t=e[1])
                rows.append(row)
            if prod is not None:
                pp = _hermite_at(prod, pd.DatetimeIndex([ek]))[0]
                if np.isfinite(pp).all():
                    res = to_rtn((pp - ref[:3])[None], ref[:3][None], ref[3:][None])[0]
                    rows.append(dict(base, method="product (MTF+PAF, in-sample)", age_h=0.0, spread_km=np.nan, n_members=np.nan, res_r=res[0], res_t=res[1], res_n=res[2], res_3d=float(np.linalg.norm(res))))
            if truth_pod is not None:
                e = to_rtn((ref[:3] - truth_pod[:3])[None], truth_pod[:3][None], truth_pod[3:][None])[0]
                rows.append(dict(base, method="reference TLE_k itself", age_h=0.0, spread_km=np.nan, n_members=1, res_r=0, res_t=0, res_n=0, res_3d=0.0, pod_err_3d=float(np.linalg.norm(e)), pod_err_t=e[1]))
        return norad, (pd.DataFrame(rows) if rows else None)
    except Exception as ex:
        return norad, str(ex)

def summarize(S: pd.DataFrame, obj: pd.DataFrame):
    S = S.merge(obj[["norad", "type", "regime"]], on="norad", how="left")
    kept = S[~S.k_removed_by_paf]
    rows = []
    for (m,), g in kept.groupby(["method"]):
        q = quantiles(g.res_3d.values); rows.append(dict(scope="ALL", method=m, n=len(g), n_obj=g.norad.nunique(), **{f"res3d_{k}_km": v for k, v in q.items()},
                                                    res_t_p50_km=float(np.nanmedian(np.abs(g.res_t))), spread_med_km=float(np.nanmedian(g.spread_km)) if g.spread_km.notna().any() else np.nan))
    for (ty, rg, m), g in kept.groupby(["type", "regime", "method"]):
        if len(g) < 100: continue
        q = quantiles(g.res_3d.values); rows.append(dict(scope=f"{ty}|{rg}", method=m, n=len(g), n_obj=g.norad.nunique(), **{f"res3d_{k}_km": v for k, v in q.items()},
                                                    res_t_p50_km=float(np.nanmedian(np.abs(g.res_t))), spread_med_km=float(np.nanmedian(g.spread_km)) if g.spread_km.notna().any() else np.nan))
    for (src, m), g in kept.groupby(["source", "method"]):
        q = quantiles(g.res_3d.values); rows.append(dict(scope=f"source={src}", method=m, n=len(g), n_obj=g.norad.nunique(), **{f"res3d_{k}_km": v for k, v in q.items()},
                                                    res_t_p50_km=float(np.nanmedian(np.abs(g.res_t))), spread_med_km=np.nan))
    by_group = pd.DataFrame(rows)
    # paired per-object comparison vs nearest-raw
    per = kept.groupby(["norad", "method"]).res_3d.median().unstack()
    pr = []
    for m in [c for c in per.columns if c != "nearest-raw"]:
        d = per[[m, "nearest-raw"]].dropna(); rel = (d[m] - d["nearest-raw"]) / d["nearest-raw"]
        pr.append(dict(method=m, n_obj=len(d), win_rate_vs_nearest_raw=float((d[m] < d["nearest-raw"]).mean()), median_rel_change_pct=float(100 * rel.median()),
                       median_res3d_km=float(d[m].median()), nearest_raw_median_km=float(d["nearest-raw"].median())))
    paired = pd.DataFrame(pr)
    # removed TLEs
    rem = S[S.method == "nearest-raw"].groupby("k_removed_by_paf").res_3d.agg(["count", "median", lambda x: np.percentile(x, 90)]).reset_index()
    rem.columns = ["k_removed_by_paf", "n", "res3d_p50_km", "res3d_p90_km"]
    return by_group, paired, rem, S

def e4_transfer(S: pd.DataFrame):
    rows = []; corr = []
    calib = json.loads((C.RESULTS / "t4_uq_calibration.json").read_text()) if (C.RESULTS / "t4_uq_calibration.json").exists() else None
    m = S[(S.method == "MTF-PAF") & (S.spread_km > 0) & (~S.k_removed_by_paf)].copy()
    if calib is not None and len(m):
        for c in RTN:
            m[f"sigma_{c}"] = np.maximum(calib[c]["oad_a"] + calib[c]["oad_b"] * m.spread_km, 1e-3)
            m[f"sigma_age_{c}"] = np.maximum(calib[c]["age_a"] + calib[c]["age_b"] * m.age_h, 1e-3)
        for scope, g in [("ALL", m)] + [(f"{t}|{r}", g) for (t, r), g in m.groupby(["type", "regime"]) if len(g) >= 100]:
            row = dict(scope=scope, n=len(g), spearman_spread_res3d=float(spearmanr(g.spread_km, g.res_3d).statistic))
            for c in RTN:
                z = g[f"res_{c}"] / g[f"sigma_{c}"]; za = g[f"res_{c}"] / g[f"sigma_age_{c}"]
                row[f"cov95_oad_{c}"] = float(np.mean(np.abs(z) <= 1.96)); row[f"nees_oad_{c}"] = float(np.mean(z ** 2))
                row[f"cov95_age_{c}"] = float(np.mean(np.abs(za) <= 1.96))
            rows.append(row)
    # object-level proxy correlation (delivered OAD proxies)
    try:
        px = pd.read_csv(C.MNT / "processed_data/multi_proxy_step6_deliverable/results/per_sat_ok_531.csv")[["norad", "overlap_median", "spread_median", "overlap_p95", "spread_p95"]]
        per = S[~S.k_removed_by_paf].groupby(["norad", "method"]).res_3d.median().unstack().reset_index().merge(px, on="norad")
        for meth in ("nearest-raw", "MTF-PAF", "product (MTF+PAF, in-sample)"):
            if meth in per:
                d = per[[meth, "overlap_median", "spread_median"]].dropna()
                if len(d) > 10:
                    corr.append(dict(loo_method=meth, n_obj=len(d), spearman_overlap=float(spearmanr(d.overlap_median, d[meth]).statistic), spearman_spread=float(spearmanr(d.spread_median, d[meth]).statistic)))
    except Exception as ex:
        log.warning(f"object proxy correlation skipped: {ex}")
    return pd.DataFrame(rows), pd.DataFrame(corr)

def main(force=False):
    if stage_done("S8b_e3_loo") and not force: log.info("done"); return
    set_status("S8b_e3_loo", "running")
    try:
        with Timer() as T:
            obj = pd.read_csv(C.DATA / "objects.csv"); t0, t1 = to_utc(C.WINDOW_START), to_utc(C.WINDOW_END)
            tasks = [(int(r.norad), r.source, r.path, t0, t1, None) for r in obj.itertuples()]
            for name, (norad, _, _) in C.SENTINELS.items():
                tasks.append((norad, "sentinel", "", t0, t1, str(C.DATA / f"sentinel_pod_{name}.parquet")))
            out = []; errs = 0
            with ProcessPoolExecutor(C.N_WORKERS) as ex:
                futs = [ex.submit(_one, t) for t in tasks]
                for i, fu in enumerate(as_completed(futs)):
                    n, r = fu.result()
                    if isinstance(r, pd.DataFrame): out.append(r)
                    elif isinstance(r, str): errs += 1
                    if (i + 1) % 300 == 0: log.info(f"LOO {i+1}/{len(tasks)}")
            S = pd.concat(out, ignore_index=True); S.to_parquet(C.RESULTS / "e3_loo_samples.parquet", index=False)
            log.info(f"LOO samples: {len(S)} rows, {S.norad.nunique()} objects, errors={errs}")
            sent = pd.DataFrame([dict(norad=n, type="PAY", regime="LEO") for _, (n, _, _) in C.SENTINELS.items()])
            objx = pd.concat([obj[["norad", "type", "regime"]], sent], ignore_index=True).drop_duplicates("norad")
            by_group, paired, rem, S2 = summarize(S, objx)
            save_df(by_group, "e3_loo_by_group.csv"); save_df(paired, "e3_loo_paired.csv"); save_df(rem, "e3_loo_removed.csv")
            # Sentinel validation of the proxy
            sv = S2[S2.pod_err_3d.notna()]
            if len(sv):
                rows = []
                for m, g in sv.groupby("method"):
                    rows.append(dict(method=m, n=len(g), loo_res3d_p50_km=float(np.median(g.res_3d)), pod_err3d_p50_km=float(np.median(g.pod_err_3d)),
                                     spearman_loo_vs_pod=float(spearmanr(g.res_3d, g.pod_err_3d).statistic) if g.res_3d.std() > 0 else np.nan))
                save_df(pd.DataFrame(rows), "e3_loo_sentinel_validation.csv")
            tr, corr = e4_transfer(S2); save_df(tr, "e4_cohort_transfer.csv"); save_df(corr, "e4_object_proxy_corr.csv")
            log.info("by group ALL:\n" + by_group[by_group.scope == "ALL"].round(4).to_string(index=False))
            log.info("paired:\n" + paired.round(4).to_string(index=False)); log.info("removed:\n" + rem.round(4).to_string(index=False))
            if len(tr): log.info("E4 transfer ALL:\n" + tr[tr.scope == "ALL"].round(3).to_string(index=False))
        set_status("S8b_e3_loo", "done", elapsed_s=round(T.elapsed), n_samples=int(len(S)))
    except Exception as ex:
        log.exception("E3-LOO failed"); set_status("S8b_e3_loo", "failed", message=str(ex)); raise

if __name__ == "__main__":
    main("--force" in sys.argv)
