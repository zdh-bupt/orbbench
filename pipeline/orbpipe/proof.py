"""Proof experiments for the two OrbBench claims, with independent POD truth (Sentinels, test window 2024-01 -> 2025-06)
and the full cohort where no truth exists.

P1  TLE thinning with truth: keep a random fraction f of each Sentinel's TLEs (KEEP x 3 seeds) to emulate debris-like
    cadence, reconstruct the orbit at 30-min grid points non-causally and compare with POD:
      nearest | blend-2 | MTF (k3,tau2h,W12h) | MTF-wide (k3,tau6h,W36h) | MTF-adaptive (tau,W scale with local gap)
      MTF-adaptive+T2-gate (members must lie in the same maneuver segment as t; cuts at T2-detected maneuvers)
      MTF-adaptive+oracle-gate (cuts at logged maneuvers; upper bound)
P2  Fault injection with truth: corrupt a fraction p of the Sentinel TLEs with faults whose mix/magnitude come from the
    cohort archive census (isolated mean-motion jump | negative B* | duplicated release); compare raw vs PAF-lite pools
    with nearest and MTF-adaptive against POD; PAF-lite recall on injected faults and false-removal rate on clean TLEs.
P3  Cohort neighbour effect (truth-free): condition LOO samples on 'nearest raw neighbour is a PAF-removed TLE' and
    compare nearest-raw vs nearest-PAF residual; sparsity statistics per type x regime.
P4  Storm-aware sigma: sigma-age vs sigma-age x g(Ap) with observed / T3-forecast Ap; coverage on storm vs quiet days.
"""
from __future__ import annotations
import sys, os
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.expanduser("~"), ".cache", "orbbench-mpl"))
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd

from . import config as C
from .util import (get_logger, set_status, stage_done, Timer, save_df, to_utc, read_tles, sgp4_states, to_rtn, quantiles, mad,
                   load_maneuvers, propagate_nearest)
from .experiments import paf_lite

log = get_logger("proof")
KEEP = [1.0, 0.5, 0.33, 0.2, 0.1]; SEEDS = [0, 1, 2]
GRID_MIN = 30
K_RES = 18.0
FAULT_P = [0.0, 0.02, 0.05, 0.10]; FAULT_MIX = {"n_jump": 0.5, "bstar_neg": 0.3, "duplicate": 0.2}

def _prop_members(tles, times, sel, w):
    N = len(times); states = np.zeros((N, 6)); wsum = np.zeros(N); ens = np.zeros(N, int)
    for j in range(len(tles)):
        m = sel[:, j]
        if not m.any(): continue
        st = sgp4_states(tles.iloc[j].line1, tles.iloc[j].line2, times[m]); ok = np.isfinite(st).all(1)
        ww = w[m, j].copy(); ww[~ok] = 0; st[~ok] = 0
        states[m] += st * ww[:, None]; wsum[m] += ww; ens[m] += ok.astype(int)
    good = wsum > 0; states[good] /= wsum[good, None]; states[~good] = np.nan
    return states, ens

def fuse(tles, times, k=3, tau_h=2.0, win_h=12.0, adaptive=False, blend2=False, seg_tle=None, seg_time=None):
    ep = tles.epoch.values.astype("datetime64[ns]").astype("int64") / 3.6e12; tt = times.values.astype("datetime64[ns]").astype("int64") / 3.6e12
    if len(tles) == 0: return np.full((len(times), 6), np.nan), np.zeros(len(times), int)
    aage = np.abs(tt[:, None] - ep[None, :])
    pos = np.searchsorted(ep, tt); prev = np.clip(pos - 1, 0, len(ep) - 1); nxt = np.clip(pos, 0, len(ep) - 1)
    gap = np.where((pos > 0) & (pos < len(ep)), ep[nxt] - ep[prev], np.nan)
    if blend2:
        sel = np.zeros_like(aage, bool); sel[np.arange(len(tt)), prev] = True; sel[np.arange(len(tt)), nxt] = True
        w = np.zeros_like(aage); g = np.where(np.isfinite(gap) & (gap > 0), gap, 1.0)
        w[np.arange(len(tt)), prev] = np.clip((ep[nxt] - tt) / g, 0, 1); w[np.arange(len(tt)), nxt] = np.clip((tt - ep[prev]) / g, 0, 1)
        edge = ~np.isfinite(gap); w[edge] = 0; w[edge, np.argmin(aage[edge], axis=1)] = 1.0
        return _prop_members(tles, times, sel, w)
    tau = np.full(len(tt), tau_h); W = np.full(len(tt), win_h)
    if adaptive:
        g = np.where(np.isfinite(gap), gap, win_h); tau = np.maximum(tau_h, g / 2); W = np.maximum(win_h, 1.5 * g)
    within = aage <= W[:, None]
    if seg_tle is not None: within &= (seg_tle[None, :] == seg_time[:, None])
    order = np.argsort(np.where(within, aage, np.inf), axis=1)[:, :k]
    sel = np.zeros_like(within); np.put_along_axis(sel, order, True, axis=1); sel &= within
    none = ~sel.any(1)
    if none.any(): sel[none, np.argmin(aage[none], axis=1)] = True
    w = np.exp(-aage / tau[:, None]) * sel
    return _prop_members(tles, times, sel, w)

def segments_from_flags(tles, flag_epochs, times):
    ep = tles.epoch.values.astype("datetime64[ns]").astype("int64")
    cut = np.isin(ep, np.asarray(flag_epochs).astype("datetime64[ns]").astype("int64")); seg_tle = np.cumsum(cut)
    tt = times.values.astype("datetime64[ns]").astype("int64"); pos = np.clip(np.searchsorted(ep, tt, side="right") - 1, 0, len(ep) - 1)
    return seg_tle, seg_tle[pos]

def t2_flags(tles):
    if len(tles) < 4: return np.array([], dtype="datetime64[ns]")
    own = np.vstack([sgp4_states(r.line1, r.line2, [r.epoch]) for r in tles.itertuples()])
    res = np.full(len(tles), np.nan)
    for i in range(1, len(tles)):
        p = tles.iloc[i - 1]; st = sgp4_states(p.line1, p.line2, [tles.epoch.iloc[i]])[0]
        if np.isfinite(st).all() and np.isfinite(own[i]).all():
            res[i] = to_rtn((st[:3] - own[i, :3])[None], own[i, :3][None], own[i, 3:][None])[0][1]
    s = np.abs(res) / max(mad(res), 1e-9)
    return tles.epoch.values[s > K_RES].astype("datetime64[ns]")

def oracle_flags(tles, man):
    ep = tles.epoch.values.astype("datetime64[ns]"); out = []
    for r in man.itertuples():
        i = np.searchsorted(ep, np.datetime64(r.stop.tz_convert(None)))
        if i < len(ep): out.append(ep[i])
    return np.array(out, dtype="datetime64[ns]")

def rtn_err(states, truth):
    e = np.full((len(states), 3), np.nan); ok = np.isfinite(states).all(1) & np.isfinite(truth).all(1)
    e[ok] = to_rtn(states[ok, :3] - truth[ok, :3], truth[ok, :3], truth[ok, 3:]); return e

def _load_truth(name):
    from .t4_forecast import load_pod
    pod = load_pod(name); t0, t1 = to_utc(C.SENTINEL_TEST_START), to_utc(C.WINDOW_END)
    pod = pod[(pod.index >= t0) & (pod.index < t1)]
    times = pod.index[::GRID_MIN]; truth = pod.loc[times, ["x", "y", "z", "vx", "vy", "vz"]].to_numpy()
    return times, truth, t0, t1

def _has_truth(name) -> bool:
    try:
        times, _, _, _ = _load_truth(name); return len(times) >= 200
    except Exception:
        return False

# ------------------------------------------------------------------ P1
def _p1_one(args):
    name, norad = args
    times, truth, t0, t1 = _load_truth(name)
    full = read_tles(norad, t0 - pd.Timedelta(days=3), t1 + pd.Timedelta(days=3)); man = load_maneuvers(name)
    man_mask = np.zeros(len(times), bool)
    for r in man.itertuples(): man_mask |= (times >= r.start - pd.Timedelta(hours=24)) & (times <= r.stop + pd.Timedelta(hours=24))
    rows = []
    for f in KEEP:
        for seed in (SEEDS if f < 1 else [0]):
            rng = np.random.default_rng(seed * 7919 + norad); keep = rng.random(len(full)) < f if f < 1 else np.ones(len(full), bool)
            tl = full[keep].reset_index(drop=True)
            if len(tl) < 4: continue
            ep = tl.epoch.values.astype("datetime64[ns]").astype("int64") / 3.6e12; tt = times.values.astype("datetime64[ns]").astype("int64") / 3.6e12
            pos = np.searchsorted(ep, tt); ok = (pos > 0) & (pos < len(ep))
            gap = np.where(ok, ep[np.clip(pos, 0, len(ep) - 1)] - ep[np.clip(pos - 1, 0, len(ep) - 1)], np.nan)
            meth = {}
            st, _ = propagate_nearest(tl, times); meth["nearest"] = st
            meth["blend-2"], _ = fuse(tl, times, blend2=True)
            meth["MTF"], _ = fuse(tl, times, 3, 2.0, 12.0)
            meth["MTF-wide"], _ = fuse(tl, times, 3, 6.0, 36.0)
            meth["MTF-adaptive"], _ = fuse(tl, times, 3, 2.0, 12.0, adaptive=True)
            fl = t2_flags(tl); s_t, s_time = segments_from_flags(tl, fl, times)
            meth["MTF-adaptive+T2-gate"], _ = fuse(tl, times, 3, 2.0, 12.0, adaptive=True, seg_tle=s_t, seg_time=s_time)
            if len(man):
                of = oracle_flags(tl, man); s_t, s_time = segments_from_flags(tl, of, times)
                meth["MTF-adaptive+oracle-gate"], _ = fuse(tl, times, 3, 2.0, 12.0, adaptive=True, seg_tle=s_t, seg_time=s_time)
            for m, st in meth.items():
                e = rtn_err(st, truth); e3 = np.linalg.norm(e, axis=1)
                rows.append(pd.DataFrame({"sat": name, "keep": f, "seed": seed, "method": m, "time": times, "e3d": e3, "et": np.abs(e[:, 1]),
                                          "gap_h": gap, "man": man_mask, "n_tle": len(tl), "n_flags_t2": len(fl)}))
    return pd.concat(rows, ignore_index=True) if rows else None

def p1_summarize(S):
    rows = []
    def agg(g, **tag):
        q = quantiles(g.e3d.values, (0.5, 0.9, 0.95)); return dict(n=int(g.e3d.notna().sum()), **{f"e3d_{k}": v for k, v in q.items()}, et_p50=float(np.nanmedian(g.et)), **tag)
    for (f, m), g in S.groupby(["keep", "method"]): rows.append(agg(g, keep=f, method=m, split="all", sat="ALL"))
    for (f, m, s), g in S.groupby(["keep", "method", "sat"]): rows.append(agg(g, keep=f, method=m, split="all", sat=s))
    for (f, m, mm), g in S.groupby(["keep", "method", "man"]): rows.append(agg(g, keep=f, method=m, split="maneuver+-24h" if mm else "quiet", sat="ALL"))
    tab = pd.DataFrame(rows)
    S = S.copy(); S["gap_bin"] = pd.cut(S.gap_h, [0, 6, 12, 24, 48, 96, 1e9], labels=["<6h", "6-12h", "12-24h", "24-48h", "48-96h", ">96h"])
    gb = S.groupby(["gap_bin", "method"], observed=True).e3d.agg(n="count", p50="median", p90=lambda x: x.quantile(.9)).reset_index()
    ci = []
    piv = S.pivot_table(index=["sat", "keep", "seed", "time"], columns="method", values="e3d").reset_index()
    piv["day"] = pd.to_datetime(piv.time).dt.floor("D")
    for f, g in piv.groupby("keep"):
        days = g.day.unique(); rng = np.random.default_rng(C.SEED)
        for m in [c for c in piv.columns if c not in ("sat", "keep", "seed", "time", "day", "nearest")]:
            if m not in g or g[m].isna().all(): continue
            base = float(np.nanmedian(g["nearest"])); val = float(np.nanmedian(g[m])); bs = []
            for _ in range(200):
                dd = rng.choice(days, len(days), replace=True); h = g[g.day.isin(dd)]
                bs.append(100 * (np.nanmedian(h[m]) - np.nanmedian(h["nearest"])) / np.nanmedian(h["nearest"]))
            ci.append(dict(keep=f, method=m, nearest_p50=base, method_p50=val, rel_change_pct=100 * (val - base) / base, ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5)),
                           win_rate=float(np.nanmean(g[m] < g["nearest"]))))
    return tab, gb, pd.DataFrame(ci)

# ------------------------------------------------------------------ P2
def jump_distribution(obj, n_obj=400):
    rng = np.random.default_rng(C.SEED); vals = []
    for norad in rng.choice(obj.norad.values, min(n_obj, len(obj)), replace=False):
        t = read_tles(int(norad), "2006-01-01", None)
        if len(t) < 5: continue
        n = t.n.values; dn = np.abs(np.diff(n)); med = np.median(dn) if len(dn) else 0
        for i in range(1, len(n) - 1):
            if abs(n[i] - n[i - 1]) > 20 * med + 1e-6 and abs(n[i] - n[i + 1]) > 20 * med + 1e-6 and abs(n[i + 1] - n[i - 1]) < 5 * med + 1e-6:
                vals.append(abs(n[i] - 0.5 * (n[i - 1] + n[i + 1])) / n[i])
    v = np.array(vals); v = v[(v > 1e-6) & (v < 0.05)]
    return v if len(v) > 20 else np.array([2e-4, 5e-4, 1e-3])

def _set_field(line, start, end, text):
    return line[:start] + text.rjust(end - start)[: end - start] + line[end:]

def _checksum(line):
    s = sum(int(c) if c.isdigit() else (1 if c == "-" else 0) for c in line[:68]); return line[:68] + str(s % 10)

def inject(tl, p, rng, jumps):
    tl = tl.reset_index(drop=True).copy(); n = len(tl); kinds = np.array([""] * n, dtype=object)
    idx = np.where(rng.random(n) < p)[0]; extra = []
    for i in idx:
        r = rng.random(); l1, l2 = tl.at[i, "line1"], tl.at[i, "line2"]
        if r < FAULT_MIX["n_jump"]:
            dn = tl.at[i, "n"] * float(rng.choice(jumps)) * rng.choice([-1, 1]); nn = tl.at[i, "n"] + dn
            l2 = _checksum(_set_field(l2, 52, 63, f"{nn:11.8f}")); tl.at[i, "line2"] = l2; tl.at[i, "n"] = nn; kinds[i] = "n_jump"
        elif r < FAULT_MIX["n_jump"] + FAULT_MIX["bstar_neg"]:
            bs = l1[53:61]
            if bs.strip() and not bs.strip().startswith("-") and bs.strip()[:5].strip("0") != "":
                l1 = _checksum(l1[:53] + "-" + bs.lstrip(" +")[:7].rjust(7) + l1[61:]); tl.at[i, "line1"] = l1; tl.at[i, "bstar"] = -abs(tl.at[i, "bstar"]); kinds[i] = "bstar_neg"
            else:
                kinds[i] = "bstar_neg_skip"
        else:
            ep = tl.at[i, "epoch"] + pd.Timedelta(minutes=5); doy = (ep - pd.Timestamp(year=ep.year, month=1, day=1, tz="UTC")).total_seconds() / 86400 + 1
            l1d = _checksum(_set_field(l1, 18, 32, f"{ep.year % 100:02d}{doy:012.8f}")); nn = tl.at[i, "n"] * (1 + 1e-5 * rng.standard_normal())
            l2d = _checksum(_set_field(l2, 52, 63, f"{nn:11.8f}"))
            extra.append(dict(epoch=ep, line1=l1d, line2=l2d, n=nn, e=tl.at[i, "e"], inc=tl.at[i, "inc"], bstar=tl.at[i, "bstar"], kind="duplicate"))
    tl["kind"] = kinds
    if extra: tl = pd.concat([tl, pd.DataFrame(extra)], ignore_index=True)
    tl = tl.sort_values("epoch").reset_index(drop=True)
    return tl, (tl.kind != "").values & (tl.kind != "bstar_neg_skip").values, tl.kind.values

def _p2_one(args):
    name, norad, jumps = args
    times, truth, t0, t1 = _load_truth(name)
    full = read_tles(norad, t0 - pd.Timedelta(days=3), t1 + pd.Timedelta(days=3))
    rows = []; det = []
    for p in FAULT_P:
        for seed in (SEEDS if p > 0 else [0]):
            rng = np.random.default_rng(seed * 104729 + norad)
            tl, is_fault, kind = inject(full, p, rng, jumps)
            paf = paf_lite(tl); kept_ns = set(paf.epoch.values.astype("datetime64[ns]").astype("int64"))
            removed = ~np.isin(tl.epoch.values.astype("datetime64[ns]").astype("int64"), list(kept_ns))
            det.append(dict(sat=name, p=p, seed=seed, n_tle=len(tl), n_fault=int(is_fault.sum()), fault_removed=int((removed & is_fault).sum()), clean_removed=int((removed & ~is_fault).sum()),
                            **{f"recall_{k}": float((removed & (kind == k)).sum() / max((kind == k).sum(), 1)) for k in ("n_jump", "bstar_neg", "duplicate")}))
            for pool_name, pool in (("raw", tl), ("PAF-lite", paf)):
                st, _ = propagate_nearest(pool, times); e = np.linalg.norm(rtn_err(st, truth), axis=1)
                rows.append(pd.DataFrame({"sat": name, "p": p, "seed": seed, "pool": pool_name, "method": "nearest", "time": times, "e3d": e}))
                st, _ = fuse(pool, times, 3, 2.0, 12.0, adaptive=True); e = np.linalg.norm(rtn_err(st, truth), axis=1)
                rows.append(pd.DataFrame({"sat": name, "p": p, "seed": seed, "pool": pool_name, "method": "MTF-adaptive", "time": times, "e3d": e}))
                st, _ = fuse(pool, times, 3, 2.0, 12.0); e = np.linalg.norm(rtn_err(st, truth), axis=1)
                rows.append(pd.DataFrame({"sat": name, "p": p, "seed": seed, "pool": pool_name, "method": "MTF", "time": times, "e3d": e}))
    return pd.concat(rows, ignore_index=True), pd.DataFrame(det)

# ------------------------------------------------------------------ P3
def _p3_one(args):
    norad, epochs = args
    try:
        t0, t1 = to_utc(C.WINDOW_START), to_utc(C.WINDOW_END)
        raw = read_tles(norad, t0 - pd.Timedelta(days=2), t1 + pd.Timedelta(days=2)); paf = paf_lite(raw)
        kept = set(paf.epoch.values.astype("datetime64[ns]").astype("int64")); ep = raw.epoch.values.astype("datetime64[ns]").astype("int64")
        out = []
        for e in epochs:
            e_ns = np.int64(pd.Timestamp(e).value); others = ep[ep != e_ns]
            if len(others) == 0: continue
            j = np.argmin(np.abs(others - e_ns)); nb = others[j]
            n12 = int((np.abs(others - e_ns) <= 12 * 3.6e12).sum()); n24 = int((np.abs(others - e_ns) <= 24 * 3.6e12).sum())
            out.append(dict(norad=norad, epoch=e, neighbour_removed=bool(nb not in kept), neighbour_gap_h=float(abs(nb - e_ns) / 3.6e12), n_within_12h=n12, n_within_24h=n24))
        return pd.DataFrame(out)
    except Exception:
        return None

def p3(obj):
    S = pd.read_parquet(C.RESULTS / "e3_loo_samples.parquet")
    S = S[S.method.isin(["nearest-raw", "nearest-PAF", "MTF-raw", "MTF-PAF"]) & (~S.k_removed_by_paf)]
    ep_by = S[S.method == "nearest-raw"].groupby("norad").epoch.apply(lambda x: list(x.values))
    ids = set(obj.norad); out = []
    with ProcessPoolExecutor(C.N_WORKERS) as ex:
        futs = [ex.submit(_p3_one, (int(n), eps)) for n, eps in ep_by.items() if n in ids]
        for fu in as_completed(futs):
            r = fu.result()
            if r is not None and len(r): out.append(r)
    NB = pd.concat(out, ignore_index=True); NB["epoch"] = pd.to_datetime(NB.epoch, utc=True)
    piv = S.pivot_table(index=["norad", "epoch"], columns="method", values="res_3d").reset_index(); piv["epoch"] = pd.to_datetime(piv.epoch, utc=True)
    M = piv.merge(NB, on=["norad", "epoch"]).merge(obj[["norad", "type", "regime"]], on="norad", how="left")
    rows = []
    for scope, g in [("ALL", M)] + [(f"{t}|{r}", g) for (t, r), g in M.groupby(["type", "regime"])]:
        a = g[g.neighbour_removed]; b = g[~g.neighbour_removed]
        if len(a) < 20: continue
        rows.append(dict(scope=scope, n_neighbour_removed=len(a), n_neighbour_kept=len(b),
                         nearest_raw_p50_when_nb_removed=float(a["nearest-raw"].median()), nearest_PAF_p50_when_nb_removed=float(a["nearest-PAF"].median()),
                         PAF_win_rate_when_nb_removed=float((a["nearest-PAF"] < a["nearest-raw"]).mean()),
                         nearest_raw_p90_when_nb_removed=float(a["nearest-raw"].quantile(.9)), nearest_PAF_p90_when_nb_removed=float(a["nearest-PAF"].quantile(.9)),
                         nearest_raw_p50_when_nb_kept=float(b["nearest-raw"].median()), MTF_raw_p50_when_nb_removed=float(a["MTF-raw"].median()), MTF_PAF_p50_when_nb_removed=float(a["MTF-PAF"].median())))
    sp = []
    for scope, g in [("ALL", M)] + [(f"{t}|{r}", g) for (t, r), g in M.groupby(["type", "regime"]) if len(g) >= 100]:
        sp.append(dict(scope=scope, n_epochs=len(g), n_obj=g.norad.nunique(), pct_ge2_within_12h=100 * (g.n_within_12h >= 2).mean(), pct_ge2_within_24h=100 * (g.n_within_24h >= 2).mean(),
                       pct_ge1_within_12h=100 * (g.n_within_12h >= 1).mean(), median_neighbour_gap_h=float(g.neighbour_gap_h.median()), pct_neighbour_removed_by_paf=100 * g.neighbour_removed.mean()))
    return pd.DataFrame(rows), pd.DataFrame(sp)

# ------------------------------------------------------------------ P4
def p4():
    p = C.RESULTS / "t4_forecast_cases.parquet"
    if not p.exists(): return pd.DataFrame()
    cs = pd.read_parquet(p); cs = cs[cs.method == "SGP4-MTF"].copy(); cs["t0"] = pd.to_datetime(cs.t0, utc=True)
    sw = pd.read_csv(C.DATA / "space_weather_daily.csv", parse_dates=["date"]).set_index("date")
    d0 = cs.t0.dt.tz_convert(None).dt.normalize()
    cs["ap_obs"] = np.maximum(sw.ap_avg.reindex(d0).to_numpy(), sw.ap_avg.reindex(d0 + pd.Timedelta(days=1)).to_numpy())
    cs["ap_lag1"] = sw.ap_avg.reindex(d0 - pd.Timedelta(days=1)).to_numpy()
    fc = pd.read_parquet(C.RESULTS / "t3_weather_forecasts.parquet"); fc = fc[(fc.variable == "ap_avg") & (fc.method == "HistGB-AR30") & (fc.horizon_d == 1)]
    apf = fc.set_index(pd.to_datetime(fc.origin_date)).pred; cs["ap_fc"] = apf.reindex(d0 - pd.Timedelta(days=1)).to_numpy()
    cs["ap_fc"] = cs.ap_fc.fillna(cs.ap_lag1)
    t0w = to_utc(C.SENTINEL_TEST_START); tr = cs[cs.t0 < t0w]; te = cs[cs.t0 >= t0w].copy(); te["storm"] = te.ap_obs >= 40
    rows = []; gfac = {}
    for c in ("r", "t", "n"):
        ah = (tr.age_h + tr.h).values; e = tr[f"err_{c}"].values
        bins = np.quantile(ah, np.linspace(0, 1, 9)); ctr, sc = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = (ah >= lo) & (ah <= hi)
            if m.sum() > 20: ctr.append(ah[m].mean()); sc.append(1.4826 * mad(e[m]))
        b, a = np.polyfit(ctr, sc, 1)
        sig_tr = np.maximum(a + b * ah, 1e-3); z = e / sig_tr
        apb = [0, 15, 30, 50, 80, 1e9]; gfac = {}
        for lo, hi in zip(apb[:-1], apb[1:]):
            m = (tr.ap_lag1.values >= lo) & (tr.ap_lag1.values < hi)
            gfac[(lo, hi)] = 1.4826 * mad(z[m]) if m.sum() > 30 else 1.0
        def g_of(ap):
            out = np.ones(len(ap))
            for (lo, hi), v in gfac.items(): out[(ap >= lo) & (ap < hi)] = v
            return out
        ah_te = (te.age_h + te.h).values; e_te = te[f"err_{c}"].values; s0 = np.maximum(a + b * ah_te, 1e-3)
        for model, s in (("sigma-age", s0), ("sigma-age x g(Ap obs lag1)", s0 * g_of(te.ap_lag1.fillna(10).values)), ("sigma-age x g(Ap T3-forecast)", s0 * g_of(te.ap_fc.fillna(10).values)), ("sigma-age x g(Ap perfect)", s0 * g_of(te.ap_obs.fillna(10).values))):
            for split, m in (("all", np.ones(len(te), bool)), ("storm(Ap>=40)", te.storm.values), ("quiet", ~te.storm.values)):
                zz = e_te[m] / s[m]
                rows.append(dict(component=c, model=model, split=split, n=int(m.sum()), cov68=float(np.mean(np.abs(zz) <= 1)), cov95=float(np.mean(np.abs(zz) <= 1.96)), nees=float(np.mean(zz ** 2)),
                                 err_p50_km=float(np.median(np.abs(e_te[m]))), sigma_p50_km=float(np.median(s[m]))))
    out = pd.DataFrame(rows); out["g_factors_last_component"] = str({f"{k[0]}-{k[1]}": round(v, 2) for k, v in gfac.items()})
    return out

# ------------------------------------------------------------------ P5 continuity
def _p5_one(args):
    """Trajectory continuity at 1-min cadence on one Sentinel over the test window: jump statistics at TLE switch
    epochs for nearest vs blend-2 vs MTF vs MTF-adaptive, and error spikes vs POD around switches."""
    name, norad = args
    from .t4_forecast import load_pod
    pod = load_pod(name); t0, t1 = to_utc(C.SENTINEL_TEST_START), to_utc(C.WINDOW_END)
    pod = pod[(pod.index >= t0) & (pod.index < t1)]
    # 30 random days at 1-min cadence to bound cost
    days = pd.DatetimeIndex(sorted(np.random.default_rng(norad).choice(pod.index.floor("D").unique(), min(30, pod.index.floor("D").nunique()), replace=False)))
    full = read_tles(norad, t0 - pd.Timedelta(days=3), t1 + pd.Timedelta(days=3))
    rows = []
    for d in days:
        times = pod.index[(pod.index >= d) & (pod.index < d + pd.Timedelta(days=1))]
        if len(times) < 1000: continue
        truth = pod.loc[times, ["x", "y", "z", "vx", "vy", "vz"]].to_numpy()
        meth = {}
        st, age = propagate_nearest(full, times); meth["nearest"] = st
        meth["blend-2"], _ = fuse(full, times, blend2=True); meth["MTF"], _ = fuse(full, times, 3, 2.0, 12.0); meth["MTF-adaptive"], _ = fuse(full, times, 3, 2.0, 12.0, adaptive=True)
        # switch epochs of nearest: where the nearest TLE changes
        sw = np.where(np.diff(age) < -0.5)[0] + 1   # age drops -> switched to a newer TLE
        for m, s in meth.items():
            v = np.linalg.norm(np.diff(s[:, :3], axis=0), axis=1)         # 1-min displacement
            vt = np.linalg.norm(truth[1:, :3] - truth[:-1, :3], axis=1)   # truth displacement
            jump = np.abs(v - vt)                                          # kinematic inconsistency of the step (km)
            e3 = np.linalg.norm(rtn_err(s, truth), axis=1)
            for i in sw:
                if 1 <= i < len(jump):
                    rows.append(dict(sat=name, day=d, method=m, jump_km=float(jump[i - 1]), bg_jump_km=float(np.nanmedian(jump)),
                                     err_before=float(np.nanmean(e3[max(0, i - 10):i])), err_after=float(np.nanmean(e3[i:i + 10]))))
    return pd.DataFrame(rows)

def p5(sats):
    parts = []
    with ProcessPoolExecutor(min(C.N_WORKERS, len(sats))) as ex:
        for fu in as_completed([ex.submit(_p5_one, s) for s in sats]):
            r = fu.result()
            if r is not None and len(r): parts.append(r)
    J = pd.concat(parts, ignore_index=True)
    out = J.groupby("method").agg(n_switches=("jump_km", "size"), jump_p50_m=("jump_km", lambda x: 1000 * x.median()), jump_p90_m=("jump_km", lambda x: 1000 * x.quantile(.9)),
                                  jump_max_m=("jump_km", lambda x: 1000 * x.max()), bg_jump_p50_m=("bg_jump_km", lambda x: 1000 * x.median()),
                                  frac_jump_gt_50m=("jump_km", lambda x: (x > 0.05).mean()), frac_jump_gt_200m=("jump_km", lambda x: (x > 0.2).mean()),
                                  err_step_km=("err_after", "mean")).reset_index()
    out["err_change_at_switch_km"] = J.groupby("method").apply(lambda g: (g.err_after - g.err_before).abs().mean()).values
    return out, J

# ------------------------------------------------------------------ figures
def figures(tab, inj):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    BLUE, RED, YELLOW, GREEN, PURPLE = "#4477AA", "#EE6677", "#FFBE7A", "#1F7940", "#BEB8DC"
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    a = tab[(tab.sat == "ALL") & (tab.split == "all")]
    fig, ax = plt.subplots(figsize=(3.3, 2.2), constrained_layout=True)
    for m, col, ls in (("nearest", RED, "--"), ("blend-2", YELLOW, "-."), ("MTF", BLUE, "-"), ("MTF-adaptive", GREEN, "-"), ("MTF-adaptive+T2-gate", PURPLE, ":")):
        g = a[a.method == m].sort_values("keep")
        if len(g): ax.plot(1 - g.keep, g.e3d_p50, marker="o", ms=2.5, lw=0.9, ls=ls, color=col, label=m)
    ax.set_xlabel("Fraction of TLEs removed"); ax.set_ylabel("3D error P50 vs POD (km)"); ax.grid(color="#DDDDDD", lw=0.4, alpha=0.6); ax.legend(frameon=False, ncol=2)
    fig.savefig(C.FIGURES / "fig_p1_thinning.pdf"); fig.savefig(C.FIGURES / "fig_p1_thinning.png", dpi=300); plt.close(fig)
    b = inj[inj.sat == "ALL"]
    fig, ax = plt.subplots(figsize=(3.3, 2.2), constrained_layout=True)
    for (pool, meth), col, ls in ((("raw", "nearest"), RED, "--"), (("PAF-lite", "nearest"), BLUE, "--"), (("raw", "MTF"), RED, "-"), (("PAF-lite", "MTF"), BLUE, "-")):
        g = b[(b.pool == pool) & (b.method == meth)].sort_values("p")
        if len(g): ax.plot(100 * g.p, g.e3d_p90, marker="o", ms=2.5, lw=0.9, ls=ls, color=col, label=f"{pool} + {meth}")
    ax.set_xlabel("Injected fault rate (% of TLEs)"); ax.set_ylabel("3D error P90 vs POD (km)"); ax.set_yscale("log"); ax.grid(color="#DDDDDD", lw=0.4, alpha=0.6); ax.legend(frameon=False)
    fig.savefig(C.FIGURES / "fig_p2_injection.pdf"); fig.savefig(C.FIGURES / "fig_p2_injection.png", dpi=300); plt.close(fig)

def main(parts="1234", force=False):
    if stage_done("S8d_proof") and not force: log.info("done"); return
    set_status("S8d_proof", "running")
    try:
        with Timer() as T:
            obj = pd.read_csv(C.DATA / "objects.csv")
            sats = [(n, v[0]) for n, v in C.SENTINELS.items() if (C.DATA / f"sentinel_pod_{n}.parquet").exists() and _has_truth(n)]
            log.info(f"satellites with POD truth in the test window: {[n for n, _ in sats]}")
            if "1" in parts:
                parts1 = []
                with ProcessPoolExecutor(min(C.N_WORKERS, len(sats))) as ex:
                    for fu in as_completed([ex.submit(_p1_one, s) for s in sats]):
                        r = fu.result()
                        if r is not None and len(r): parts1.append(r); log.info(f"P1 {r.sat.iloc[0]}: {len(r)} rows")
                S1 = pd.concat(parts1, ignore_index=True); S1.to_parquet(C.RESULTS / "p1_thinning_samples.parquet", index=False)
                tab, gb, ci = p1_summarize(S1); save_df(tab, "p1_thinning.csv"); save_df(gb, "p1_thinning_gapbins.csv"); save_df(ci, "p1_thinning_ci.csv")
                log.info("P1 pooled P50 by keep x method:\n" + tab[(tab.sat == "ALL") & (tab.split == "all")].pivot_table(index="method", columns="keep", values="e3d_p50").round(3).to_string())
                log.info("P1 CI:\n" + ci.round(3).to_string(index=False))
            if "2" in parts:
                jumps = jump_distribution(obj); log.info(f"empirical isolated-jump |dn/n|: n={len(jumps)}, median={np.median(jumps):.2e}, p90={np.percentile(jumps, 90):.2e}")
                np.save(C.RESULTS / "p2_jump_distribution.npy", jumps)
                rows, dets = [], []
                with ProcessPoolExecutor(min(C.N_WORKERS, len(sats))) as ex:
                    for fu in as_completed([ex.submit(_p2_one, (n, nid, jumps)) for n, nid in sats]):
                        r, d = fu.result()
                        if r is not None and len(r): rows.append(r); dets.append(d); log.info(f"P2 {r.sat.iloc[0]} done")
                S2 = pd.concat(rows, ignore_index=True); D2 = pd.concat(dets, ignore_index=True)
                S2.to_parquet(C.RESULTS / "p2_injection_samples.parquet", index=False)
                inj = S2.groupby(["p", "pool", "method"]).e3d.agg(n="count", e3d_p50="median", e3d_p90=lambda x: x.quantile(.9), e3d_p99=lambda x: x.quantile(.99)).reset_index(); inj["sat"] = "ALL"
                inj_s = S2.groupby(["sat", "p", "pool", "method"]).e3d.agg(n="count", e3d_p50="median", e3d_p90=lambda x: x.quantile(.9), e3d_p99=lambda x: x.quantile(.99)).reset_index()
                inj = pd.concat([inj, inj_s], ignore_index=True); save_df(inj, "p2_injection.csv")
                D2s = D2.groupby("p").agg(n_tle=("n_tle", "sum"), n_fault=("n_fault", "sum"), fault_removed=("fault_removed", "sum"), clean_removed=("clean_removed", "sum"), recall_n_jump=("recall_n_jump", "mean"), recall_bstar_neg=("recall_bstar_neg", "mean"), recall_duplicate=("recall_duplicate", "mean")).reset_index()
                D2s["fault_recall"] = D2s.fault_removed / D2s.n_fault.clip(lower=1); D2s["false_removal_rate"] = D2s.clean_removed / (D2s.n_tle - D2s.n_fault).clip(lower=1)
                save_df(D2s, "p2_paf_detection.csv")
                log.info("P2 pooled:\n" + inj[inj.sat == "ALL"].round(3).to_string(index=False)); log.info("P2 detection:\n" + D2s.round(3).to_string(index=False))
            if "3" in parts and (C.RESULTS / "e3_loo_samples.parquet").exists():
                nb, sp = p3(obj); save_df(nb, "p3_neighbour_effect.csv"); save_df(sp, "p3_sparsity.csv")
                log.info("P3 neighbour effect:\n" + nb.round(4).to_string(index=False)); log.info("P3 sparsity:\n" + sp.round(2).to_string(index=False))
            if "5" in parts:
                c5, J5 = p5(sats); save_df(c5, "p5_continuity.csv"); J5.to_parquet(C.RESULTS / "p5_continuity_samples.parquet", index=False)
                log.info("P5 continuity:\n" + c5.round(4).to_string(index=False))
            if "4" in parts:
                st = p4()
                if len(st): save_df(st, "p4_storm_sigma.csv"); log.info("P4 storm sigma (T component):\n" + st[st.component == "t"].round(3).to_string(index=False))
            try:
                tab = pd.read_csv(C.RESULTS / "p1_thinning.csv"); inj = pd.read_csv(C.RESULTS / "p2_injection.csv"); figures(tab, inj)
            except Exception as ex: log.warning(f"figures skipped: {ex}")
        set_status("S8d_proof", "done", elapsed_s=round(T.elapsed))
    except Exception as ex:
        log.exception("proof failed"); set_status("S8d_proof", "failed", message=str(ex)); raise

if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    main(parts=args[0] if args else "12354", force="--force" in sys.argv)
