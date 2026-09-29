"""End-to-end chain experiments on the Starlink truth cohort (20 satellites, MEME operator ephemerides with 6x6
covariance, raw + official step6/PAF TLEs to 2026-08-24).  Window 2026-06-10 .. 2026-08-24; train (thresholds,
sigma fits) 06-10 .. 07-05, test 07-05 .. 08-23.

S1  Forecast + UQ (causal, origins every ORIGIN_H, horizons HZ), truth = MEME (J2000):
      A0  raw nearest-prev TLE, fixed sigma                    (independent task 4)
      +PAF          step6 nearest-prev
      +MTF          step6 causal MTF (k3, tau 2 h, W 12 h)
      +T2 gate      step6 MTF using only TLEs after the last T2-detected maneuver (48 h look-back)
      +T1 window    ... and W = 6 h when T1 phase != STABLE (semi-major-axis trend over the last 5 TLEs)
      raw MTF, raw MTF+T2 (to separate PAF from fusion)
    sigma: fixed | age | age x g(T2 score) ; metrics RTN P50/P90, coverage/NEES; also NEES against the operator
    covariance alone (shows the operator covariance describes MEME, not TLE error).
    T2 truth for the detector: MEME-derived thrust arcs (|d a_mean/dt| of daily-smoothed SMA above threshold).
S2  T6 on Starlink pairs (190 pairs): reference approaches from MEME (< REF_KM) with TRUE encounter-plane Pc from
    the two MEME covariances; predicted approaches from A0 vs A1 trajectories at lead 6/24 h; matching, ranking by
    Pc(OrbBench sigma) vs point miss; recall of true-Pc>=1e-5 events.
S3  T7 on matched events: Pc-gate with OrbBench sigma vs distance rule; safety by TRUE Pc from MEME covariance.
S4  1B data ablation table on Starlink (B0 full, B1 no maneuver info, B4 no precise reference for calibration,
    B5 TLE only) and the end-to-end A0 vs A1 table.
Outputs: results/s1_starlink_forecast.csv, s1_starlink_uq.csv, s1_starlink_cases.parquet, s1_t2_detection.csv,
         s2_starlink_events.csv, s2_starlink_ranking.csv, s3_starlink_avoidance.csv, s4_starlink_ablation.csv,
         starlink_a0_a1.csv ; figures/fig_s1_starlink_forecast.pdf
"""
from __future__ import annotations
import sys, os, json
os.environ.setdefault("MPLCONFIGDIR", os.path.join(os.path.expanduser("~"), ".cache", "orbbench-mpl"))
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
from scipy.stats import spearmanr

from . import config as C
from .util import get_logger, set_status, stage_done, Timer, save_df, sgp4_states, teme_to_gcrs, to_rtn, rtn_basis, mad, quantiles, clean_line, tle_epoch
from .stage_starlink import RAW_TLE_DIR, STEP6_TLE_DIR, starlink_ids
from .orbit import classical_elements

log = get_logger("starlink_chain")
W0, W1 = pd.Timestamp("2026-06-10", tz="UTC"), pd.Timestamp("2026-08-24", tz="UTC")
TRAIN_END = pd.Timestamp("2026-07-05", tz="UTC")
ORIGIN_H = 3; HZ = [1.0, 4.0, 12.0, 24.0]
K, TAU, WIN = 3, 2.0, 12.0
THRUST_KM_PER_DAY = 0.15          # |d a_mean / dt| above this = thrust arc (truth for T2); station-keeping is ~0.02 km/day
REF_KM = 50.0; PRED_KM = 100.0; TCA_TOL_MIN = 10.0; HBR_KM = 0.010   # two Starlink v1: ~10 m combined hard body

# ------------------------------------------------------------------ IO
def read_txt(p, start=None, end=None) -> pd.DataFrame:
    lines = open(p, encoding="utf-8", errors="replace").read().splitlines(); rows = []
    for i in range(len(lines) - 1):
        if lines[i].startswith("1 ") and lines[i + 1].startswith("2 "):
            l1, l2 = clean_line(lines[i]), clean_line(lines[i + 1])
            try:
                bs = l1[53:61]; m = bs[:6].strip(); ex = bs[6:8]; bstar = (-1 if m.startswith("-") else 1) * float("0." + m.lstrip("+-")) * 10 ** int(ex)
                rows.append((tle_epoch(l1), l1, l2, float(l2[52:63]), float("0." + l2[26:33].strip()), float(l2[8:16]), bstar))
            except Exception: pass
    d = pd.DataFrame(rows, columns=["epoch", "line1", "line2", "n", "e", "inc", "bstar"]); d["epoch"] = pd.to_datetime(d.epoch, utc=True)
    d = d.sort_values("epoch").drop_duplicates("epoch", keep="last")
    if start is not None: d = d[d.epoch >= start]
    if end is not None: d = d[d.epoch <= end]
    return d.reset_index(drop=True)

def load_meme(norad) -> pd.DataFrame:
    m = pd.read_parquet(C.DATA / f"starlink/meme_{norad}.parquet"); m["time"] = pd.to_datetime(m.time, utc=True)
    return m.set_index("time").sort_index()

def cov_pos_rtn(row) -> np.ndarray:
    """3x3 position covariance (UVW = R,T,N) from the lower-triangle 21 values c0..c20 (km^2)."""
    c = [row[f"c{k}"] for k in range(21)]
    P = np.zeros((6, 6)); idx = 0
    for i in range(6):
        for j in range(i + 1):
            P[i, j] = P[j, i] = c[idx]; idx += 1
    return P[:3, :3]

def thrust_arcs(m: pd.DataFrame) -> pd.Series:
    """Boolean per MEME minute: inside a thrust arc, from the slope of the 1-day rolling-mean SMA."""
    el = classical_elements(m[["x", "y", "z"]].to_numpy(), m[["vx", "vy", "vz"]].to_numpy())
    a = pd.Series(el["a_km"], index=m.index)
    a_d = a.resample("12h").mean().interpolate(); slope = a_d.diff() / 0.5           # km/day, 12-h means average ~7.6 orbits
    arc = (slope.abs() > THRUST_KM_PER_DAY)
    return arc.reindex(m.index, method="nearest").fillna(False)

# ------------------------------------------------------------------ T2 / T1 from TLEs
def t2_scores(tles: pd.DataFrame) -> np.ndarray:
    if len(tles) < 4: return np.full(len(tles), np.nan)
    own = np.vstack([sgp4_states(r.line1, r.line2, [r.epoch]) for r in tles.itertuples()])
    res = np.full(len(tles), np.nan)
    for i in range(1, len(tles)):
        p = tles.iloc[i - 1]; st = sgp4_states(p.line1, p.line2, [tles.epoch.iloc[i]])[0]
        if np.isfinite(st).all() and np.isfinite(own[i]).all():
            res[i] = to_rtn((st[:3] - own[i, :3])[None], own[i, :3][None], own[i, 3:][None])[0][1]
    return np.abs(res) / max(mad(res), 1e-9)

def t1_phase(tles: pd.DataFrame, k=5) -> np.ndarray:
    """Per-TLE phase from SMA trend over the previous k TLEs: 1 = raise, -1 = lower, 0 = stable."""
    a = (C.MU_KM3_S2 / ((tles.n.values * 2 * np.pi / 86400) ** 2)) ** (1 / 3); out = np.zeros(len(tles), int)
    t = tles.epoch.values.astype("datetime64[ns]").astype("int64") / 8.64e13
    for i in range(k, len(tles)):
        x = t[i - k:i + 1]; y = a[i - k:i + 1]; b = np.polyfit(x - x[0], y, 1)[0]      # km/day
        out[i] = 1 if b > THRUST_KM_PER_DAY else (-1 if b < -THRUST_KM_PER_DAY else 0)
    return out

def paf_aware(t: pd.DataFrame, aware: bool) -> pd.DataFrame:
    """PAF-lite on a TLE table. aware=True: the B*<0 and isolated-n rules are applied only where the T1 phase is
    stable (phase==0); during detected raise/lower arcs those TLEs are kept (they describe a real maneuver)."""
    t = t.sort_values("epoch").reset_index(drop=True); n = len(t)
    if n < 3: return t
    keep = np.ones(n, bool)
    ns = t.epoch.values.astype("datetime64[ns]").astype("int64"); dt = np.diff(ns) / 1e9; T = 86400.0 / t.n.values
    keep[:-1] &= ~(dt < 0.5 * T[1:])
    stable = (t.phase.values == 0) if aware else np.ones(n, bool)
    keep &= ~((t.bstar.values < 0) & stable)
    nn = t.n.values; dn = np.abs(np.diff(nn)); med = np.median(dn) if len(dn) else 0
    for i in range(1, n - 1):
        if stable[i] and abs(nn[i] - nn[i - 1]) > 20 * med + 1e-6 and abs(nn[i] - nn[i + 1]) > 20 * med + 1e-6 and abs(nn[i + 1] - nn[i - 1]) < 5 * med + 1e-6: keep[i] = False
    return t[keep].reset_index(drop=True)

# ------------------------------------------------------------------ fusion (causal)
def causal_fuse(tles, t0, targets, k=K, tau=TAU, win=WIN, after=None, phase_win=None):
    c = tles[tles.epoch <= t0]
    if after is not None:
        c2 = c[c.epoch >= after]
        if len(c2): c = c2
    w = phase_win if phase_win is not None else win
    cc = c[c.epoch >= t0 - pd.Timedelta(hours=w)]
    if len(cc) == 0: cc = c.tail(1)
    cc = cc.tail(k)
    if len(cc) == 0: return np.full((len(targets), 6), np.nan), np.nan, 0
    S, Wt = [], []
    for r in cc.itertuples():
        st = sgp4_states(r.line1, r.line2, targets); age = (t0 - r.epoch).total_seconds() / 3600
        if np.isfinite(st).all(): S.append(st); Wt.append(np.exp(-age / tau))
    if not S: return np.full((len(targets), 6), np.nan), np.nan, 0
    S = np.array(S); Wt = np.array(Wt) / np.sum(Wt)
    return np.einsum("m,mtk->tk", Wt, S), float(np.sum(Wt * [(t0 - r.epoch).total_seconds() / 3600 for r in cc.itertuples()][: len(S)])), len(S)

# ------------------------------------------------------------------ S1 per satellite
def _s1_one(args):
    norad, k_t2 = args
    try:
        m = load_meme(norad); arc = thrust_arcs(m)
        raw = read_txt(RAW_TLE_DIR / f"{norad}.txt", W0 - pd.Timedelta(days=3), W1); s6p = STEP6_TLE_DIR / f"{norad}.txt"
        s6 = read_txt(s6p, W0 - pd.Timedelta(days=3), W1) if s6p.exists() else raw.copy()
        for d in (raw, s6):
            d["t2"] = t2_scores(d); d["phase"] = t1_phase(d)
        paf_blind = paf_aware(raw, aware=False); paf_t1 = paf_aware(raw, aware=True)      # PAF-lite without / with T1 protection
        retention = dict(step6=len(s6) / max(len(raw), 1), paf_blind=len(paf_blind) / max(len(raw), 1), paf_t1=len(paf_t1) / max(len(raw), 1))
        origins = pd.date_range(W0 + pd.Timedelta(days=2), W1 - pd.Timedelta(hours=24), freq=f"{ORIGIN_H}h", tz="UTC")
        rows = []; det = []
        # T2 detector evaluation against MEME thrust arcs: per TLE interval, positive if arc active anywhere in (prev epoch, epoch]
        for pool, d in (("raw", raw), ("step6", s6)):
            for i in range(1, len(d)):
                lo, hi = d.epoch.iloc[i - 1], d.epoch.iloc[i]
                seg = arc[(arc.index > lo) & (arc.index <= hi)]
                if len(seg) == 0: continue
                det.append(dict(norad=norad, pool=pool, epoch=hi, t2=float(d.t2.iloc[i]), truth=bool(seg.mean() > 0.5), phase=int(d.phase.iloc[i]), is_test=bool(hi >= TRAIN_END)))
        for t0 in origins:
            nominal = pd.DatetimeIndex([t0 + pd.Timedelta(hours=h) for h in HZ])
            idx = m.index.get_indexer(nominal, method="nearest", tolerance=pd.Timedelta(seconds=90))
            if (idx < 0).any(): continue
            targets = m.index[idx]                                   # exact MEME epochs (sub-minute offsets matter: 7.5 km/s)
            truth = m.iloc[idx][["x", "y", "z", "vx", "vy", "vz"]].to_numpy(float)
            in_arc = bool(arc.iloc[idx].any()); arc_recent = bool(arc[(arc.index <= t0) & (arc.index >= t0 - pd.Timedelta(hours=48))].any())
            meth = {}
            pr = raw[raw.epoch <= t0].tail(1)
            if len(pr) == 0: continue
            meth["A0 raw prev"] = (sgp4_states(pr.line1.iloc[0], pr.line2.iloc[0], targets), (t0 - pr.epoch.iloc[0]).total_seconds() / 3600, 1)
            p6 = s6[s6.epoch <= t0].tail(1)
            if len(p6): meth["+PAF prev"] = (sgp4_states(p6.line1.iloc[0], p6.line2.iloc[0], targets), (t0 - p6.epoch.iloc[0]).total_seconds() / 3600, 1)
            meth["raw MTF"] = causal_fuse(raw, t0, targets)
            meth["+PAF +MTF"] = causal_fuse(s6, t0, targets)
            meth["+PAF-lite(blind) +MTF"] = causal_fuse(paf_blind, t0, targets)
            meth["+PAF-lite(T1-aware) +MTF"] = causal_fuse(paf_t1, t0, targets)
            # fallback gate (PDF G1-style): use step6 pool only if it has a TLE within 24 h of t0, else raw
            recent6 = s6[(s6.epoch <= t0) & (s6.epoch >= t0 - pd.Timedelta(hours=24))]
            meth["+PAF +MTF +gate"] = causal_fuse(s6 if len(recent6) else raw, t0, targets)
            # T2 gate: last flagged TLE within 48 h
            def last_flag(d):
                f = d[(d.epoch <= t0) & (d.epoch >= t0 - pd.Timedelta(hours=48)) & (d.t2 > k_t2)]
                return f.epoch.max() if len(f) else None
            meth["raw MTF +T2"] = causal_fuse(raw, t0, targets, after=last_flag(raw))
            meth["+PAF +MTF +T2"] = causal_fuse(s6, t0, targets, after=last_flag(s6))
            ph = s6[s6.epoch <= t0].tail(1).phase.iloc[0] if len(s6[s6.epoch <= t0]) else 0
            meth["+PAF +MTF +T2 +T1"] = causal_fuse(s6, t0, targets, after=last_flag(s6), phase_win=(6.0 if ph != 0 else None))
            t2_now = float(s6[(s6.epoch <= t0) & (s6.epoch >= t0 - pd.Timedelta(hours=48))].t2.max()) if len(s6[(s6.epoch <= t0) & (s6.epoch >= t0 - pd.Timedelta(hours=48))]) else 0.0
            rr = raw[raw.epoch <= t0].tail(6)
            if len(rr) >= 3:
                aa = (C.MU_KM3_S2 / ((rr.n.values * 2 * np.pi / 86400) ** 2)) ** (1 / 3); tt_ = rr.epoch.values.astype("datetime64[ns]").astype("int64") / 8.64e13
                sma_rate = float(np.polyfit(tt_ - tt_[0], aa, 1)[0])
            else: sma_rate = 0.0
            ndot = float(pr.line1.iloc[0][33:43]) if pr.line1.iloc[0][33:43].strip() else 0.0
            for name, (st, age, nm) in meth.items():
                if not np.isfinite(st).all(): continue
                g = teme_to_gcrs(st, targets)
                err = to_rtn(g[:, :3] - truth[:, :3], truth[:, :3], truth[:, 3:])
                for j, h in enumerate(HZ):
                    row = dict(norad=norad, t0=t0, h=h, method=name, age_h=age, n_members=nm, err_r=err[j, 0], err_t=err[j, 1], err_n=err[j, 2], err_3d=float(np.linalg.norm(err[j])),
                               in_arc=in_arc, arc_recent=arc_recent, t2_score=t2_now, phase=int(ph), sma_rate_km_d=sma_rate, ndot=ndot, bstar=float(pr.bstar.iloc[0]), is_test=bool(t0 >= TRAIN_END),
                               ret_step6=retention["step6"], ret_paf_t1=retention["paf_t1"],
                               op_sig_u=float(m.iloc[idx[j]].sig_u_m) / 1000, op_sig_v=float(m.iloc[idx[j]].sig_v_m) / 1000, op_sig_w=float(m.iloc[idx[j]].sig_w_m) / 1000)
                    rows.append(row)
        return norad, pd.DataFrame(rows), pd.DataFrame(det)
    except Exception as ex:
        import traceback; return norad, str(ex) + "\n" + traceback.format_exc(), None

def tune_k_t2(det: pd.DataFrame) -> float:
    """Pick the T2 threshold that maximises F2 on train intervals (step6 pool) against MEME thrust arcs."""
    d = det[(det.pool == "step6") & (~det.is_test) & det.t2.notna()]
    best = (6.0, -1)
    for k in np.arange(2, 30, 0.5):
        p = d.t2 > k; tp = (p & d.truth).sum(); P = tp / max(p.sum(), 1); R = tp / max(d.truth.sum(), 1); f2 = 5 * P * R / max(4 * P + R, 1e-9)
        if f2 > best[1]: best = (float(k), f2)
    return best[0]

def s1_summaries(cases: pd.DataFrame):
    te = cases[cases.is_test]; tr = cases[~cases.is_test]
    rows = []
    def agg(g, **tag):
        r = dict(n=len(g), **tag)
        for c in ("3d", "t", "r", "n"):
            x = g.err_3d.values if c == "3d" else np.abs(g[f"err_{c}"].values)
            r[f"{c}_p50_km"] = float(np.percentile(x, 50)); r[f"{c}_p90_km"] = float(np.percentile(x, 90))
        return r
    for (m, h), g in te.groupby(["method", "h"]): rows.append(agg(g, method=m, h=h, split="all"))
    for (m, h, a), g in te.groupby(["method", "h", "arc_recent"]): rows.append(agg(g, method=m, h=h, split="post-thrust(48h)" if a else "no-recent-thrust"))
    for (m, h, a), g in te.groupby(["method", "h", "in_arc"]): rows.append(agg(g, method=m, h=h, split="thrust-in-horizon" if a else "coast"))
    fc = pd.DataFrame(rows)
    # per-satellite win rate of A1 vs A0 (median 3D error per sat per horizon)
    per = te.groupby(["norad", "h", "method"]).err_3d.median().unstack("method")
    wins = []
    for h in HZ:
        p = per.xs(h, level="h")
        for m in [c for c in p.columns if c != "A0 raw prev"]:
            wins.append(dict(h=h, method=m, n_sat=len(p), win_rate_vs_A0=float((p[m] < p["A0 raw prev"]).mean()), median_rel_change_pct=float(100 * ((p[m] - p["A0 raw prev"]) / p["A0 raw prev"]).median())))
    # UQ: fit on train for the OrbBench product (+PAF +MTF +T2), evaluate on test
    uq = []; calib = {}
    for prod in ("A0 raw prev", "+PAF +MTF", "+PAF +MTF +T2"):
        trp = tr[tr.method == prod]; tep = te[te.method == prod].copy()
        for c in ("r", "t", "n"):
            e = trp[f"err_{c}"].values; ah = (trp.age_h + trp.h).values
            s_fixed = 1.4826 * mad(e)
            bins = np.quantile(ah, np.linspace(0, 1, 9)); ctr, sc = [], []
            for lo, hi in zip(bins[:-1], bins[1:]):
                mm = (ah >= lo) & (ah <= hi)
                if mm.sum() > 30: ctr.append(ah[mm].mean()); sc.append(1.4826 * mad(e[mm]))
            b, a = np.polyfit(ctr, sc, 1) if len(ctr) > 2 else (0.0, s_fixed)
            z = e / np.maximum(a + b * ah, 1e-3)
            g_t2 = {}
            for lo, hi in ((0, 3), (3, 6), (6, 12), (12, 1e9)):
                mm = (trp.t2_score.values >= lo) & (trp.t2_score.values < hi); g_t2[(lo, hi)] = (1.4826 * mad(z[mm])) if mm.sum() > 50 else 1.0
            base = g_t2[(0, 3)] if g_t2[(0, 3)] > 0 else 1.0
            g_t2 = {k: v / base for k, v in g_t2.items()}
            calib[f"{prod}|{c}"] = dict(fixed=s_fixed, age_a=a, age_b=b, g_t2={f"{k[0]}-{k[1]}": v for k, v in g_t2.items()})
            ah_te = (tep.age_h + tep.h).values; e_te = tep[f"err_{c}"].values
            s_age = np.maximum(a + b * ah_te, 1e-3) * base
            gfac = np.ones(len(tep))
            for (lo, hi), v in g_t2.items(): gfac[(tep.t2_score.values >= lo) & (tep.t2_score.values < hi)] = v
            op = tep[{"r": "op_sig_u", "t": "op_sig_v", "n": "op_sig_w"}[c]].values
            for model, s in (("sigma-fixed", np.full(len(tep), s_fixed)), ("sigma-age", s_age), ("sigma-age x g(T2)", s_age * gfac), ("operator covariance (MEME) alone", np.maximum(op, 1e-4))):
                for split, mm in (("all", np.ones(len(tep), bool)), ("post-thrust(48h)", tep.arc_recent.values), ("no-recent-thrust", ~tep.arc_recent.values)):
                    for h in [None] + HZ:
                        mk = mm & ((tep.h.values == h) if h is not None else True)
                        if mk.sum() < 30: continue
                        zz = e_te[mk] / s[mk]
                        uq.append(dict(product=prod, component=c, model=model, split=split, h=("all" if h is None else h), n=int(mk.sum()), cov68=float(np.mean(np.abs(zz) <= 1)), cov95=float(np.mean(np.abs(zz) <= 1.96)),
                                       nees=float(np.mean(zz ** 2)), err_p50_km=float(np.median(np.abs(e_te[mk]))), sigma_p50_km=float(np.median(s[mk]))))
    (C.RESULTS / "s1_starlink_calibration.json").write_text(json.dumps(calib, indent=1, default=float))
    # residual correctors (chained T1/T2 features vs orbit-only), trained on train origins of the product trajectory
    from sklearn.ensemble import HistGradientBoostingRegressor
    corr_rows = []
    base_name = "+PAF-lite(T1-aware) +MTF"
    if base_name not in set(cases.method): base_name = "raw MTF"
    trb = tr[tr.method == base_name]; teb = te[te.method == base_name].copy()
    feats = {"orbit-only": ["age_h", "h", "bstar", "ndot"], "+T1 (SMA rate, phase)": ["age_h", "h", "bstar", "ndot", "sma_rate_km_d", "phase"], "+T1 +T2": ["age_h", "h", "bstar", "ndot", "sma_rate_km_d", "phase", "t2_score"]}
    for fname, cols in feats.items():
        pred = {}
        for c in ("r", "t", "n"):
            mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=C.SEED).fit(trb[cols], trb[f"err_{c}"])
            pred[c] = teb[f"err_{c}"].values - mdl.predict(teb[cols])
        e3 = np.sqrt(sum(pred[c] ** 2 for c in ("r", "t", "n")))
        for h in HZ:
            mk = teb.h.values == h
            corr_rows.append(dict(base=base_name, corrector=fname, h=h, n=int(mk.sum()), base_p50_km=float(np.median(teb.err_3d.values[mk])), corrected_p50_km=float(np.median(e3[mk])),
                                  base_p90_km=float(np.percentile(teb.err_3d.values[mk], 90)), corrected_p90_km=float(np.percentile(e3[mk], 90)), t_abs_p50_base=float(np.median(np.abs(teb.err_t.values[mk]))), t_abs_p50_corr=float(np.median(np.abs(pred["t"][mk])))))
            for a_ in (True, False):
                mk2 = mk & (teb.arc_recent.values == a_)
                if mk2.sum() > 30: corr_rows.append(dict(base=base_name, corrector=fname, h=h, split="post-thrust" if a_ else "coast", n=int(mk2.sum()), base_p50_km=float(np.median(teb.err_3d.values[mk2])), corrected_p50_km=float(np.median(e3[mk2]))))
    CR = pd.DataFrame(corr_rows); save_df(CR, "s1_starlink_correctors.csv")
    return fc, pd.DataFrame(wins), pd.DataFrame(uq), calib

# ------------------------------------------------------------------ S2/S3: conjunctions among the Starlink cohort
def pc_encounter_plane(rel_r: np.ndarray, rel_v: np.ndarray, P_rel: np.ndarray, hbr_km: float, n=121) -> float:
    """2-D Gaussian collision probability on the encounter plane with a general covariance (Foster-type integral)."""
    v = rel_v / max(np.linalg.norm(rel_v), 1e-12)
    u = np.array([1.0, 0, 0]) if abs(v[0]) < 0.9 else np.array([0, 1.0, 0])
    e1 = np.cross(v, u); e1 /= np.linalg.norm(e1); e2 = np.cross(v, e1)
    B = np.vstack([e1, e2]); mu = B @ rel_r; P2 = B @ P_rel @ B.T; P2 = 0.5 * (P2 + P2.T) + np.eye(2) * 1e-12
    try: Pi = np.linalg.inv(P2); det = np.linalg.det(P2)
    except np.linalg.LinAlgError: return float(np.linalg.norm(mu) < hbr_km)
    xs = np.linspace(-hbr_km, hbr_km, n); X, Y = np.meshgrid(xs, xs); disk = X ** 2 + Y ** 2 <= hbr_km ** 2
    d = np.stack([X - mu[0], Y - mu[1]], -1); q = np.einsum("...i,ij,...j->...", d, Pi, d)
    pdf = np.exp(-0.5 * q) / (2 * np.pi * np.sqrt(det)); dA = (xs[1] - xs[0]) ** 2
    return float(np.sum(pdf[disk]) * dA)

def _hermite(P0, V0, P1, V1, h, s):
    s = s[:, None]; h00 = 2 * s ** 3 - 3 * s ** 2 + 1; h10 = s ** 3 - 2 * s ** 2 + s; h01 = -2 * s ** 3 + 3 * s ** 2; h11 = s ** 3 - s ** 2
    return h00 * P0 + h10 * h * V0 + h01 * P1 + h11 * h * V1

def approaches(Xa, Xb, times, thr_km, gate_km=400.0):
    """Close approaches between two 1-min trajectories [T,6] km: list of (idx, tca, miss, rel_r, rel_v)."""
    d = np.linalg.norm(Xa[:, :3] - Xb[:, :3], axis=1); out = []
    if not np.isfinite(d).any(): return out
    loc = np.where((d[1:-1] <= d[:-2]) & (d[1:-1] <= d[2:]) & (d[1:-1] <= gate_km))[0] + 1
    for k in loc:
        best = (np.inf, None, None, None, None)
        for kk in (k - 1, k):
            if kk < 0 or kk + 1 >= len(Xa) or not (np.isfinite(Xa[kk:kk + 2]).all() and np.isfinite(Xb[kk:kk + 2]).all()): continue
            s = np.linspace(0, 1, 61); pa = _hermite(Xa[kk, :3], Xa[kk, 3:], Xa[kk + 1, :3], Xa[kk + 1, 3:], 60.0, s); pb = _hermite(Xb[kk, :3], Xb[kk, 3:], Xb[kk + 1, :3], Xb[kk + 1, 3:], 60.0, s)
            dd = np.linalg.norm(pa - pb, axis=1); j = int(np.argmin(dd))
            if dd[j] < best[0]:
                va = Xa[kk, 3:] * (1 - s[j]) + Xa[kk + 1, 3:] * s[j]; vb = Xb[kk, 3:] * (1 - s[j]) + Xb[kk + 1, 3:] * s[j]
                best = (dd[j], times[kk] + pd.Timedelta(seconds=float(s[j] * 60)), kk, pa[j] - pb[j], va - vb)
        if best[1] is not None and best[0] < thr_km: out.append(best)
    return out

def s2_s3(calib: dict, k_t2: float):
    ids = [n for n in starlink_ids() if (C.DATA / f"starlink/meme_{n}.parquet").exists()]
    grid = pd.date_range(TRAIN_END, W1, freq="1min", tz="UTC")
    M = {}; TR = {}; RAW = {}; S6 = {}
    from .stage0_staging import hermite_to_grid
    for n in ids:
        m = load_meme(n)
        mm = m.rename(columns={"x": "X", "y": "Y", "z": "Z", "vx": "VX", "vy": "VY", "vz": "VZ"}).copy()
        for c in ("X", "Y", "Z", "VX", "VY", "VZ"): mm[c] = mm[c] * 1000.0          # hermite_to_grid expects metres
        st, ok = hermite_to_grid(mm, grid, max_gap_s=90.0)                             # MEME is 1-min; returns km, km/s
        X = np.full((len(grid), 6), np.nan); X[ok] = st
        M[n] = X; TR[n] = m
        RAW[n] = read_txt(RAW_TLE_DIR / f"{n}.txt", W0 - pd.Timedelta(days=3), W1); p6 = STEP6_TLE_DIR / f"{n}.txt"; S6[n] = read_txt(p6, W0 - pd.Timedelta(days=3), W1) if p6.exists() else RAW[n]
        S6[n]["t2"] = t2_scores(S6[n])
    # reference approaches + true Pc from MEME covariances
    ev = []
    for a in range(len(ids)):
        for b in range(a + 1, len(ids)):
            for miss, tca, kk, rr, rv in approaches(M[ids[a]], M[ids[b]], grid, REF_KM):
                ra, rb = TR[ids[a]].iloc[TR[ids[a]].index.get_indexer([tca], method="nearest")[0]], TR[ids[b]].iloc[TR[ids[b]].index.get_indexer([tca], method="nearest")[0]]
                Pa = cov_pos_rtn(ra); Pb = cov_pos_rtn(rb)
                Ba = rtn_basis(ra[["x", "y", "z"]].to_numpy(float)[None], ra[["vx", "vy", "vz"]].to_numpy(float)[None])[0]; Bb = rtn_basis(rb[["x", "y", "z"]].to_numpy(float)[None], rb[["vx", "vy", "vz"]].to_numpy(float)[None])[0]
                P = Ba.T @ Pa @ Ba + Bb.T @ Pb @ Bb
                ev.append(dict(i=ids[a], j=ids[b], tca=tca, miss_km=miss, rel_speed=float(np.linalg.norm(rv)), pc_true=pc_encounter_plane(rr, rv, P, HBR_KM), sig_true_km=float(np.sqrt(np.trace(P) / 3)),
                               rx=rr[0], ry=rr[1], rz=rr[2], vx=rv[0], vy=rv[1], vz=rv[2]))
    EV = pd.DataFrame(ev); EV.to_csv(C.RESULTS / "s2_starlink_events.csv", index=False)
    log.info(f"S2: {len(ids)} Starlinks, {len(EV)} reference approaches < {REF_KM} km in test period; miss quantiles {EV.miss_km.quantile([.1,.5,.9]).round(2).tolist() if len(EV) else '-'}; true Pc>=1e-6: {(EV.pc_true>=1e-6).sum() if len(EV) else 0}")
    # predicted trajectories at lead L (issue time = grid[0] - L): A0 raw prev vs A1 step6 MTF + T2 gate
    rows = []; ev_pred = []
    sig_age = lambda prod, c, ah: max(calib[f"{prod}|{c}"]["age_a"] + calib[f"{prod}|{c}"]["age_b"] * ah, 1e-3)
    for L in (6, 24):
        t_issue = grid[0] - pd.Timedelta(hours=L)
        for label, use_s6, gate in (("A0 raw prev", False, False), ("A1 PAF+MTF+T2", True, True)):
            PRED = {}; AGE = {}; T2S = {}
            for n in ids:
                d = S6[n] if use_s6 else RAW[n]
                if gate:
                    f = d[(d.epoch <= t_issue) & (d.epoch >= t_issue - pd.Timedelta(hours=48)) & (d.t2 > k_t2)]
                    st, age, _ = causal_fuse(d, t_issue, grid, after=(f.epoch.max() if len(f) else None))
                    T2S[n] = float(d[(d.epoch <= t_issue) & (d.epoch >= t_issue - pd.Timedelta(hours=48))].t2.max()) if len(d[(d.epoch <= t_issue) & (d.epoch >= t_issue - pd.Timedelta(hours=48))]) else 0.0
                else:
                    pr = d[d.epoch <= t_issue].tail(1)
                    st = sgp4_states(pr.line1.iloc[0], pr.line2.iloc[0], grid) if len(pr) else np.full((len(grid), 6), np.nan); age = (t_issue - pr.epoch.iloc[0]).total_seconds() / 3600 if len(pr) else np.nan; T2S[n] = 0.0
                PRED[n] = teme_to_gcrs(st, grid); AGE[n] = age
            det = []
            for a in range(len(ids)):
                for b in range(a + 1, len(ids)):
                    for miss, tca, kk, rr, rv in approaches(PRED[ids[a]], PRED[ids[b]], grid, PRED_KM):
                        lead_h = (tca - t_issue).total_seconds() / 3600
                        prod = "+PAF +MTF +T2" if use_s6 else "A0 raw prev"
                        def sig(n):
                            ah = AGE[n] + lead_h; s = np.array([sig_age(prod, c, ah) for c in ("r", "t", "n")])
                            if gate:
                                g = calib[f"{prod}|t"]["g_t2"]; sc = T2S[n]; key = "0-3" if sc < 3 else ("3-6" if sc < 6 else ("6-12" if sc < 12 else "12-1000000000.0"))
                                s = s * g.get(key, 1.0)
                            return s
                        sa, sb = sig(ids[a]), sig(ids[b])
                        Ba = rtn_basis(PRED[ids[a]][kk, :3][None], PRED[ids[a]][kk, 3:][None])[0]; Bb = rtn_basis(PRED[ids[b]][kk, :3][None], PRED[ids[b]][kk, 3:][None])[0]
                        P = Ba.T @ np.diag(sa ** 2) @ Ba + Bb.T @ np.diag(sb ** 2) @ Bb
                        s_fixed = np.array([calib[f"{prod}|{c}"]["fixed"] for c in ("r", "t", "n")]); Pf = Ba.T @ np.diag(s_fixed ** 2) @ Ba + Bb.T @ np.diag(s_fixed ** 2) @ Bb
                        det.append(dict(lead_h=L, chain=label, i=ids[a], j=ids[b], tca=tca, miss_km=miss, pc_orb=pc_encounter_plane(rr, rv, P, HBR_KM), pc_fixed=pc_encounter_plane(rr, rv, Pf, HBR_KM),
                                        sig_orb_km=float(np.sqrt(np.trace(P) / 3)), rx=rr[0], ry=rr[1], rz=rr[2], vx=rv[0], vy=rv[1], vz=rv[2]))
            D = pd.DataFrame(det)
            if len(D) == 0: rows.append(dict(lead_h=L, chain=label, n_pred=0)); continue
            D["true_miss"] = np.nan; D["true_pc"] = np.nan
            for di, d in D.iterrows():
                mm = EV[(EV.i == d.i) & (EV.j == d.j)]
                if len(mm):
                    dt = (mm.tca - d.tca).dt.total_seconds().abs() / 60
                    if (dt <= TCA_TOL_MIN).any(): kbest = dt.idxmin(); D.loc[di, "true_miss"] = mm.loc[kbest, "miss_km"]; D.loc[di, "true_pc"] = mm.loc[kbest, "pc_true"]
            matched = D.true_miss.notna(); rec_ref = EV.copy(); rec_ref["hit"] = False
            for ei, e in EV.iterrows():
                mm = D[(D.i == e.i) & (D.j == e.j)]
                if len(mm) and ((mm.tca - e.tca).dt.total_seconds().abs() / 60 <= TCA_TOL_MIN).any(): rec_ref.loc[ei, "hit"] = True
            close = rec_ref[rec_ref.miss_km < 10]; danger = rec_ref[rec_ref.pc_true >= 1e-6]
            r = dict(lead_h=L, chain=label, n_pred=len(D), n_matched=int(matched.sum()), precision=float(matched.mean()),
                     recall_ref_lt50km=float(rec_ref.hit.mean()), recall_ref_lt10km=float(close.hit.mean()) if len(close) else np.nan, recall_truePc_ge_1e6=float(danger.hit.mean()) if len(danger) else np.nan,
                     miss_err_med_km=float((D.miss_km - D.true_miss).abs().median()) if matched.any() else np.nan)
            if matched.sum() > 5:
                mm = D[matched]
                r["spearman_pcorb_vs_truepc"] = float(spearmanr(mm.pc_orb, mm.true_pc).statistic) if mm.pc_orb.std() > 0 else np.nan
                r["spearman_pointmiss_vs_truepc"] = float(spearmanr(-mm.miss_km, mm.true_pc).statistic)
                r["spearman_pcfixed_vs_truepc"] = float(spearmanr(mm.pc_fixed, mm.true_pc).statistic) if mm.pc_fixed.std() > 0 else np.nan
                y = (mm.true_pc >= 1e-6).values; K_ = max(int(y.sum()), 1)
                for nm, sc in (("pc_orb", mm.pc_orb.values), ("pc_fixed", mm.pc_fixed.values), ("point", -mm.miss_km.values)):
                    o = np.argsort(-sc); r[f"recall_at_k_{nm}"] = float(y[o[:K_]].mean()) if y.sum() else np.nan
            rows.append(r); D["matched"] = matched; ev_pred.append(D)
    R = pd.DataFrame(rows)
    if ev_pred: pd.concat(ev_pred, ignore_index=True).to_csv(C.RESULTS / "s2_starlink_predicted.csv", index=False)
    # S3 avoidance on matched A1 events: Pc-gate with OrbBench sigma vs distance rule; truth Pc from MEME covariance
    from .t7_avoidance import simulate, pol_none, pol_max, AMAX, SAFE_M, HBR_M
    from . import t6_warning as W
    s3 = []
    if ev_pred:
        A = pd.concat(ev_pred, ignore_index=True); A = A[(A.chain == "A1 PAF+MTF+T2") & (A.lead_h == 24) & A.matched]
        for name in ("No-maneuver", "Distance rule (2 km)", "Pc-gate OrbBench sigma", "Pc-gate fixed sigma", "Pc-gate operator covariance (oracle)"):
            res = []
            for e in A.itertuples():
                rel_r = np.array([e.rx, e.ry, e.rz]) * 1000; rel_v = np.array([e.vx, e.vy, e.vz]) * 1000; s_true = float(EV[(EV.i == e.i) & (EV.j == e.j)].sig_true_km.iloc[0]) * 1000
                sd = {"Pc-gate OrbBench sigma": e.sig_orb_km * 1000, "Pc-gate fixed sigma": float(np.sqrt(2) * np.mean([calib["A0 raw prev|" + c]["fixed"] for c in ("r", "t", "n")])) * 1000, "Pc-gate operator covariance (oracle)": s_true}.get(name)
                if name == "No-maneuver": pol = pol_none
                elif name.startswith("Distance"): pol = pol_max
                else:
                    def pol(d, k, dir0, s, vn=None, sd=sd): return dir0 * AMAX if W.pc_2d(d / 1000, HBR_KM, max(sd, 1.0) / 1000) >= 1e-4 else np.zeros(3)
                out = simulate(rel_r, rel_v, pol, max(s_true, 1.0))
                res.append(dict(dv=out["dv_m_s"], safe_pc=W.pc_2d(out["min_dist_m"] / 1000, HBR_KM, max(s_true, 1.0) / 1000) < 1e-4, safe_2km=out["safe"]))
            if res:
                Rr = pd.DataFrame(res); s3.append(dict(policy=name, n_events=len(Rr), safety_rate_true_pc=float(Rr.safe_pc.mean()), safety_rate_2km=float(Rr.safe_2km.mean()), dv_mean_m_s=float(Rr.dv.mean()), dv_p90_m_s=float(Rr.dv.quantile(.9))))
    return R, pd.DataFrame(s3), EV

# ------------------------------------------------------------------ S4 tables
def s4_tables(fc, wins, uq, r2, s3):
    te = fc[fc.split == "all"]
    def p50(m, h): v = te[(te.method == m) & (te.h == h)]["3d_p50_km"]; return float(v.iloc[0]) if len(v) else np.nan
    def p90(m, h): v = te[(te.method == m) & (te.h == h)]["3d_p90_km"]; return float(v.iloc[0]) if len(v) else np.nan
    def cov(prod, model, split="all", h="all"):
        v = uq[(uq.product == prod) & (uq.model == model) & (uq.component == "t") & (uq.split == split) & (uq.h.astype(str) == str(h))]
        return (float(v.cov95.iloc[0]), float(v.nees.iloc[0])) if len(v) else (np.nan, np.nan)
    abl = []
    for h in (4.0, 24.0):
        abl.append(dict(h=h, config="B5 TLE-only (raw prev, fixed sigma)", forecast_p50_km=p50("A0 raw prev", h), forecast_p90_km=p90("A0 raw prev", h), cov95_T=cov("A0 raw prev", "sigma-fixed")[0], nees_T=cov("A0 raw prev", "sigma-fixed")[1]))
        abl.append(dict(h=h, config="B1 no maneuver info (PAF+MTF, sigma-age)", forecast_p50_km=p50("+PAF +MTF", h), forecast_p90_km=p90("+PAF +MTF", h), cov95_T=cov("+PAF +MTF", "sigma-age")[0], nees_T=cov("+PAF +MTF", "sigma-age")[1]))
        abl.append(dict(h=h, config="B4 no precise reference (PAF+MTF+T2, uncalibrated fixed sigma)", forecast_p50_km=p50("+PAF +MTF +T2", h), forecast_p90_km=p90("+PAF +MTF +T2", h), cov95_T=cov("+PAF +MTF +T2", "sigma-fixed")[0], nees_T=cov("+PAF +MTF +T2", "sigma-fixed")[1]))
        abl.append(dict(h=h, config="B0 full (PAF+MTF+T2, sigma-age x g(T2))", forecast_p50_km=p50("+PAF +MTF +T2", h), forecast_p90_km=p90("+PAF +MTF +T2", h), cov95_T=cov("+PAF +MTF +T2", "sigma-age x g(T2)")[0], nees_T=cov("+PAF +MTF +T2", "sigma-age x g(T2)")[1]))
    a01 = []
    for h in (1.0, 4.0, 24.0):
        a01.append(dict(stage="T4 forecast vs MEME", metric=f"3D P50 km, +{h:g} h", A0="raw prev TLE", A0_value=round(p50("A0 raw prev", h), 3), A1="PAF+MTF+T2", A1_value=round(p50("+PAF +MTF +T2", h), 3)))
        a01.append(dict(stage="T4 forecast vs MEME", metric=f"3D P90 km, +{h:g} h", A0="raw prev TLE", A0_value=round(p90("A0 raw prev", h), 3), A1="PAF+MTF+T2", A1_value=round(p90("+PAF +MTF +T2", h), 3)))
    pt = fc[(fc.split == "post-thrust(48h)") & (fc.h == 24.0)]
    for m in ("A0 raw prev", "+PAF +MTF +T2"):
        v = pt[pt.method == m]
        if len(v): a01.append(dict(stage="T4 forecast, 48 h after thrust", metric="3D P50 km, +24 h", A0="raw prev TLE" if m.startswith("A0") else "", A0_value=round(float(v["3d_p50_km"].iloc[0]), 3) if m.startswith("A0") else "", A1="" if m.startswith("A0") else "PAF+MTF+T2", A1_value="" if m.startswith("A0") else round(float(v["3d_p50_km"].iloc[0]), 3)))
    c0 = cov("A0 raw prev", "sigma-fixed"); c1 = cov("+PAF +MTF +T2", "sigma-age x g(T2)")
    a01.append(dict(stage="UQ (along-track, all test)", metric="95% coverage / NEES", A0="fixed sigma", A0_value=f"{c0[0]:.3f} / {c0[1]:.2f}", A1="sigma-age x g(T2)", A1_value=f"{c1[0]:.3f} / {c1[1]:.2f}"))
    c0 = cov("A0 raw prev", "sigma-fixed", "post-thrust(48h)"); c1 = cov("+PAF +MTF +T2", "sigma-age x g(T2)", "post-thrust(48h)")
    a01.append(dict(stage="UQ (along-track, post-thrust)", metric="95% coverage / NEES", A0="fixed sigma", A0_value=f"{c0[0]:.3f} / {c0[1]:.2f}", A1="sigma-age x g(T2)", A1_value=f"{c1[0]:.3f} / {c1[1]:.2f}"))
    for L in (6, 24):
        a = r2[(r2.lead_h == L) & (r2.chain == "A0 raw prev")]; b = r2[(r2.lead_h == L) & (r2.chain == "A1 PAF+MTF+T2")]
        if len(a) and len(b) and "recall_ref_lt50km" in a:
            a01.append(dict(stage=f"T6 warning, lead {L} h", metric="recall of MEME approaches < 50 km / precision", A0="raw prev, point", A0_value=f"{a.recall_ref_lt50km.iloc[0]:.2f} / {a.precision.iloc[0]:.2f}", A1="PAF+MTF+T2", A1_value=f"{b.recall_ref_lt50km.iloc[0]:.2f} / {b.precision.iloc[0]:.2f}"))
            if "spearman_pcorb_vs_truepc" in b and np.isfinite(b.spearman_pcorb_vs_truepc.iloc[0]):
                a01.append(dict(stage=f"T6 ranking, lead {L} h", metric="Spearman with true Pc (MEME covariance)", A0="point miss", A0_value=round(float(b.spearman_pointmiss_vs_truepc.iloc[0]), 3), A1="Pc(OrbBench sigma)", A1_value=round(float(b.spearman_pcorb_vs_truepc.iloc[0]), 3)))
    if len(s3):
        d = s3.set_index("policy")
        if "Distance rule (2 km)" in d.index and "Pc-gate OrbBench sigma" in d.index:
            a01.append(dict(stage="T7 avoidance (matched events)", metric="mean delta-v m/s [safety by true Pc]", A0="distance rule", A0_value=f"{d.loc['Distance rule (2 km)','dv_mean_m_s']:.2f} [{d.loc['Distance rule (2 km)','safety_rate_true_pc']:.2f}]", A1="Pc-gate OrbBench sigma", A1_value=f"{d.loc['Pc-gate OrbBench sigma','dv_mean_m_s']:.2f} [{d.loc['Pc-gate OrbBench sigma','safety_rate_true_pc']:.2f}]"))
    return pd.DataFrame(abl), pd.DataFrame(a01)

def figure(fc):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    BLUE, RED, YELLOW, GREEN, PURPLE, LBLUE = "#4477AA", "#EE6677", "#FFBE7A", "#1F7940", "#BEB8DC", "#66AFC3"
    plt.rcParams.update({"font.size": 7, "axes.labelsize": 7, "xtick.labelsize": 6.5, "ytick.labelsize": 6.5, "legend.fontsize": 6, "axes.linewidth": 0.6, "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, 2, figsize=(3.3, 1.9), constrained_layout=True, sharey=True)
    for ax, split in zip(axes, ("all", "post-thrust(48h)")):
        a = fc[fc.split == split]
        for m, col, ls in (("A0 raw prev", RED, "--"), ("+PAF prev", YELLOW, "--"), ("+PAF +MTF", LBLUE, "-"), ("+PAF +MTF +T2", BLUE, "-"), ("+PAF +MTF +T2 +T1", GREEN, ":")):
            g = a[a.method == m].sort_values("h")
            if len(g): ax.plot(g.h, g["3d_p50_km"], marker="o", ms=2.5, lw=0.9, ls=ls, color=col, label=m)
        ax.set_xscale("log"); ax.set_xticks(HZ); ax.set_xticklabels([f"{h:g}" for h in HZ]); ax.set_xlabel("Horizon (h)"); ax.set_title(split, fontsize=7); ax.grid(color="#DDDDDD", lw=0.4, alpha=0.6)
    axes[0].set_ylabel("3D error P50 vs MEME (km)"); axes[0].legend(frameon=False)
    fig.savefig(C.FIGURES / "fig_s1_starlink_forecast.pdf"); fig.savefig(C.FIGURES / "fig_s1_starlink_forecast.png", dpi=300); plt.close(fig)

def main(parts="1234", force=False, smoke=False):
    if stage_done("S8f_starlink_chain") and not force: log.info("done"); return
    set_status("S8f_starlink_chain", "running")
    try:
        with Timer() as T:
            ids = [n for n in starlink_ids() if (C.DATA / f"starlink/meme_{n}.parquet").exists()]
            if smoke: ids = ids[:3]
            # pass 1: detection scores to tune k_t2 on train (cheap: run _s1_one with a provisional k and reuse det tables)
            parts_c, parts_d = [], []
            with ProcessPoolExecutor(min(C.N_WORKERS, len(ids))) as ex:
                for fu in as_completed([ex.submit(_s1_one, (n, 6.0)) for n in ids]):
                    n, cs, det = fu.result()
                    if isinstance(cs, pd.DataFrame): parts_c.append(cs); parts_d.append(det); log.info(f"S1 {n}: {len(cs)} rows")
                    else: log.warning(f"S1 {n} failed: {str(cs)[:300]}")
            DET = pd.concat(parts_d, ignore_index=True); k_t2 = tune_k_t2(DET); log.info(f"T2 threshold tuned on train vs MEME thrust arcs: k={k_t2}")
            d = DET[(DET.pool == "step6") & DET.is_test & DET.t2.notna()]; p = d.t2 > k_t2; tp = (p & d.truth).sum()
            det_tab = pd.DataFrame([dict(pool="step6", k=k_t2, n_intervals=len(d), n_thrust=int(d.truth.sum()), precision=tp / max(p.sum(), 1), recall=tp / max(d.truth.sum(), 1), thrust_fraction=float(d.truth.mean()))])
            save_df(det_tab, "s1_t2_detection.csv"); log.info("T2 vs MEME thrust arcs (test):\n" + det_tab.round(3).to_string(index=False))
            if abs(k_t2 - 6.0) > 0.5:
                parts_c = []
                with ProcessPoolExecutor(min(C.N_WORKERS, len(ids))) as ex:
                    for fu in as_completed([ex.submit(_s1_one, (n, k_t2)) for n in ids]):
                        n, cs, det = fu.result()
                        if isinstance(cs, pd.DataFrame): parts_c.append(cs)
            cases = pd.concat(parts_c, ignore_index=True); cases.to_parquet(C.RESULTS / "s1_starlink_cases.parquet", index=False)
            fc, wins, uq, calib = s1_summaries(cases); save_df(fc, "s1_starlink_forecast.csv"); save_df(wins, "s1_starlink_winrates.csv"); save_df(uq, "s1_starlink_uq.csv")
            log.info("S1 3D P50 (test, all):\n" + fc[fc.split == "all"].pivot_table(index="method", columns="h", values="3d_p50_km").round(3).to_string())
            log.info("S1 3D P50 post-thrust:\n" + fc[fc.split == "post-thrust(48h)"].pivot_table(index="method", columns="h", values="3d_p50_km").round(3).to_string())
            log.info("S1 win rates:\n" + wins.round(3).to_string(index=False))
            CR = pd.read_csv(C.RESULTS / "s1_starlink_correctors.csv"); log.info("S1 correctors:\n" + CR[CR.get("split").isna() if "split" in CR else slice(None)].round(3).to_string(index=False))
            log.info("retention step6 / paf-T1 (per sat): " + str(cases.groupby("norad")[["ret_step6", "ret_paf_t1"]].first().round(2).to_dict("index")))
            log.info("S1 UQ (T, all h):\n" + uq[(uq.component == "t") & (uq.h.astype(str) == "all")][["product", "model", "split", "n", "cov68", "cov95", "nees", "sigma_p50_km"]].round(3).to_string(index=False))
            figure(fc)
            r2, s3, EV = s2_s3(calib, k_t2); save_df(r2, "s2_starlink_ranking.csv"); save_df(s3, "s3_starlink_avoidance.csv")
            log.info("S2:\n" + r2.round(3).to_string(index=False)); log.info("S3:\n" + s3.round(3).to_string(index=False))
            abl, a01 = s4_tables(fc, wins, uq, r2, s3); save_df(abl, "s4_starlink_ablation.csv"); save_df(a01, "starlink_a0_a1.csv")
            log.info("S4 ablation:\n" + abl.round(3).to_string(index=False)); log.info("A0 vs A1 (Starlink):\n" + a01.to_string(index=False))
        set_status("S8f_starlink_chain", "done", elapsed_s=round(T.elapsed), n_cases=int(len(cases)))
    except Exception as ex:
        log.exception("starlink chain failed"); set_status("S8f_starlink_chain", "failed", message=str(ex)); raise

if __name__ == "__main__":
    main(force="--force" in sys.argv, smoke="--smoke" in sys.argv)
