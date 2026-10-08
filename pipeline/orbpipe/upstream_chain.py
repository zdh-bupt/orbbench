"""Upstream task connections that are absent from the basic graph but physically motivated, tested with truth:
  T1 -> T2, T3 -> T2  on the three Sentinels with maneuver logs (train 2019-23, test 2024-25H1):
       the SGP4-consistency detector's threshold is conditioned on the T1 phase at the epoch (stable vs non-stable, causal
       rolling-slope rule) and on the T3 geomagnetic index (Ap of the previous day >= AP_STORM raises the threshold).
       Variants: A0 residual-T; +T1; +T3; +T1+T3 (A1); supervised HistGB with/without the two inputs.
       Metrics: event F1/F2 per satellite and pooled; false alarms on storm days; recall in non-stable phases.
  T2 -> T1  on the cohort (silver rule labels, 3 object-level seeds): HistGB phase classifier with/without per-window T2
       features (number of flagged epochs, max and mean consistency score).
Outputs: results/uc_t2_sentinel.csv, uc_t2_storm_fp.csv, uc_t1_cohort.csv
"""
from __future__ import annotations
import sys, os
os.environ.setdefault("OMP_NUM_THREADS", "8")
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score

from . import config as C
from .util import get_logger, set_status, stage_done, Timer, save_df, to_utc, load_maneuvers, mad
from .tlefeat import tle_features
from .t2_maneuver import label_intervals, event_scores, best_threshold, FEATS as T2FEATS
from .t1_phase import FEATS as T1FEATS, CLASSES

log = get_logger("upstream_chain")
AP_STORM = 40.0; KGRID = np.arange(0.5, 30, 0.5); FGRID = np.arange(1.0, 3.01, 0.25)

def causal_phase(f: pd.DataFrame, n_win: int = 10) -> np.ndarray:
    """T1 phase at each epoch from the trailing n_win TLEs (same thresholds as the T1 slope rule): 1 = non-stable."""
    a = f.a_km.values; t = f.epoch.values.astype("datetime64[ns]").astype("int64") / 86400e9; out = np.zeros(len(f), bool)
    for i in range(n_win, len(f)):
        x = t[i - n_win:i + 1]; y = a[i - n_win:i + 1]; sl = np.polyfit(x - x[0], y, 1)[0]; d = np.diff(y)
        pp = np.mean(d > 0); pn = np.mean(d < 0)
        out[i] = (sl >= 4 / 7 and pp >= .55) or (sl <= -0.9 / 7 and pn >= .7)
    return out

def t2_sentinels():
    sw = pd.read_csv(C.DATA / "space_weather_daily.csv", parse_dates=["date"]).set_index("date").ap_avg
    t0 = to_utc(C.SENTINEL_TEST_START); t1 = to_utc(C.WINDOW_END); data = {}
    for name, (norad, _, man) in C.SENTINELS.items():
        if not man: continue
        f = tle_features(norad, C.SENTINEL_START, C.WINDOW_END).dropna(subset=T2FEATS).reset_index(drop=True)
        f["y"] = label_intervals(f, load_maneuvers(name)); f["is_test"] = (f.epoch >= t0) & (f.epoch < t1); f["nonstable"] = causal_phase(f)
        d1 = f.epoch.dt.tz_convert(None).dt.normalize() - pd.Timedelta(days=1); f["ap_lag1"] = sw.reindex(d1).to_numpy(); f["storm"] = f.ap_lag1 >= AP_STORM
        data[name] = f; log.info(f"{name}: {len(f)} intervals, {int(f.y.sum())} maneuvers, non-stable {f.nonstable.mean()*100:.1f}%, storm intervals {int(f.storm.sum())}")
    rows = []; fp_rows = []
    for name, f in data.items():
        tr, te = f[~f.is_test], f[f.is_test]; scale = max(mad(tr.res_t.values), 1e-9); s_tr = np.abs(tr.res_t.values) / scale; s_te = np.abs(te.res_t.values) / scale
        preds = {}
        k0, _ = best_threshold(s_tr, tr.y.values, KGRID); preds["SGP4-residual (A0)"] = s_te > k0
        # + T1: separate thresholds for stable / non-stable epochs (tuned on train)
        ns_tr, ns_te = tr.nonstable.values, te.nonstable.values
        k_s, _ = best_threshold(s_tr[~ns_tr], tr.y.values[~ns_tr], KGRID) if (~ns_tr).sum() > 50 else (k0, 0)
        k_n, _ = best_threshold(s_tr[ns_tr], tr.y.values[ns_tr], KGRID) if ns_tr.sum() > 50 and tr.y.values[ns_tr].sum() > 3 else (k0, 0)
        thr_t1 = np.where(ns_te, k_n, k_s); preds["SGP4-residual + T1 phase"] = s_te > thr_t1
        # + T3: storm factor on the threshold (tuned on train)
        st_tr, st_te = tr.storm.values, te.storm.values
        def f2_of(fac, thr_base_tr): return event_scores(tr.y.values, s_tr > thr_base_tr * np.where(st_tr, fac, 1.0))["f2"]
        fac = max(FGRID, key=lambda g: f2_of(g, k0)); preds["SGP4-residual + T3 Ap"] = s_te > k0 * np.where(st_te, fac, 1.0)
        fac2 = max(FGRID, key=lambda g: f2_of(g, np.where(ns_tr, k_n, k_s))); preds["SGP4-residual + T1 phase + T3 Ap (A1)"] = s_te > thr_t1 * np.where(st_te, fac2, 1.0)
        # supervised with / without the upstream inputs
        for nm, cols in (("HistGB (A0)", T2FEATS), ("HistGB + T1 + T3 (A1)", T2FEATS + ["nonstable", "ap_lag1"])):
            X_tr = tr[cols].fillna(0).astype(float); X_te = te[cols].fillna(0).astype(float)
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=C.SEED).fit(X_tr, tr.y)
            thr, _ = best_threshold(clf.predict_proba(X_tr)[:, 1], tr.y.values, np.linspace(0.05, 0.95, 19)); preds[nm] = clf.predict_proba(X_te)[:, 1] > thr
        for nm, pr in preds.items():
            sc = event_scores(te.y.values, pr); rows.append(dict(satellite=name, variant=nm, k_stable=k_s, k_nonstable=k_n, storm_factor=fac2 if "A1" in nm else (fac if "T3" in nm else 1.0), **sc))
            fp_rows.append(dict(satellite=name, variant=nm, storm_intervals=int(st_te.sum()), fp_storm=int((pr & ~te.y.values & st_te).sum()), fp_quiet=int((pr & ~te.y.values & ~st_te).sum()),
                                fp_rate_storm=float((pr & ~te.y.values)[st_te].mean()) if st_te.any() else np.nan, fp_rate_quiet=float((pr & ~te.y.values)[~st_te].mean()),
                                recall_nonstable=float((pr & te.y.values)[ns_te].sum() / max(te.y.values[ns_te].sum(), 1)), recall_stable=float((pr & te.y.values)[~ns_te].sum() / max(te.y.values[~ns_te].sum(), 1))))
    met = pd.DataFrame(rows); pooled = met.groupby("variant")[["tp", "fp", "fn"]].sum()
    pooled["precision"] = pooled.tp / (pooled.tp + pooled.fp).clip(lower=1); pooled["recall"] = pooled.tp / (pooled.tp + pooled.fn).clip(lower=1)
    pooled["f1"] = 2 * pooled.precision * pooled.recall / (pooled.precision + pooled.recall).clip(lower=1e-9); pooled["f2"] = 5 * pooled.precision * pooled.recall / (4 * pooled.precision + pooled.recall).clip(lower=1e-9)
    pooled = pooled.reset_index(); pooled.insert(0, "satellite", "ALL"); met = pd.concat([met, pooled], ignore_index=True)
    fp = pd.DataFrame(fp_rows); fpp = fp.groupby("variant")[["storm_intervals", "fp_storm", "fp_quiet"]].sum().reset_index(); fpp.insert(0, "satellite", "ALL"); fp = pd.concat([fp, fpp], ignore_index=True)
    return met, fp

def t1_cohort():
    lab = pd.read_parquet(C.RESULTS / "t1_phase_labels.parquet"); lab["t_start"] = pd.to_datetime(lab.t_start, utc=True); lab["t_end"] = pd.to_datetime(lab.t_end, utc=True)
    fl = pd.read_parquet(C.RESULTS / "t2_cohort_flags.parquet"); fl["epoch"] = pd.to_datetime(fl.epoch, utc=True)
    main = lab[lab.silver.isin(CLASSES)].copy().reset_index(drop=True)
    # per-window T2 features
    n_fl = np.zeros(len(main)); mx = np.zeros(len(main)); mn = np.zeros(len(main))
    for norad, g in main.groupby("norad"):
        f = fl[fl.norad == norad]
        if len(f) == 0: continue
        ep = f.epoch.values; sc = np.nan_to_num(f.score_res.values, nan=0.0); flg = f.flag.values
        for i, r in g.iterrows():
            m = (ep >= r.t_start.to_datetime64()) & (ep <= r.t_end.to_datetime64())
            if m.any(): n_fl[i] = flg[m].sum(); mx[i] = sc[m].max(); mn[i] = sc[m].mean()
    main["t2_n_flags"] = n_fl; main["t2_max_score"] = mx; main["t2_mean_score"] = mn
    rows = []; obj = main.norad.unique()
    for seed in (42, 43, 44):
        rng = np.random.default_rng(seed); perm = rng.permutation(obj); n = len(perm); tr_o, te_o = perm[: int(.7 * n)], perm[int(.85 * n):]
        tr, te = main[main.norad.isin(tr_o)], main[main.norad.isin(te_o)]
        for nm, cols in (("HistGB (A0)", T1FEATS), ("HistGB + T2 flags (A1)", T1FEATS + ["t2_n_flags", "t2_max_score", "t2_mean_score"])):
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=seed).fit(tr[cols], tr.silver); p = clf.predict(te[cols])
            r = dict(seed=seed, variant=nm, n_test=len(te), macro_f1=f1_score(te.silver, p, labels=CLASSES, average="macro", zero_division=0))
            for cl in CLASSES: r[f"f1_{cl}"] = f1_score(te.silver == cl, p == cl, zero_division=0)
            rows.append(r)
    return pd.DataFrame(rows)

def t2_starlink():
    """T1 -> T2 on Starlink: detector threshold conditioned on the T1 phase (raise / lower / stable from the SMA trend of the
    previous 5 TLEs); truth = MEME-derived thrust arcs per TLE interval; train < TRAIN_END, test after."""
    from .starlink_chain import load_meme, thrust_arcs, read_txt, t2_scores, t1_phase, W0, W1, TRAIN_END
    from .stage_starlink import RAW_TLE_DIR, starlink_ids
    rows = []
    for norad in starlink_ids():
        mp = C.DATA / f"starlink/meme_{norad}.parquet"; tp = RAW_TLE_DIR / f"{norad}.txt"
        if not (mp.exists() and tp.exists()): continue
        arc = thrust_arcs(load_meme(norad)); d = read_txt(tp, W0 - pd.Timedelta(days=3), W1).reset_index(drop=True)
        if len(d) < 10: continue
        d["t2"] = t2_scores(d); d["phase"] = t1_phase(d)
        for i in range(1, len(d)):
            lo, hi = d.epoch.iloc[i - 1], d.epoch.iloc[i]; seg = arc[(arc.index > lo) & (arc.index <= hi)]
            if len(seg) == 0 or not np.isfinite(d.t2.iloc[i]): continue
            rows.append(dict(norad=norad, epoch=hi, t2=float(d.t2.iloc[i]), phase=int(d.phase.iloc[i]), truth=bool(seg.mean() > 0.5), is_test=bool(hi >= TRAIN_END)))
    D = pd.DataFrame(rows); tr, te = D[~D.is_test], D[D.is_test]
    def prf(p, y):
        p = np.asarray(p, bool); y = np.asarray(y, bool); tp = int((p & y).sum()); fp = int((p & ~y).sum()); fn = int((~p & y).sum()); tn = int((~p & ~y).sum())
        P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1); spec = tn / max(tn + fp, 1); den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
        return dict(tp=tp, fp=fp, fn=fn, tn=tn, precision=P, recall=R, f1=2 * P * R / max(P + R, 1e-9), balanced_acc=0.5 * (R + spec), mcc=(tp * tn - fp * fn) / den if den > 0 else 0.0,
                    coast_precision=tn / max(tn + fn, 1), coast_recall=spec)
    def best_k(sc, y):
        best = (KGRID[0], -1.0)
        for k in KGRID:
            f = prf(sc > k, y)["mcc"]
            if f > best[1]: best = (float(k), f)
        return best[0]
    k0 = best_k(tr.t2.values, tr.truth.values); out = []
    out.append(dict(variant="always thrust (trivial)", k_stable=np.nan, k_raise=np.nan, k_lower=np.nan, n_test=len(te), n_thrust=int(te.truth.sum()), **prf(np.ones(len(te), bool), te.truth.values)))
    out.append(dict(variant="SGP4-residual (A0)", k_stable=k0, k_raise=k0, k_lower=k0, n_test=len(te), n_thrust=int(te.truth.sum()), **prf(te.t2.values > k0, te.truth.values)))
    ks = {}
    for ph in (0, 1, -1):
        m = tr.phase.values == ph; ks[ph] = best_k(tr.t2.values[m], tr.truth.values[m]) if m.sum() >= 30 and tr.truth.values[m].sum() >= 5 and (~tr.truth.values[m]).sum() >= 5 else k0
    thr = np.array([ks[int(ph)] for ph in te.phase.values]); out.append(dict(variant="SGP4-residual, threshold per T1 phase", k_stable=ks[0], k_raise=ks[1], k_lower=ks[-1], n_test=len(te), n_thrust=int(te.truth.sum()), **prf(te.t2.values > thr, te.truth.values)))
    ph_te = te.phase.values != 0; out.append(dict(variant="T1 phase alone (non-stable = thrust)", k_stable=np.nan, k_raise=np.nan, k_lower=np.nan, n_test=len(te), n_thrust=int(te.truth.sum()), **prf(ph_te, te.truth.values)))
    # A1: residual OR phase, with the residual threshold re-tuned on train given the phase
    def f1_or(k): return prf((tr.t2.values > k) | (tr.phase.values != 0), tr.truth.values)["mcc"]
    def mcc_and(k): return prf((tr.t2.values > k) & (tr.phase.values != 0), tr.truth.values)["mcc"]
    k_and = float(max(KGRID, key=mcc_and)); out.append(dict(variant="SGP4-residual AND T1 phase", k_stable=k_and, k_raise=k_and, k_lower=k_and, n_test=len(te), n_thrust=int(te.truth.sum()), **prf((te.t2.values > k_and) & ph_te, te.truth.values)))
    k_or = float(max(KGRID, key=f1_or)); out.append(dict(variant="SGP4-residual OR T1 phase (A1)", k_stable=k_or, k_raise=k_or, k_lower=k_or, n_test=len(te), n_thrust=int(te.truth.sum()), **prf((te.t2.values > k_or) | ph_te, te.truth.values)))
    for ph, nm in ((0, "stable"), (1, "raise"), (-1, "lower")):
        m = te.phase.values == ph
        if m.sum() < 10: continue
        out.append(dict(variant=f"  phase={nm}: A0", k_stable=k0, k_raise=k0, k_lower=k0, n_test=int(m.sum()), n_thrust=int(te.truth.values[m].sum()), **prf(te.t2.values[m] > k0, te.truth.values[m])))
        out.append(dict(variant=f"  phase={nm}: A1", k_stable=ks[0], k_raise=ks[1], k_lower=ks[-1], n_test=int(m.sum()), n_thrust=int(te.truth.values[m].sum()), **prf(te.t2.values[m] > ks[ph], te.truth.values[m])))
    return pd.DataFrame(out)

def main(force=False):
    if stage_done("S8k_upstream_chain") and not force: log.info("done"); return
    set_status("S8k_upstream_chain", "running")
    try:
        with Timer() as T:
            met, fp = t2_sentinels(); save_df(met, "uc_t2_sentinel.csv"); save_df(fp, "uc_t2_storm_fp.csv")
            log.info("T2 pooled:\n" + met[met.satellite == "ALL"][["variant", "precision", "recall", "f1", "f2"]].round(3).to_string(index=False)); log.info("storm FP:\n" + fp[fp.satellite == "ALL"].to_string(index=False))
            st = t2_starlink(); save_df(st, "uc_t2_starlink.csv"); log.info("T2 Starlink (T1 -> T2):\n" + st.round(3).to_string(index=False))
            t1 = t1_cohort(); save_df(t1, "uc_t1_cohort.csv"); log.info("T1:\n" + t1.groupby("variant")[["macro_f1"] + [f"f1_{c}" for c in CLASSES]].mean().round(3).to_string())
        set_status("S8k_upstream_chain", "done", elapsed_s=round(T.elapsed))
    except Exception as ex:
        log.exception("upstream_chain failed"); set_status("S8k_upstream_chain", "failed", message=str(ex)); raise

if __name__ == "__main__":
    main(force="--force" in sys.argv)
