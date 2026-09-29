"""T2 Maneuver detection from TLE sequences.

Truth: Sentinel maneuver logs (S1A, S2A, S3A). Unit of decision = TLE interval (epoch_{i-1}, epoch_i].
Positive if a maneuver start falls in the interval. Event-level scoring with ±1-interval tolerance.

Detectors
  Δa-MAD        |Δa| > k · MAD(Δa)_train                       (element jump, Kelecy-style)
  SGP4-residual |res_t| > k · MAD                              (TLE consistency check, Lemmens-style)
  CUSUM(Δa)     one-sided CUSUM on standardised Δa
  HistGB        supervised on [dt, Δa, Δn/n, Δe, Δi, ΔB*, res_r, res_t, res_n, res_3d] (train: pre-window; test: window)
  HistGB-LOSO   leave-one-satellite-out generalisation (train on two sats, test on the third)
Thresholds k chosen on the training period to maximise F2 (recall-weighted, STM convention).
Outputs: results/t2_maneuver_metrics.csv, results/t2_maneuver_events_<S>.csv, results/t2_cohort_flags.parquet
(flags for the 3.6k cohort with the best unsupervised detector, used by T4/T6 in chained mode).
"""
from __future__ import annotations
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from . import config as C
from .util import get_logger, set_status, stage_done, Timer, save_df, to_utc, load_maneuvers, load_cohort, mad
from .tlefeat import tle_features

log = get_logger("t2_maneuver")
FEATS = ["dt_h", "da_km", "dn_rel", "de", "di_deg", "dbstar", "res_r", "res_t", "res_n", "res_3d"]
TRAIN_START = C.SENTINEL_START

def label_intervals(f: pd.DataFrame, man: pd.DataFrame) -> np.ndarray:
    ep = f.epoch.values.astype("datetime64[ns]")
    y = np.zeros(len(f), bool)
    if len(man) == 0: return y
    idx = np.searchsorted(ep, man.start.values.astype("datetime64[ns]"))
    idx = idx[(idx > 0) & (idx < len(f))]
    y[idx] = True
    return y

def event_scores(y: np.ndarray, pred: np.ndarray, tol: int = 1) -> dict:
    """Event-level P/R/F1/F2 with ±tol interval tolerance; delay in intervals."""
    ti = np.where(y)[0]; pi = np.where(pred)[0]
    tp = 0; matched = set(); delays = []
    for t in ti:
        c = [p for p in pi if abs(p - t) <= tol and p not in matched]
        if c:
            p = min(c, key=lambda p: abs(p - t)); matched.add(p); tp += 1; delays.append(p - t)
    fn = len(ti) - tp; fp = len(pi) - len(matched)
    P = tp / max(tp + fp, 1); R = tp / max(tp + fn, 1)
    F1 = 2 * P * R / max(P + R, 1e-9); F2 = 5 * P * R / max(4 * P + R, 1e-9)
    return dict(tp=tp, fp=fp, fn=fn, precision=P, recall=R, f1=F1, f2=F2, n_true=len(ti), n_pred=len(pi),
                mean_delay_intervals=float(np.mean(delays)) if delays else np.nan)

def cusum(z: np.ndarray, k: float = 0.5) -> np.ndarray:
    s = np.zeros_like(z); acc = 0.0
    for i, v in enumerate(z):
        acc = max(0.0, acc + abs(v) - k); s[i] = acc
    return s

def best_threshold(score: np.ndarray, y: np.ndarray, grid) -> tuple[float, float]:
    best = (grid[0], -1)
    for k in grid:
        f2 = event_scores(y, score > k)["f2"]
        if f2 > best[1]: best = (k, f2)
    return best

def run_sentinels() -> tuple[pd.DataFrame, dict]:
    t0, t1 = to_utc(C.SENTINEL_TEST_START), to_utc(C.WINDOW_END)
    data = {}
    for name, (norad, _, man) in C.SENTINELS.items():
        if not man: continue
        f = tle_features(norad, TRAIN_START, C.WINDOW_END).dropna(subset=FEATS)
        m = load_maneuvers(name)
        f["y"] = label_intervals(f, m); f["sat"] = name; f["is_test"] = (f.epoch >= t0) & (f.epoch < t1)
        data[name] = f
        log.info(f"{name}: {len(f)} intervals, {f.y.sum()} maneuver intervals ({f[f.is_test].y.sum()} in test)")
    rows = []; events = []
    for name, f in data.items():
        tr, te = f[~f.is_test], f[f.is_test]
        preds = {}
        # unsupervised detectors with train-tuned k
        for det, col in (("Δa-MAD", "da_km"), ("SGP4-residual-T", "res_t"), ("SGP4-residual-3D", "res_3d")):
            s_tr = np.abs(tr[col].values) / max(mad(tr[col].values), 1e-9); s_te = np.abs(te[col].values) / max(mad(tr[col].values), 1e-9)
            k, _ = best_threshold(s_tr, tr.y.values, np.arange(2, 30, 0.5)); preds[det] = s_te > k
        z_tr = (tr.da_km.values - np.median(tr.da_km)) / max(mad(tr.da_km.values) * 1.4826, 1e-9)
        z_te = (te.da_km.values - np.median(tr.da_km)) / max(mad(tr.da_km.values) * 1.4826, 1e-9)
        k, _ = best_threshold(cusum(z_tr), tr.y.values, np.arange(1, 30, 0.5)); preds["CUSUM(Δa)"] = cusum(z_te) > k
        # supervised
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=C.SEED).fit(tr[FEATS], tr.y)
        p = clf.predict_proba(te[FEATS])[:, 1]; p_tr = clf.predict_proba(tr[FEATS])[:, 1]
        thr, _ = best_threshold(p_tr, tr.y.values, np.linspace(0.05, 0.95, 19)); preds["HistGB"] = p > thr
        # leave-one-satellite-out
        others = pd.concat([d[~d.is_test] for n, d in data.items() if n != name])
        if len(others) and others.y.sum() > 5:
            clf2 = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=C.SEED).fit(others[FEATS], others.y)
            p2 = clf2.predict_proba(te[FEATS])[:, 1]; p2_tr = clf2.predict_proba(others[FEATS])[:, 1]
            thr2, _ = best_threshold(p2_tr, others.y.values, np.linspace(0.05, 0.95, 19)); preds["HistGB-LOSO"] = p2 > thr2
        for det, pr in preds.items():
            sc = event_scores(te.y.values, pr); rows.append(dict(task="T2", satellite=name, method=det, **sc))
            events.append(pd.DataFrame({"sat": name, "method": det, "epoch": te.epoch.values, "y": te.y.values, "pred": pr}))
        # persistence-of-nothing baseline
        rows.append(dict(task="T2", satellite=name, method="Never-alarm", **event_scores(te.y.values, np.zeros(len(te), bool))))
    met = pd.DataFrame(rows); ev = pd.concat(events, ignore_index=True)
    for name in data: ev[ev.sat == name].to_csv(C.RESULTS / f"t2_maneuver_events_{name}.csv", index=False)
    # pooled
    pooled = met[met.method != "Never-alarm"].groupby("method")[["tp", "fp", "fn"]].sum()
    pooled["precision"] = pooled.tp / (pooled.tp + pooled.fp).clip(lower=1); pooled["recall"] = pooled.tp / (pooled.tp + pooled.fn).clip(lower=1)
    pooled["f1"] = 2 * pooled.precision * pooled.recall / (pooled.precision + pooled.recall).clip(lower=1e-9)
    pooled["f2"] = 5 * pooled.precision * pooled.recall / (4 * pooled.precision + pooled.recall).clip(lower=1e-9)
    pooled = pooled.reset_index(); pooled.insert(0, "satellite", "ALL"); pooled.insert(0, "task", "T2")
    met = pd.concat([met, pooled], ignore_index=True)
    # record tuned thresholds from the last satellite-independent run (for cohort inference we re-tune pooled)
    return met, data

def _cohort_one(args):
    norad, k_res, k_da, mad_ref = args
    try:
        f = tle_features(norad, pd.Timestamp(C.WINDOW_START) - pd.Timedelta(days=60), C.WINDOW_END)
        if f.empty: return norad, None
        s_res = np.abs(f.res_t.values) / max(mad(f.res_t.values), 1e-9)
        s_da = np.abs(f.da_km.values) / max(mad(f.da_km.values), 1e-9)
        flag = (s_res > k_res) | (s_da > k_da)
        return norad, pd.DataFrame({"norad": norad, "epoch": f.epoch, "flag_res": s_res > k_res, "flag_da": s_da > k_da, "flag": flag,
                                    "score_res": s_res, "score_da": s_da})
    except Exception as ex:
        return norad, str(ex)

def run_cohort(data: dict) -> pd.DataFrame:
    """Apply the pooled-tuned unsupervised detector (residual OR Δa, per-object MAD scaling) to the cohort."""
    tr = pd.concat([d[~d.is_test] for d in data.values()])
    s_res = np.abs(tr.res_t.values) / max(mad(tr.res_t.values), 1e-9); s_da = np.abs(tr.da_km.values) / max(mad(tr.da_km.values), 1e-9)
    k_res, _ = best_threshold(s_res, tr.y.values, np.arange(2, 30, 0.5)); k_da, _ = best_threshold(s_da, tr.y.values, np.arange(2, 30, 0.5))
    log.info(f"cohort detector thresholds: k_res={k_res}, k_da={k_da} (MAD units, per-object scaling)")
    obj = load_cohort()
    out = []; errs = 0
    with ProcessPoolExecutor(C.N_WORKERS) as ex:
        futs = [ex.submit(_cohort_one, (int(n), k_res, k_da, None)) for n in obj.norad]
        for i, fu in enumerate(as_completed(futs)):
            n, r = fu.result()
            if isinstance(r, pd.DataFrame): out.append(r)
            elif r is not None: errs += 1
            if (i + 1) % 500 == 0: log.info(f"cohort flags {i+1}/{len(futs)}")
    fl = pd.concat(out, ignore_index=True); fl.to_parquet(C.RESULTS / "t2_cohort_flags.parquet", index=False)
    summ = fl.groupby("norad").flag.mean().describe()
    log.info(f"cohort flags: {len(fl)} intervals, {fl.flag.mean()*100:.2f}% flagged; per-object flag rate quantiles {summ[['25%','50%','75%']].round(3).to_dict()}; errors={errs}")
    return fl

def main(force=False):
    if stage_done("S3_t2_maneuver") and not force: log.info("done"); return
    set_status("S3_t2_maneuver", "running")
    try:
        with Timer() as T:
            met, data = run_sentinels(); save_df(met, "t2_maneuver_metrics.csv")
            log.info("pooled:\n" + met[met.satellite == "ALL"][["method", "precision", "recall", "f1", "f2"]].round(3).to_string(index=False))
            run_cohort(data)
        set_status("S3_t2_maneuver", "done", elapsed_s=round(T.elapsed))
    except Exception as ex:
        log.exception("T2 failed"); set_status("S3_t2_maneuver", "failed", message=str(ex)); raise

if __name__ == "__main__":
    main("--force" in sys.argv)
