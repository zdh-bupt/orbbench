"""T1 Orbit-phase (observable motion state) segmentation from TLE sequences.

Silver labels follow STARLINK_MOTION_PHASE_TEST_CASE (rule table on fragment statistics), applied to
sliding fragments of N_WIN consecutive TLEs (stride N_WIN//2) for every cohort object over the window
(plus 60 d before). Fragments are cut at gaps > 36 h. Main classes: STABLE_ORBIT / SUSTAINED_RAISE /
SUSTAINED_DECAY; OFFSET_PLATEAU, SHORT_EXCURSION_RETURN reported as case classes; UNKNOWN excluded.

Methods (object-level 70/15/15 split, seeds 3)
  Constant-stable          control
  Slope-rule (no training) sign/magnitude of da/dt over the fragment (physics baseline)
  Raw PCA + KMeans(k=3)    unsupervised, Hungarian mapping on train (as in the existing use case)
  HistGB (supervised)      fragment statistics -> class, trained on silver labels
Metrics: Macro-F1, ARI, NMI, per-class F1; stratified by type x regime.
Output: results/t1_phase_metrics.csv, results/t1_phase_labels.parquet (norad, t_start, t_end, silver, pred_best) for T4/T6.
"""
from __future__ import annotations
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import f1_score, adjusted_rand_score, normalized_mutual_info_score
from scipy.optimize import linear_sum_assignment

from . import config as C
from .util import get_logger, set_status, stage_done, Timer, save_df, load_cohort, read_tles, elements_from_ne

log = get_logger("t1_phase")
N_WIN = 10
CLASSES = ["STABLE_ORBIT", "SUSTAINED_RAISE", "SUSTAINED_DECAY"]
CASE = ["OFFSET_PLATEAU", "SHORT_EXCURSION_RETURN"]
FEATS = ["n_win", "dur_d", "net_da_km", "range_a_km", "p_pos", "p_neg", "max_gap_h", "slope_km_d", "mean_e", "mean_inc", "mean_bstar", "std_da_km"]

def rule_label(N, T, dA, Ra, pp, pn) -> str:
    if N >= 3 and T < 10 and Ra >= 4 and abs(dA) <= 1 and pp >= 0.25 and pn >= 0.25: return "SHORT_EXCURSION_RETURN"
    if N >= 5 and T >= 10 and Ra >= 5 and abs(dA) <= 1 and pp >= 0.30 and pn >= 0.30: return "OFFSET_PLATEAU"
    if N >= 5 and T >= 2 and dA >= 4 and pp >= 0.55: return "SUSTAINED_RAISE"
    if N >= 5 and T >= 3 and dA <= -0.9 and pn >= 0.70: return "SUSTAINED_DECAY"
    if N >= 5 and Ra <= 2.5 and abs(dA) < 0.9: return "STABLE_ORBIT"
    return "UNKNOWN_TRANSITION"

def fragments_one(norad: int) -> pd.DataFrame | None:
    t = read_tles(norad, pd.Timestamp(C.WINDOW_START) - pd.Timedelta(days=60), C.WINDOW_END)
    if len(t) < N_WIN: return None
    a, _, _ = elements_from_ne(t.n.values, t.e.values)
    ep = t.epoch.values.astype("datetime64[ns]").astype("int64") / 3.6e12   # hours
    gaps = np.diff(ep); cut = np.where(gaps > 36)[0] + 1
    segs = np.split(np.arange(len(t)), cut)
    rows = []
    for s in segs:
        if len(s) < N_WIN: continue
        for st in range(0, len(s) - N_WIN + 1, N_WIN // 2):
            idx = s[st:st + N_WIN]
            aa = a[idx]; tt = ep[idx]; da = np.diff(aa); dt = np.diff(tt)
            rate = da / np.maximum(dt / 24.0, 1e-6)
            N = len(idx); T = (tt[-1] - tt[0]) / 24.0; dA = aa[-1] - aa[0]; Ra = aa.max() - aa.min()
            pp = float((rate > 0).mean()); pn = float((rate < 0).mean())
            rows.append(dict(norad=norad, t_start=t.epoch.iloc[idx[0]], t_end=t.epoch.iloc[idx[-1]], n_win=N, dur_d=T, net_da_km=dA, range_a_km=Ra,
                             p_pos=pp, p_neg=pn, max_gap_h=float(dt.max()), slope_km_d=float(dA / max(T, 1e-3)),
                             mean_e=float(t.e.values[idx].mean()), mean_inc=float(t.inc.values[idx].mean()), mean_bstar=float(t.bstar.values[idx].mean()),
                             std_da_km=float(np.std(da)), silver=rule_label(N, T, dA, Ra, pp, pn)))
    return pd.DataFrame(rows) if rows else None

def hungarian_map(y_true: np.ndarray, clusters: np.ndarray, classes) -> dict:
    k = clusters.max() + 1; M = np.zeros((k, len(classes)))
    for c in range(k):
        for j, cl in enumerate(classes): M[c, j] = np.sum((clusters == c) & (y_true == cl))
    r, c = linear_sum_assignment(-M)
    m = {int(ri): classes[ci] for ri, ci in zip(r, c)}
    return {c: m.get(c, classes[0]) for c in range(k)}

def evaluate(fr: pd.DataFrame) -> pd.DataFrame:
    main = fr[fr.silver.isin(CLASSES)].copy()
    obj = main.norad.unique(); rows = []
    for seed in (42, 43, 44):
        rng = np.random.default_rng(seed); perm = rng.permutation(obj)
        n = len(perm); tr_o, va_o, te_o = perm[: int(.7 * n)], perm[int(.7 * n): int(.85 * n)], perm[int(.85 * n):]
        tr, te = main[main.norad.isin(tr_o)], main[main.norad.isin(te_o)]
        yt = te.silver.values; preds = {}
        preds["Constant-stable"] = np.array(["STABLE_ORBIT"] * len(te))
        sl = te.slope_km_d.values
        preds["Slope-rule"] = np.where((sl >= 4 / 7) & (te.p_pos >= .55), "SUSTAINED_RAISE", np.where((sl <= -0.9 / 7) & (te.p_neg >= .7), "SUSTAINED_DECAY", "STABLE_ORBIT"))
        sc = StandardScaler().fit(tr[FEATS]); Ztr = PCA(5, random_state=seed).fit(sc.transform(tr[FEATS]))
        km = KMeans(3, n_init=10, random_state=seed).fit(Ztr.transform(sc.transform(tr[FEATS])))
        mp = hungarian_map(tr.silver.values, km.labels_, CLASSES)
        preds["Raw PCA + KMeans"] = np.array([mp[c] for c in km.predict(Ztr.transform(sc.transform(te[FEATS])))])
        clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=seed).fit(tr[FEATS], tr.silver)
        preds["HistGB"] = clf.predict(te[FEATS])
        for m, p in preds.items():
            r = dict(task="T1", seed=seed, method=m, n_test=len(te), n_objects=len(te_o),
                     macro_f1=f1_score(yt, p, labels=CLASSES, average="macro", zero_division=0),
                     ari=adjusted_rand_score(yt, p), nmi=normalized_mutual_info_score(yt, p))
            for cl in CLASSES: r[f"f1_{cl}"] = f1_score(yt == cl, p == cl, zero_division=0)
            rows.append(r)
            # stratified
            for (ty, rg), g in te.assign(pred=p).groupby(["type", "regime"]):
                if len(g) < 50: continue
                rows.append(dict(task="T1", seed=seed, method=m, stratum=f"{ty}|{rg}", n_test=len(g),
                                 macro_f1=f1_score(g.silver, g.pred, labels=CLASSES, average="macro", zero_division=0)))
        if seed == 42:
            fr.loc[te.index, "pred_best"] = preds["HistGB"]
    met = pd.DataFrame(rows)
    return met

def main(force=False):
    if stage_done("S2_t1_phase") and not force: log.info("done"); return
    set_status("S2_t1_phase", "running")
    try:
        with Timer() as T:
            obj = load_cohort()
            out = []
            with ProcessPoolExecutor(C.N_WORKERS) as ex:
                futs = {ex.submit(fragments_one, int(n)): int(n) for n in obj.norad}
                for i, fu in enumerate(as_completed(futs)):
                    r = fu.result()
                    if r is not None: out.append(r)
                    if (i + 1) % 500 == 0: log.info(f"fragments {i+1}/{len(futs)}")
            fr = pd.concat(out, ignore_index=True).merge(obj[["norad", "type", "regime"]], on="norad", how="left")
            log.info(f"fragments: {len(fr)} from {fr.norad.nunique()} objects; silver counts {fr.silver.value_counts().to_dict()}")
            fr["pred_best"] = None
            met = evaluate(fr); save_df(met, "t1_phase_metrics.csv")
            # cohort-wide prediction with model trained on all main-class fragments (for chained mode)
            main_ = fr[fr.silver.isin(CLASSES)]
            clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.05, class_weight="balanced", random_state=C.SEED).fit(main_[FEATS], main_.silver)
            fr["pred_all"] = clf.predict(fr[FEATS])
            fr.to_parquet(C.RESULTS / "t1_phase_labels.parquet", index=False)
            agg = met[met.stratum.isna()].groupby("method")[["macro_f1", "ari", "nmi"] + [f"f1_{c}" for c in CLASSES]].mean().round(3)
            log.info("mean over seeds:\n" + agg.to_string())
        set_status("S2_t1_phase", "done", elapsed_s=round(T.elapsed), n_fragments=int(len(fr)))
    except Exception as ex:
        log.exception("T1 failed"); set_status("S2_t1_phase", "failed", message=str(ex)); raise

if __name__ == "__main__":
    main("--force" in sys.argv)
