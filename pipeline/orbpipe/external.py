"""Same tasks on existing datasets (§5.3): does the OrbBench evaluation transfer, and what do the other datasets lack?

E54a  T2 maneuver detection on the Shorten et al. TLE benchmark (15 satellites, curated maneuver timestamps): the OrbBench
      SGP4-residual detector with the threshold tuned on Sentinel (transfer) and with the best threshold on this dataset,
      on the raw TLE files and after PAF-lite. Event-level P/R/F1/F2 with ±1-interval tolerance, per satellite and pooled.
E54b  T6 final-risk prediction on the full ESA Kelvins CDM dataset with the competition protocol (inputs: CDMs >= 2 d before
      TCA; target: risk of the last CDM; high risk = risk >= -6; score = MSE on high-risk events / F2): naive latest risk,
      Ridge on signed-log features (OrbBench T6-CDM method), HistGB; 5-fold CV by event. Also checks how the 199 OrbBench
      CDM events relate to Kelvins.
Outputs: results/e54a_shorten_t2.csv, e54b_kelvins_t6.csv, e54_dataset_relation.json
"""
from __future__ import annotations
import sys, json, time, zipfile, re, glob
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from . import config as C
from .util import get_logger, set_status, stage_done, Timer, save_df, mad
from .starlink_chain import read_txt
from .chain_proof import _t2_scores
from .experiments import paf_lite

log = get_logger("external")
EXT = C.DATA / "external"; SHORTEN = EXT / "shorten_tle_benchmark" / "processed_files"; KELVINS = EXT / "kelvins_cdm"
GEO = {"Fengyun-2D", "Fengyun-2E", "Fengyun-2F", "Fengyun-2H", "Fengyun-4A"}
KS = np.arange(2, 40, 0.5)

def sentinel_k() -> float:
    for p in (C.ROOT / "logs" / "t2_maneuver.log",):
        if p.exists():
            m = re.findall(r"k_res=([\d.]+)", p.read_text(errors="replace"))
            if m: return float(m[-1])
    return 18.0

def score_events(epochs_ns: np.ndarray, score: np.ndarray, man_ns: np.ndarray, k: float, tol: int = 1):
    """Detection at TLE i covers (t_{i-1-tol}, t_{i+tol}]; each maneuver matched once."""
    det = np.where(np.nan_to_num(score, nan=0.0) > k)[0]; det = det[det >= 1]
    matched = np.zeros(len(man_ns), bool); tp = 0
    for i in det:
        lo = epochs_ns[max(i - 1 - tol, 0)]; hi = epochs_ns[min(i + tol, len(epochs_ns) - 1)]
        cand = np.where((man_ns > lo) & (man_ns <= hi) & ~matched)[0]
        if len(cand): matched[cand[0]] = True; tp += 1
    fp = len(det) - tp; fn = int((~matched).sum()); p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
    return dict(tp=tp, fp=fp, fn=fn, n_pred=int(len(det)), precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-9), f2=5 * p * r / max(4 * p + r, 1e-9))

def e54a() -> pd.DataFrame:
    import yaml
    k_tr = sentinel_k(); rows = []; pooled = {}
    for tle_path in sorted(SHORTEN.glob("*.tle")):
        name = tle_path.stem; yml = SHORTEN / f"manoeuvres_{name}.yaml"
        if not yml.exists(): continue
        meta = yaml.safe_load(yml.read_text()); man = pd.to_datetime(pd.Series(meta["manoeuvre_timestamps"]).astype(str), utc=True).sort_values()
        t = read_txt(tle_path); t = t[(t.epoch >= man.min() - pd.Timedelta(days=30)) & (t.epoch <= man.max() + pd.Timedelta(days=30))].reset_index(drop=True)
        man = man[(man >= t.epoch.min()) & (man <= t.epoch.max())]; man_ns = man.values.astype("datetime64[ns]").astype("int64")
        if len(t) < 10 or len(man) == 0: continue
        gap_h = float(np.median(np.diff(t.epoch.values).astype("timedelta64[s]").astype(float)) / 3600)
        for pool, tt in (("raw", t), ("PAF-lite", paf_lite(t).reset_index(drop=True))):
            ep, sc = _t2_scores(tt)
            best_k = max(KS, key=lambda k: score_events(ep, sc, man_ns, k)["f1"])
            for setting, k in (("transfer (Sentinel k)", k_tr), ("best k on this dataset", best_k)):
                m = score_events(ep, sc, man_ns, k)
                rows.append(dict(dataset="Shorten TLE benchmark", satellite=name, regime="GEO" if name in GEO else "LEO", pool=pool, setting=setting, k=k, n_tle=len(tt), n_maneuvers=len(man), median_gap_h=gap_h, **m))
                key = (pool, setting, "GEO" if name in GEO else "LEO"); pooled.setdefault(key, []).append(m); pooled.setdefault((pool, setting, "ALL"), []).append(m)
        log.info(f"E54a {name}: {len(t)} TLEs, {len(man)} maneuvers, gap {gap_h:.1f} h")
    for (pool, setting, reg), ms in pooled.items():
        tp = sum(m["tp"] for m in ms); fp = sum(m["fp"] for m in ms); fn = sum(m["fn"] for m in ms); p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1)
        rows.append(dict(dataset="Shorten TLE benchmark", satellite=f"POOLED-{reg}", regime=reg, pool=pool, setting=setting, k=np.nan, n_tle=np.nan, n_maneuvers=tp + fn, median_gap_h=np.nan, tp=tp, fp=fp, fn=fn, n_pred=tp + fp, precision=p, recall=r, f1=2 * p * r / max(p + r, 1e-9), f2=5 * p * r / max(4 * p + r, 1e-9)))
    # OrbBench Sentinel reference rows (same detector, logs as truth)
    try:
        s = pd.read_csv(C.RESULTS / "t2_maneuver_metrics.csv"); s = s[s.method == "SGP4-residual-T"]
        for _, r in s.iterrows(): rows.append(dict(dataset="OrbBench (Sentinel POD logs)", satellite=r.satellite, regime="LEO", pool="PAF", setting="tuned on train years", k=np.nan, n_tle=np.nan, n_maneuvers=r.n_true, median_gap_h=np.nan, tp=np.nan, fp=np.nan, fn=np.nan, n_pred=r.n_pred, precision=r.precision, recall=r.recall, f1=r.f1, f2=r.f2))
    except Exception as ex: log.warning(f"Sentinel reference rows: {ex}")
    return pd.DataFrame(rows)

def _wait_zip(p: Path, timeout_s=3600):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            with zipfile.ZipFile(p) as z:
                if z.testzip() is None: return True
        except Exception: pass
        time.sleep(30)
    return False

def kelvins_features(d: pd.DataFrame, target_from: pd.DataFrame | None = None) -> pd.DataFrame:
    """Per event: features from the last CDM with time_to_tca >= 2 d (competition protocol); target = risk of the last CDM
    (train) or the private final risk (test)."""
    d = d.sort_values(["event_id", "time_to_tca"], ascending=[True, False]); pre_all = d[d.time_to_tca >= 2.0]
    pre = pre_all.groupby("event_id").tail(1).set_index("event_id")
    if target_from is None: y = d.groupby("event_id").tail(1).set_index("event_id").risk
    else: y = target_from.set_index("event_id")["true_risk" if "true_risk" in target_from.columns else "risk"]
    ev = pre.index.intersection(y.index); pre = pre.loc[ev]
    num = pre.select_dtypes(include=[np.number]).copy()
    g = pre_all.groupby("event_id"); num["n_cdm_pre"] = g.size().reindex(ev).values; num["risk_max_pre"] = g.risk.max().reindex(ev).values; num["risk_mean_pre"] = g.risk.mean().reindex(ev).values
    num["risk_slope_pre"] = (g.risk.last() - g.risk.first()).reindex(ev).values
    num["y"] = y.loc[ev].values; return num

def kelvins_score(y, yhat):
    hr = y >= -6; mse_hr = float(np.mean((yhat[hr] - y[hr]) ** 2)) if hr.any() else np.nan
    tp = int(((yhat >= -6) & hr).sum()); fp = int(((yhat >= -6) & ~hr).sum()); fn = int(((yhat < -6) & hr).sum()); p = tp / max(tp + fp, 1); r = tp / max(tp + fn, 1); f2 = 5 * p * r / max(4 * p + r, 1e-9)
    return dict(mse_hr=mse_hr, f2=f2, precision_hr=p, recall_hr=r, score=mse_hr / max(f2, 1e-9), n_events=len(y), n_high_risk=int(hr.sum()))

def _fit_predict(Xtr, ytr, Xte, kind):
    if kind == "ridge":
        Ztr = np.sign(Xtr) * np.log1p(np.abs(Xtr)); Zte = np.sign(Xte) * np.log1p(np.abs(Xte)); mu, sd = Ztr.mean(), Ztr.std().replace(0, 1)
        return np.clip(Ridge(alpha=10.0).fit((Ztr - mu) / sd, ytr).predict((Zte - mu) / sd), -30, 0)
    return np.clip(HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, l2_regularization=1.0, random_state=C.SEED).fit(Xtr, ytr).predict(Xte), -30, 0)

def e54b() -> tuple[pd.DataFrame, dict]:
    zp = KELVINS / "Collision Avoidance Challenge - Dataset.zip"
    if not _wait_zip(zp): raise RuntimeError("Kelvins zip not complete")
    with zipfile.ZipFile(zp) as z: z.extractall(KELVINS)
    for iz in glob.glob(str(KELVINS / "**" / "train_data.zip"), recursive=True):
        with zipfile.ZipFile(iz) as z: z.extractall(Path(iz).parent)
    csvs = glob.glob(str(KELVINS / "**" / "*.csv"), recursive=True); pick = lambda n: [f for f in csvs if Path(f).name.lower() == n][0]
    tr = pd.read_csv(pick("train_data.csv")); te = pd.read_csv(pick("test_data.csv")); pr = pd.read_csv(pick("test_data_private.csv"))
    log.info(f"Kelvins train {tr.shape} ({tr.event_id.nunique()} events); public test {te.shape} ({te.event_id.nunique()} events); private truth {pr.shape}")
    ours = pd.read_csv(C.CDM_CSV)
    key = ["event_id", "time_to_tca", "miss_distance"]; m = ours.merge(te[key].drop_duplicates(), on=key, how="left", indicator=True)
    prk = "true_risk" if "true_risk" in pr.columns else "risk"; j = ours.merge(pr[["event_id", prk, "time_to_tca", "miss_distance"]].rename(columns={prk: "risk_priv"}), on="event_id", suffixes=("", "_priv"))
    rel = dict(kelvins_train_rows=int(len(tr)), kelvins_train_events=int(tr.event_id.nunique()), kelvins_test_rows=int(len(te)), kelvins_test_events=int(te.event_id.nunique()), kelvins_test_time_to_tca_min=float(te.time_to_tca.min()),
               orbbench_cdm_rows=int(len(ours)), orbbench_events=int(ours.event_id.nunique()), orbbench_only_cols=[c for c in ours.columns if c not in tr.columns], kelvins_only_cols=[c for c in tr.columns if c not in ours.columns],
               orbbench_rows_found_in_kelvins_public_test=float(m._merge.eq("both").mean()), orbbench_true_risk_equals_kelvins_private_risk=float(np.isclose(j.true_risk, j.risk_priv).mean()) if "true_risk" in ours else None, orbbench_rows_identical_to_private_rows=float((np.isclose(j.time_to_tca, j.time_to_tca_priv) & np.isclose(j.miss_distance, j.miss_distance_priv)).mean()), kelvins_private_cols_first=list(pr.columns[:6]),
               orbbench_time_to_tca_range=[float(ours.time_to_tca.min()), float(ours.time_to_tca.max())],
               interpretation="OrbBench's 199 CDM rows are rows of test_data_private.csv: the final CDM (closest to TCA) of 199 of the 2,167 Kelvins test events, with the final risk as true_risk.")
    Ftr = kelvins_features(tr); Fte = kelvins_features(te, target_from=pr); cols = [c for c in Ftr.columns if c != "y" and c in Fte.columns]
    Xtr = Ftr[cols].replace([np.inf, -np.inf], np.nan); med = Xtr.median(); Xtr = Xtr.fillna(med); Xte = Fte[cols].replace([np.inf, -np.inf], np.nan).fillna(med); ytr = Ftr.y.values; yte = Fte.y.values
    preds = {"naive latest risk (>= 2 d before TCA)": Xte.risk.values}
    for kind, nm in (("ridge", "Ridge signed-log, all events"), ("gb", "HistGB, all events")): preds[nm] = _fit_predict(Xtr, ytr, Xte, kind)
    focus_tr = Xtr.risk.values >= -8; focus_te = Xte.risk.values >= -8
    for kind, nm in (("ridge", "Ridge signed-log, high-risk-focused (pre-risk >= -8, else naive)"), ("gb", "HistGB, high-risk-focused (pre-risk >= -8, else naive)")):
        yh = Xte.risk.values.copy(); yh[focus_te] = _fit_predict(Xtr[focus_tr], ytr[focus_tr], Xte[focus_te], kind); preds[nm] = yh
    yh = Xte.risk.values.copy(); res = _fit_predict(Xtr[focus_tr], ytr[focus_tr] - Xtr.risk.values[focus_tr], Xte[focus_te], "gb"); yh[focus_te] = np.clip(Xte.risk.values[focus_te] + res, -30, 0); preds["naive + HistGB residual, high-risk-focused"] = yh
    geo = [c for c in cols if c in ("risk", "time_to_tca", "miss_distance", "relative_speed", "mahalanobis_distance", "risk_max_pre", "risk_mean_pre", "risk_slope_pre") or c.startswith(("relative_position", "relative_velocity", "t_sigma_", "c_sigma_", "t_position_covariance", "c_position_covariance"))]
    yh = Xte.risk.values.copy(); yh[focus_te] = _fit_predict(Xtr.loc[focus_tr, geo], ytr[focus_tr], Xte.loc[focus_te, geo], "gb"); preds[f"HistGB, geometry + risk-history features only ({len(geo)}), high-risk-focused"] = yh
    yh = Xte.risk.values.copy(); res = _fit_predict(Xtr.loc[focus_tr, geo], ytr[focus_tr] - Xtr.risk.values[focus_tr], Xte.loc[focus_te, geo], "gb"); yh[focus_te] = np.clip(Xte.risk.values[focus_te] + res, -30, 0); preds["naive + HistGB residual, geometry features, high-risk-focused"] = yh
    rows = []
    sub = np.isin(Fte.index.values, ours.event_id.unique())
    for nm, yh in preds.items():
        rows.append(dict(dataset=f"Kelvins public test ({len(yte)} events, competition protocol)", method=nm, **kelvins_score(yte, yh)))
        rows.append(dict(dataset=f"Kelvins test, OrbBench 199-event subset (n={int(sub.sum())})", method=nm, **kelvins_score(yte[sub], yh[sub])))
    rel["n_focus_train_events"] = int(focus_tr.sum()); rel["n_focus_test_events"] = int(focus_te.sum())
    return pd.DataFrame(rows), rel

def main(force=False):
    if stage_done("S8i_external") and not force: log.info("done"); return
    set_status("S8i_external", "running")
    try:
        with Timer() as T:
            a = pd.read_csv(C.RESULTS / "e54a_shorten_t2.csv") if (C.RESULTS / "e54a_shorten_t2.csv").exists() and "--all" not in sys.argv else e54a(); save_df(a, "e54a_shorten_t2.csv"); log.info("E54a pooled:\n" + a[a.satellite.str.startswith("POOLED") | (a.dataset != "Shorten TLE benchmark")][["dataset", "satellite", "pool", "setting", "n_maneuvers", "precision", "recall", "f1", "f2"]].round(3).to_string(index=False))
            b, rel = e54b(); save_df(b, "e54b_kelvins_t6.csv"); (C.RESULTS / "e54_dataset_relation.json").write_text(json.dumps(rel, indent=1)); log.info("E54b:\n" + b.round(3).to_string(index=False)); log.info("relation: " + json.dumps(rel))
        set_status("S8i_external", "done", elapsed_s=round(T.elapsed))
    except Exception as ex:
        log.exception("external failed"); set_status("S8i_external", "failed", message=str(ex)); raise

if __name__ == "__main__":
    main(force="--force" in sys.argv)
