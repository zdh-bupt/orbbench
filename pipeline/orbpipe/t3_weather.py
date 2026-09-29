"""T3 Space-weather forecasting (F10.7, Kp daily mean, Ap) at 1/3/7-day horizons.

Baselines (CPU): persistence, climatology-27d (solar rotation recurrence), seasonal-naive-27,
AR ridge (lags 1..30 + 27-d recurrence), gradient boosting (sklearn HistGB) on lags.
Reported: Chronos-2 zero-shot (imported from 汇总（空间天气）).
Protocol: daily rolling origin over the window W; fit only on data before each origin's train cutoff
(fixed model fit on < WINDOW_START, no refit, to keep cost low and match "frozen model" protocol).
Outputs: results/t3_weather_metrics.csv, results/t3_weather_forecasts.parquet (used by T4 chained mode).
"""
from __future__ import annotations
import sys, json
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor

from . import config as C
from .util import get_logger, set_status, stage_done, Timer, save_df, to_utc

log = get_logger("t3_weather")
VARS = ["f107_obs", "kp_daily_avg", "ap_avg"]
HORIZONS = [1, 3, 7]
LAGS = 30

def load() -> pd.DataFrame:
    d = pd.read_csv(C.DATA / "space_weather_daily.csv", parse_dates=["date"])
    d = d.set_index("date").asfreq("D")
    d[VARS] = d[VARS].interpolate(limit=3)
    return d

def make_xy(s: pd.Series, h: int):
    X = np.column_stack([s.shift(k).values for k in range(0, LAGS)] + [s.shift(27 - h).values, s.rolling(27).mean().shift(0).values, s.rolling(81).mean().values])
    y = s.shift(-h).values
    return X, y

def evaluate() -> tuple[pd.DataFrame, pd.DataFrame]:
    d = load(); t0 = pd.Timestamp(C.WINDOW_START).tz_localize(None); t1 = pd.Timestamp(C.WINDOW_END).tz_localize(None)
    rows, fc = [], []
    for v in VARS:
        s = d[v]
        for h in HORIZONS:
            X, y = make_xy(s, h)
            idx = d.index
            tr = (idx < t0) & np.isfinite(X).all(1) & np.isfinite(y) & (idx >= pd.Timestamp("1990-01-01"))
            te = (idx >= t0) & (idx < t1) & np.isfinite(X).all(1) & np.isfinite(y)
            preds = {}
            preds["Persistence"] = X[te, 0]
            preds["Recurrence-27d"] = s.shift(27 - h).values[te]
            preds["Climatology-81d"] = s.rolling(81).mean().values[te]
            rg = Ridge(alpha=10.0).fit(X[tr], y[tr]); preds["Ridge-AR30"] = rg.predict(X[te])
            gb = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, max_leaf_nodes=15, random_state=C.SEED).fit(X[tr], y[tr])
            preds["HistGB-AR30"] = gb.predict(X[te])
            yt = y[te]; dates = idx[te] + pd.Timedelta(days=h)
            for m, p in preds.items():
                e = p - yt
                rows.append(dict(task="T3", variable=v, horizon_d=h, method=m, n=int(te.sum()),
                                 mae=float(np.mean(np.abs(e))), rmse=float(np.sqrt(np.mean(e ** 2))),
                                 bias=float(np.mean(e)), mape_pct=float(np.mean(np.abs(e) / np.maximum(np.abs(yt), 1e-6)) * 100),
                                 skill_vs_persistence=float(1 - np.mean(np.abs(e)) / np.mean(np.abs(preds["Persistence"] - yt)))))
                fc.append(pd.DataFrame({"target_date": dates, "origin_date": idx[te], "variable": v, "horizon_d": h, "method": m, "pred": p, "truth": yt}))
    met = pd.DataFrame(rows); fcd = pd.concat(fc, ignore_index=True)
    # import reported Chronos-2 backtest (ctx90, pred3/pred7) as reference rows (different period; flagged)
    try:
        rep = json.loads(C.REPORTED["weather_chronos2_backtest"].read_text())
        for cfg, hh in (("ctx90_pred3", 3), ("ctx90_pred7", 7)):
            for v in VARS:
                r = rep[cfg][v]
                met.loc[len(met)] = dict(task="T3", variable=v, horizon_d=hh, method="Chronos-2 (reported, 2025-06 backtest)", n=np.nan,
                                         mae=r["MAE_mean"], rmse=r["RMSE_mean"], bias=np.nan, mape_pct=np.nan, skill_vs_persistence=np.nan)
    except Exception as ex:
        log.warning(f"could not import Chronos-2 reported metrics: {ex}")
    return met, fcd

def main(force=False):
    if stage_done("S1_t3_weather") and not force: log.info("done"); return
    set_status("S1_t3_weather", "running")
    try:
        with Timer() as T:
            met, fcd = evaluate()
            save_df(met, "t3_weather_metrics.csv"); fcd.to_parquet(C.RESULTS / "t3_weather_forecasts.parquet", index=False)
            best = met[met.method != "Chronos-2 (reported, 2025-06 backtest)"].sort_values("mae").groupby(["variable", "horizon_d"]).head(1)
            log.info("best per var/horizon:\n" + best[["variable", "horizon_d", "method", "mae", "skill_vs_persistence"]].to_string(index=False))
        set_status("S1_t3_weather", "done", elapsed_s=round(T.elapsed), n_rows=int(len(met)))
    except Exception as ex:
        log.exception("T3 failed"); set_status("S1_t3_weather", "failed", message=str(ex)); raise

if __name__ == "__main__":
    main("--force" in sys.argv)
