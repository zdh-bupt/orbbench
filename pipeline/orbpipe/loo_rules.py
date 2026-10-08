"""Tag every PAF-lite-removed held-out TLE in the LOO samples with the rule that removed it, and compare the LOO
residual of removed TLEs by rule against kept TLEs (per type x regime). Output: results/e7_removed_by_rule.csv"""
from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
import numpy as np, pandas as pd
from . import config as C
from .util import get_logger, read_tles, to_utc, save_df
log = get_logger("loo_rules")

def rules_one(norad):
    try:
        t0, t1 = to_utc(C.WINDOW_START), to_utc(C.WINDOW_END)
        t = read_tles(norad, t0 - pd.Timedelta(days=2), t1 + pd.Timedelta(days=2)).sort_values("epoch").reset_index(drop=True)
        n = len(t)
        if n < 3: return None
        ns = t.epoch.values.astype("datetime64[ns]").astype("int64"); dt = np.diff(ns) / 1e9; T = 86400.0 / t.n.values
        half = np.zeros(n, bool); half[:-1] = dt < 0.5 * T[1:]
        bneg = t.bstar.values < 0
        nn = t.n.values; dn = np.abs(np.diff(nn)); med = np.median(dn) if len(dn) else 0; iso = np.zeros(n, bool)
        for i in range(1, n - 1):
            if abs(nn[i] - nn[i - 1]) > 20 * med + 1e-6 and abs(nn[i] - nn[i + 1]) > 20 * med + 1e-6 and abs(nn[i + 1] - nn[i - 1]) < 5 * med + 1e-6: iso[i] = True
        rule = np.where(iso, "isolated_n", np.where(bneg, "bstar_neg", np.where(half, "half_period", "kept")))
        return pd.DataFrame({"norad": norad, "epoch": t.epoch, "rule": rule})
    except Exception:
        return None

def main():
    S = pd.read_parquet(C.RESULTS / "e3_loo_samples.parquet"); S = S[S.method.isin(["nearest-raw", "MTF-raw"])]
    S["epoch"] = pd.to_datetime(S.epoch, utc=True)
    obj = pd.read_csv(C.DATA / "objects.csv"); out = []
    with ProcessPoolExecutor(C.N_WORKERS) as ex:
        for fu in as_completed([ex.submit(rules_one, int(n)) for n in S.norad.unique()]):
            r = fu.result()
            if r is not None: out.append(r)
    R = pd.concat(out, ignore_index=True); R["epoch"] = pd.to_datetime(R.epoch, utc=True)
    M = S.merge(R, on=["norad", "epoch"], how="left").merge(obj[["norad", "type", "regime"]], on="norad", how="left")
    M["rule"] = M.rule.fillna("kept")
    rows = []
    for (m, scope), g in [((m, "ALL"), g) for m, g in M.groupby("method")] + [((m, f"{t}|{r}"), g) for (m, t, r), g in M.groupby(["method", "type", "regime"]) if len(g) >= 300]:
        kept = g[g.rule == "kept"].res_3d
        for rule, gg in g.groupby("rule"):
            if len(gg) < 10: continue
            rows.append(dict(method=m, scope=scope, rule=rule, n=len(gg), res3d_p50=float(gg.res_3d.median()), res3d_p90=float(gg.res_3d.quantile(.9)),
                             ratio_p50_vs_kept=float(gg.res_3d.median() / max(kept.median(), 1e-9)), ratio_p90_vs_kept=float(gg.res_3d.quantile(.9) / max(kept.quantile(.9), 1e-9)),
                             frac_gt_1km=float((gg.res_3d > 1).mean()), kept_frac_gt_1km=float((kept > 1).mean())))
    out = pd.DataFrame(rows); save_df(out, "e7_removed_by_rule.csv")
    log.info("\n" + out[out.scope == "ALL"].round(4).to_string(index=False))

if __name__ == "__main__":
    main()
