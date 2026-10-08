"""Table 5 (data ablation) assembled from per-sample results already on disk. Every column is tied to one cohort and one
experiment, so a row is filled only where the removed data type actually enters that experiment; '--' otherwise.

Columns
  starlink_p90_4h_km, starlink_p95_4h_km, starlink_p50_1h_km : Starlink 17 sats vs MEME (s1_starlink_cases, test 2026-07-05..08-23)
  sentinel_p50_1h_km, sentinel_p95_1h_km                     : Sentinel 5 sats vs POD (t4_forecast_cases, test 2024-25H1)
  screening_recall_20pct                                     : 600-object LEO cohort, reference approaches < 10 km (x3_leo_chain)
  cdm_umc, cdm_dv_m_s                                        : 199 ESA CDM events (x1_cdm_chain)
  cohort_resolved, cohort_dv_m_s                             : 1,500-object cohort, 123 constructed dangerous approaches (e2e_decision)
Output: results/table5_data_ablation.csv
"""
from __future__ import annotations
import numpy as np, pandas as pd
from . import config as C
from .util import get_logger, save_df

log = get_logger("table5")

def main():
    s1 = pd.read_parquet(C.RESULTS / "s1_starlink_cases.parquet"); s1 = s1[s1.is_test]
    def sl(method, h, q): d = s1[(s1.method == method) & (s1.h == h)].err_3d; return float(d.quantile(q))
    t4 = pd.read_parquet(C.RESULTS / "t4_forecast_cases.parquet"); t4["t0"] = pd.to_datetime(t4.t0, utc=True); t4 = t4[t4.t0 >= pd.Timestamp(C.SENTINEL_TEST_START)]
    def se(method, h, q): d = t4[(t4.method == method) & (t4.h == h)].err_3d; return float(d.quantile(q))
    x3 = pd.read_csv(C.RESULTS / "x3_leo_chain.csv"); x3["chain"] = x3.chain.str.strip(); x3 = x3.set_index("chain")
    x1 = pd.read_csv(C.RESULTS / "x1_cdm_chain.csv"); x1["chain"] = x1.chain.str.strip(); x1 = x1.set_index("chain")
    e2 = pd.read_csv(C.RESULTS / "e2e_decision.csv").set_index("config")
    def cdm(k): return int(x1.loc[k, "unnecessary_maneuvers"]), float(x1.loc[k, "dv_total_m_s"])
    def coh(k): return int(e2.loc[k, "INJ_safe_after"]), float(e2.loc[k, "INJ_dv_m_s"])
    NA = np.nan
    rows = []
    def add(data, setting, sl_method, se_method, screen, cdm_key, coh_key, note=""):
        r = dict(data=data, setting=setting, note=note)
        r["starlink_p90_4h_km"] = sl(sl_method, 4.0, .90) if sl_method else NA; r["starlink_p95_4h_km"] = sl(sl_method, 4.0, .95) if sl_method else NA; r["starlink_p50_1h_km"] = sl(sl_method, 1.0, .50) if sl_method else NA
        r["sentinel_p50_1h_km"] = se(se_method, 1.0, .50) if se_method else NA; r["sentinel_p95_1h_km"] = se(se_method, 1.0, .95) if se_method else NA
        r["screening_recall_20pct"] = float(x3.loc[screen, "recall"]) if screen else NA
        r["cdm_umc"], r["cdm_dv_m_s"] = cdm(cdm_key) if cdm_key else (NA, NA)
        r["cohort_resolved"], r["cohort_dv_m_s"] = coh(coh_key) if coh_key else (NA, NA)
        rows.append(r)
    FULL_SL = "+PAF +MTF +gate"; FULL_SE = "HistGB-weather+T3"; FULL_X3 = "A1 full (T5 order -> T6 Pc -> T7 Pc-gate)"; FULL_X1 = "A1 full (T3+T5+T4 -> T6 Pc -> T7 Pc-gate)"; FULL_E2 = "A1 chained"
    add("All", "Full", FULL_SL, FULL_SE, FULL_X3, FULL_X1, FULL_E2, "Starlink: PAF+MTF with availability gate; Sentinel: POD-trained corrector with weather + T3")
    add("L2", "Raw TLE instead of L2 trajectories", "A0 raw prev", None, "-T5 (random order)", None, "A1 -T5 order (random)", "no continuous states -> no density ordering (random); Starlink forecast from the latest raw TLE; the Sentinel corrector does not use L2")
    add("L2", "Fixed sigma instead of OAD uncertainty", None, None, FULL_X3, "-T4 (fixed sigma, no uncertainty task)", "A1 -T4 sigma (fixed)", "point forecast unchanged; screening order unchanged (density, not sigma)")
    add("L3/L4", "w/o REF (no POD/MEME-trained corrector)", None, "SGP4-MTF", None, None, None, "REF also calibrates sigma: that effect is the fixed-sigma row")
    add("Supplementary", "w/o MAN", "+PAF +MTF", None, None, None, "A1 -T2 maneuver", "Starlink: filtered pool without maneuver awareness; the Sentinel corrector does not use maneuver information")
    add("Supplementary", "w/o SW", None, "HistGB-orbit", None, "-T3 (no space weather)", "A1 -T3 weather", "Sentinel corrector without weather features")
    add("Supplementary", "w/o PROP", None, None, None, "-T5 (no object/orbit attributes)", None, "")
    add("Supplementary", "w/o AUX (TLE only)", "A0 raw prev", "SGP4-prev", "A0 none (random order + distance rule)", "A0 none (fixed sigma + distance rule)", "A0 independent (random 20% budget)", "")
    out = pd.DataFrame(rows); save_df(out, "table5_data_ablation.csv"); log.info("\n" + out.round(3).to_string(index=False))

if __name__ == "__main__":
    main()
