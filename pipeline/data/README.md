# pipeline/data — staged inputs shipped with the release

Only the small staged files are included here; everything else is produced by `python -m orbpipe.stage0_staging`
(and `stage_starlink`) from the raw sources listed in `../../DATA.md`.

| File | Produced by | Content |
|---|---|---|
| `objects.csv` | stage0 | cohort table: norad, source (pruned/multi), type, regime, status, DISCOS mass/area, path of the L2 CSV |
| `sentinel_man_S1A.csv`, `_S2A.csv`, `_S3A.csv` | stage0 | logged maneuver intervals (start, stop UTC) of Sentinel-1A/2A/3A (T2 truth) |
| `space_weather_daily.csv` | stage0 | date, f107_obs, kp_daily_avg, ap_avg (observed section of the daily table, label swap applied) |
| `Kp_ap_Ap_SN_F107_since_1932.txt` | GFZ Potsdam | definitive Kp/ap/Ap/SN/F10.7 table (CC BY 4.0; sunspot numbers CC BY-NC 4.0), read by `chain_ablation.py` |
| `norad_pruned_extracted.txt`, `norad_multi_extracted.txt` | stage0 | NORAD lists of the two L2 trajectory sources |
| `e2e_tles_600.parquet`, `e2e_tles_1500.parquet` | e2e | TLEs of the 600 / 1,500-object LEO cohorts used by the end-to-end run (T6/T7 rows of Table 4) |
| `starlink/summary.csv` | stage_starlink | per-satellite summary of the staged Starlink MEME ephemerides (the ephemerides themselves are not shipped) |
| `deep/train_manifest.json` | deep_forecast | the 120 Pruned LEO objects and time ranges used to retrain the six supervised forecasters |
| `deep/ckpt/*.pt` | deep_forecast | trained checkpoints (LSTM, GRU, DLinear, TCN, Informer, FEDformer) behind the T4 position-error rows |
| `deep/preds/*.npy` | deep_forecast | per-window predictions of all eight models on the Sentinel / Starlink / cohort test windows |

Not shipped (size): `traj_window.npz` (1.7 GB, cohort trajectories at 10-min cadence), `c4_leo_600_3d.npz` /
`c4_leo_1500_3d.npz` (56 / 123 MB, 1-min L2 reference of the LEO cohorts), `t6_shell_700_1000_450_3d.npz` (42 MB),
`sentinel_pod_*.parquet` (POD truth, 41–347 MB each), `starlink/meme_*.parquet` (468 MB), `deep/train/` (2.7 GB) and
`deep/windows/` (208 MB), `external/` (Kelvins CDM dataset and Shorten TLE benchmark, 675 MB), `smoke/` (32 MB).
