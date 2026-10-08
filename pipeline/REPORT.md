# OrbBench unified STM pipeline — automated results

Generated 2026-09-28 14:14 UTC. Window 2025-01-01 → 2025-06-24; cohort = objects with L2 trajectories staged in `data/objects.csv`; truth cohort = Sentinel POD.

## Stage status
| stage | state | updated | elapsed_s | n_objects | n_rows | n_fragments | n_cases | n_truth_events | n_events | n_samples | n_states |
|---|---|---|---|---|---|---|---|---|---|---|---|
| S0_staging | done | 2026-09-13T09:25:05Z | 3258.000 | 3603.000 |  |  |  |  |  |  |  |
| S1_t3_weather | done | 2026-09-13T08:32:47Z | 2.000 |  | 51.000 |  |  |  |  |  |  |
| S3_t2_maneuver | done | 2026-09-13T15:43:47Z | 99.000 |  |  |  |  |  |  |  |  |
| S2_t1_phase | done | 2026-09-13T08:56:49Z | 108.000 |  |  | 187971.000 |  |  |  |  |  |
| S4_t4_forecast | done | 2026-09-13T16:15:39Z | 95.000 |  |  |  | 536310.000 |  |  |  |  |
| S5_t5_evolution | done | 2026-09-16T11:38:18Z | 1797.000 |  | 3123.000 |  |  |  |  |  |  |
| S6_t6_warning | done | 2026-09-16T12:02:59Z | 1481.000 |  |  |  |  | 47.000 |  |  |  |
| S7_t7_avoidance | done | 2026-09-13T11:10:17Z | 82.000 |  |  |  |  |  | 63.000 |  |  |
| S8_experiments | done | 2026-09-16T12:22:00Z | 8.000 |  |  |  |  |  |  |  |  |
| S9_report | running | 2026-09-28T14:14:09Z |  |  |  |  |  |  |  |  |  |
| S8b_e3_loo | done | 2026-09-13T15:05:33Z | 644.000 |  |  |  |  |  |  | 589223.000 |  |
| S8c_challenge_evidence | done | 2026-09-13T15:18:56Z | 74.000 |  |  |  |  |  |  |  |  |
| S8d_proof | done | 2026-09-13T16:49:57Z | 164.000 |  |  |  |  |  |  |  |  |
| S8e_chain_proof | done | 2026-09-16T12:11:37Z | 517.000 |  |  |  |  |  |  |  |  |
| S0b_starlink | done | 2026-09-14T04:53:03Z | 41.000 |  |  |  |  |  |  |  | 1584788.000 |
| S8f_starlink_chain | done | 2026-09-14T05:48:04Z | 1513.000 |  |  |  | 203400.000 |  |  |  |  |
| S8g_chain_ablation | done | 2026-09-16T12:20:22Z | 523.000 |  |  |  |  |  |  |  |  |
| S8i_external | done | 2026-09-16T06:09:08Z | 7.000 |  |  |  |  |  |  |  |  |
| S8h_exp_summary | done | 2026-09-16T12:21:51Z | 89.000 |  |  |  |  |  |  |  |  |
| S8j_e2e | done | 2026-09-16T12:49:17Z | 1636.000 |  |  |  |  |  |  |  |  |
| S8k_upstream_chain | done | 2026-09-16T08:17:33Z | 58.000 |  |  |  |  |  |  |  |  |
| S8l_detail_table | done | 2026-09-17T05:01:19Z | 242.000 |  |  |  |  |  |  |  |  |
| S8m_deep_forecast | done | 2026-09-20T06:32:21Z | 3640.000 |  |  |  |  |  |  |  |  |

## Cohort
| type | GEO | HEO | LEO | MEO | VLEO |
|---|---|---|---|---|---|
| DEB | 1 | 4 | 1510 | 87 | 2 |
| PAY | 206 | 105 | 633 | 156 | 10 |
| R/B | 76 | 85 | 338 | 169 | 2 |

mean window coverage of staged trajectories: 0.926

## E0 Benchmark table (headline metrics)

### T3 space weather
| subtask | method | MAE |
|---|---|---|
| ap_avg +3d | Chronos-2 (reported, 2025-06 backtest) | 13.550 |
| ap_avg +3d | Climatology-81d | 9.121 |
| ap_avg +3d | HistGB-AR30 | 8.533 |
| ap_avg +3d | Persistence | 12.157 |
| ap_avg +3d | Recurrence-27d | 9.017 |
| ap_avg +3d | Ridge-AR30 | 8.656 |
| f107_obs +3d | Chronos-2 (reported, 2025-06 backtest) | 11.225 |
| f107_obs +3d | Climatology-81d | 21.408 |
| f107_obs +3d | HistGB-AR30 | 13.008 |
| f107_obs +3d | Persistence | 14.476 |
| f107_obs +3d | Recurrence-27d | 23.099 |
| f107_obs +3d | Ridge-AR30 | 14.206 |
| kp_daily_avg +3d | Chronos-2 (reported, 2025-06 backtest) | 0.878 |
| kp_daily_avg +3d | Climatology-81d | 0.897 |
| kp_daily_avg +3d | HistGB-AR30 | 0.807 |
| kp_daily_avg +3d | Persistence | 1.106 |
| kp_daily_avg +3d | Recurrence-27d | 0.874 |
| kp_daily_avg +3d | Ridge-AR30 | 0.783 |

### T1 phase
| subtask | method | Macro-F1 |
|---|---|---|
| 3-class | Constant-stable | 0.329 |
| 3-class | HistGB | 0.666 |
| 3-class | Raw PCA + KMeans | 0.398 |
| 3-class | Slope-rule | 0.587 |

### T2 maneuver
| subtask | method | F2 | Recall |
|---|---|---|---|
| pooled Sentinels | CUSUM(Δa) | 0.167 | 1.000 |
| pooled Sentinels | HistGB | 0.155 | 0.140 |
| pooled Sentinels | HistGB-LOSO | 0.436 | 0.496 |
| pooled Sentinels | SGP4-residual-3D | 0.537 | 0.669 |
| pooled Sentinels | SGP4-residual-T | 0.538 | 0.643 |
| pooled Sentinels | Δa-MAD | 0.450 | 0.842 |

### T4 forecast
| subtask | method | 3D P50 km |
|---|---|---|
| +1 h vs POD | HistGB-orbit | 0.311 |
| +1 h vs POD | HistGB-weather | 0.339 |
| +1 h vs POD | HistGB-weather+T3 | 0.332 |
| +1 h vs POD | Kepler-24h | 39.448 |
| +1 h vs POD | Ridge-orbit | 0.792 |
| +1 h vs POD | Ridge-weather | 0.795 |
| +1 h vs POD | Ridge-weather+T3 | 0.773 |
| +1 h vs POD | SGP4-MTF | 0.571 |
| +1 h vs POD | SGP4-MTF+T1+T2 | 0.571 |
| +1 h vs POD | SGP4-MTF+T2 | 0.571 |
| +1 h vs POD | SGP4-prev | 0.571 |
| +24 h vs POD | HistGB-orbit | 0.525 |
| +24 h vs POD | HistGB-weather | 0.548 |
| +24 h vs POD | HistGB-weather+T3 | 0.542 |
| +24 h vs POD | Kepler-24h | 722.392 |
| +24 h vs POD | Ridge-orbit | 0.482 |
| +24 h vs POD | Ridge-weather | 0.489 |
| +24 h vs POD | Ridge-weather+T3 | 0.497 |
| +24 h vs POD | SGP4-MTF | 0.457 |
| +24 h vs POD | SGP4-MTF+T1+T2 | 0.455 |
| +24 h vs POD | SGP4-MTF+T2 | 0.455 |
| +24 h vs POD | SGP4-prev | 0.455 |

### T5 evolution
| subtask | method | overlap % |
|---|---|---|
| +1 h density | Kepler | 88.912 |
| +1 h density | Persistence | 75.431 |
| +1 h density | SGP4-prev | 97.572 |
| +24 h density | Kepler | 81.594 |
| +24 h density | Persistence | 75.724 |
| +24 h density | SGP4-prev | 97.534 |

### T6 warning
| subtask | method | recall |
|---|---|---|
| lead 24 h, Chained-T5 (10% budget) | SGP4-prev screening | 0.255 |
| lead 24 h, Chained-T5 (25% budget) | SGP4-prev screening | 0.383 |
| lead 24 h, Chained-T5 (50% budget) | SGP4-prev screening | 0.489 |
| lead 24 h, Independent | SGP4-prev screening | 0.915 |
| lead 6 h, Chained-T5 (10% budget) | SGP4-prev screening | 0.255 |
| lead 6 h, Chained-T5 (25% budget) | SGP4-prev screening | 0.383 |
| lead 6 h, Chained-T5 (50% budget) | SGP4-prev screening | 0.489 |
| lead 6 h, Independent | SGP4-prev screening | 0.915 |

### T6 CDM risk
| subtask | method | MAE |
|---|---|---|
| final log10 Pc | HistGB (5-fold) | 0.263 |
| final log10 Pc | Last-estimate | 0.524 |
| final log10 Pc | Ridge (5-fold) | 0.445 |

### T7 avoidance
| subtask | method | safety rate | Δv m/s |
|---|---|---|---|
| all events | No-maneuver | 0.000 | 0.000 |
| all events | PPO | 1.000 | 45.000 |
| all events | Physics-Pc | 0.000 | 0.114 |
| all events | Physics-max | 1.000 | 9.843 |
| all events | Proportional | 0.032 | 10.520 |

## T3 space weather (MAE by horizon)
| variable | method | 1 | 3 | 7 |
|---|---|---|---|---|
| ap_avg | Chronos-2 (reported, 2025-06 backtest) |  | 13.550 | 10.956 |
| ap_avg | Climatology-81d | 9.022 | 9.121 | 9.113 |
| ap_avg | HistGB-AR30 | 6.974 | 8.533 | 8.320 |
| ap_avg | Persistence | 9.115 | 12.157 | 14.827 |
| ap_avg | Recurrence-27d | 9.040 | 9.017 | 8.988 |
| ap_avg | Ridge-AR30 | 7.495 | 8.656 | 8.731 |
| f107_obs | Chronos-2 (reported, 2025-06 backtest) |  | 11.225 | 9.548 |
| f107_obs | Climatology-81d | 20.643 | 21.408 | 22.142 |
| f107_obs | HistGB-AR30 | 6.959 | 13.008 | 19.685 |
| f107_obs | Persistence | 7.059 | 14.476 | 22.962 |
| f107_obs | Recurrence-27d | 23.134 | 23.099 | 23.395 |
| f107_obs | Ridge-AR30 | 8.741 | 14.206 | 18.930 |
| kp_daily_avg | Chronos-2 (reported, 2025-06 backtest) |  | 0.878 | 0.861 |
| kp_daily_avg | Climatology-81d | 0.887 | 0.897 | 0.897 |
| kp_daily_avg | HistGB-AR30 | 0.664 | 0.807 | 0.787 |
| kp_daily_avg | Persistence | 0.827 | 1.106 | 1.414 |
| kp_daily_avg | Recurrence-27d | 0.882 | 0.874 | 0.866 |
| kp_daily_avg | Ridge-AR30 | 0.659 | 0.783 | 0.785 |

## T1 phase segmentation (mean over 3 seeds)
| method | macro_f1 | ari | nmi | f1_STABLE_ORBIT | f1_SUSTAINED_RAISE | f1_SUSTAINED_DECAY |
|---|---|---|---|---|---|---|
| Constant-stable | 0.329 | 0.000 | 0.000 | 0.988 | 0.000 | 0.000 |
| HistGB | 0.666 | 0.998 | 0.993 | 1.000 | 0.000 | 0.998 |
| Raw PCA + KMeans | 0.398 | 0.028 | 0.073 | 0.943 | 0.000 | 0.250 |
| Slope-rule | 0.587 | 0.750 | 0.619 | 0.993 | 0.000 | 0.768 |

Silver labels follow the Starlink observable-state rules; no SUSTAINED_RAISE exists in this legacy-catalog cohort and most SUSTAINED_DECAY windows are MEO objects with lunisolar-driven semi-major-axis trends, so the class semantics differ from the Starlink use case.

## T2 maneuver detection (Sentinel logs as truth, ±1 interval)
| satellite | method | precision | recall | f1 | f2 | n_true | n_pred |
|---|---|---|---|---|---|---|---|
| S1A | Δa-MAD | 0.131 | 0.846 | 0.227 | 0.405 | 195.000 | 1257.000 |
| S1A | SGP4-residual-T | 0.326 | 0.554 | 0.411 | 0.486 | 195.000 | 331.000 |
| S1A | SGP4-residual-3D | 0.297 | 0.595 | 0.397 | 0.496 | 195.000 | 390.000 |
| S1A | CUSUM(Δa) | 0.083 | 1.000 | 0.153 | 0.310 | 195.000 | 2361.000 |
| S1A | HistGB | 0.731 | 0.097 | 0.172 | 0.118 | 195.000 | 26.000 |
| S1A | HistGB-LOSO | 0.284 | 0.374 | 0.323 | 0.352 | 195.000 | 257.000 |
| S1A | Never-alarm | 0.000 | 0.000 | 0.000 | 0.000 | 195.000 | 0.000 |
| S2A | Δa-MAD | 0.397 | 0.763 | 0.523 | 0.644 | 38.000 | 73.000 |
| S2A | SGP4-residual-T | 0.323 | 0.842 | 0.467 | 0.637 | 38.000 | 99.000 |
| S2A | SGP4-residual-3D | 0.369 | 0.816 | 0.508 | 0.657 | 38.000 | 84.000 |
| S2A | CUSUM(Δa) | 0.016 | 1.000 | 0.032 | 0.077 | 38.000 | 2310.000 |
| S2A | HistGB | 0.183 | 0.289 | 0.224 | 0.259 | 38.000 | 60.000 |
| S2A | HistGB-LOSO | 0.342 | 0.711 | 0.462 | 0.584 | 38.000 | 79.000 |
| S2A | Never-alarm | 0.000 | 0.000 | 0.000 | 0.000 | 38.000 | 0.000 |
| S3A | Δa-MAD | 0.278 | 0.897 | 0.424 | 0.621 | 39.000 | 126.000 |
| S3A | SGP4-residual-T | 0.327 | 0.897 | 0.479 | 0.665 | 39.000 | 107.000 |
| S3A | SGP4-residual-3D | 0.265 | 0.897 | 0.409 | 0.608 | 39.000 | 132.000 |
| S3A | CUSUM(Δa) | 0.016 | 1.000 | 0.032 | 0.077 | 39.000 | 2393.000 |
| S3A | HistGB | 0.154 | 0.205 | 0.176 | 0.192 | 39.000 | 52.000 |
| S3A | HistGB-LOSO | 0.285 | 0.897 | 0.432 | 0.627 | 39.000 | 123.000 |
| S3A | Never-alarm | 0.000 | 0.000 | 0.000 | 0.000 | 39.000 | 0.000 |
| ALL | CUSUM(Δa) | 0.039 | 1.000 | 0.074 | 0.167 |  |  |
| ALL | HistGB | 0.275 | 0.140 | 0.185 | 0.155 |  |  |
| ALL | HistGB-LOSO | 0.294 | 0.496 | 0.369 | 0.436 |  |  |
| ALL | SGP4-residual-3D | 0.300 | 0.669 | 0.415 | 0.537 |  |  |
| ALL | SGP4-residual-T | 0.326 | 0.643 | 0.433 | 0.538 |  |  |
| ALL | Δa-MAD | 0.157 | 0.842 | 0.265 | 0.450 |  |  |

## T4 orbit forecasting vs POD (3D P50 km, all Sentinels)
| method | 0.5 | 1.0 | 4.0 | 8.0 | 12.0 | 24.0 |
|---|---|---|---|---|---|---|
| HistGB-orbit | 0.270 | 0.311 | 0.381 | 0.310 | 0.409 | 0.525 |
| HistGB-weather | 0.299 | 0.339 | 0.402 | 0.348 | 0.452 | 0.548 |
| HistGB-weather+T3 | 0.296 | 0.332 | 0.393 | 0.338 | 0.445 | 0.542 |
| Kepler-24h | 10.509 | 39.448 | 119.384 | 248.103 | 363.561 | 722.392 |
| Ridge-orbit | 0.697 | 0.792 | 0.644 | 0.604 | 0.516 | 0.482 |
| Ridge-weather | 0.707 | 0.795 | 0.641 | 0.609 | 0.525 | 0.489 |
| Ridge-weather+T3 | 0.686 | 0.773 | 0.622 | 0.640 | 0.522 | 0.497 |
| SGP4-MTF | 0.434 | 0.571 | 0.495 | 0.733 | 0.398 | 0.457 |
| SGP4-MTF+T1+T2 | 0.434 | 0.571 | 0.496 | 0.731 | 0.400 | 0.455 |
| SGP4-MTF+T2 | 0.434 | 0.571 | 0.496 | 0.731 | 0.400 | 0.455 |
| SGP4-prev | 0.434 | 0.571 | 0.495 | 0.731 | 0.399 | 0.455 |

### P95
| method | 0.5 | 1.0 | 4.0 | 8.0 | 12.0 | 24.0 |
|---|---|---|---|---|---|---|
| HistGB-orbit | 1.510 | 1.601 | 1.684 | 2.142 | 2.447 | 3.957 |
| HistGB-weather | 1.554 | 1.590 | 1.719 | 2.112 | 2.384 | 3.954 |
| HistGB-weather+T3 | 1.542 | 1.591 | 1.720 | 2.127 | 2.398 | 3.966 |
| Kepler-24h | 27.859 | 100.088 | 314.193 | 679.481 | 968.817 | 1949.688 |
| Ridge-orbit | 1.816 | 1.953 | 1.807 | 1.817 | 2.026 | 3.893 |
| Ridge-weather | 1.878 | 1.984 | 1.858 | 1.883 | 2.187 | 3.904 |
| Ridge-weather+T3 | 1.931 | 2.034 | 1.886 | 1.923 | 2.233 | 3.948 |
| SGP4-MTF | 1.081 | 1.219 | 1.695 | 2.147 | 2.249 | 4.216 |
| SGP4-MTF+T1+T2 | 1.065 | 1.222 | 1.695 | 2.147 | 2.258 | 4.246 |
| SGP4-MTF+T2 | 1.065 | 1.222 | 1.695 | 2.147 | 2.258 | 4.246 |
| SGP4-prev | 1.061 | 1.218 | 1.703 | 2.140 | 2.248 | 4.288 |

### +24 h by maneuver presence
| method | maneuver | quiet |
|---|---|---|
| HistGB-orbit | 0.822 | 0.503 |
| HistGB-weather | 0.816 | 0.522 |
| HistGB-weather+T3 | 0.810 | 0.525 |
| Kepler-24h | 1007.337 | 713.488 |
| Ridge-orbit | 0.877 | 0.446 |
| Ridge-weather | 0.865 | 0.455 |
| Ridge-weather+T3 | 0.861 | 0.462 |
| SGP4-MTF | 0.969 | 0.429 |
| SGP4-MTF+T1+T2 | 0.954 | 0.429 |
| SGP4-MTF+T2 | 0.954 | 0.429 |
| SGP4-prev | 0.960 | 0.429 |

## T4 uncertainty (in-window)
| model | component | cov68 | cov95 | nees | spearman_sigma_abserr | sigma_median_km |
|---|---|---|---|---|---|---|
| σ-fixed | r | 0.810 | 0.997 | 0.546 |  | 0.145 |
| σ-fixed | t | 0.674 | 0.923 | 5.468 |  | 0.655 |
| σ-fixed | n | 0.900 | 0.995 | 0.480 |  | 0.249 |
| σ-age | r | 0.795 | 0.989 | 0.604 | 0.073 | 0.134 |
| σ-age | t | 0.670 | 0.925 | 4.070 | 0.069 | 0.647 |
| σ-age | n | 0.898 | 0.994 | 0.484 | 0.018 | 0.246 |
| σ-oad | r | 0.801 | 0.995 | 0.573 | 0.058 | 0.137 |
| σ-oad | t | 0.647 | 0.923 | 4.703 | 0.091 | 0.596 |
| σ-oad | n | 0.883 | 0.993 | 0.538 | 0.047 | 0.228 |

## T5 population evolution (mean over origins)
| method | h | overlap_pct | tv | regional_rel_mae | hotspot_f1 | shell_spearman |
|---|---|---|---|---|---|---|
| Kepler | 1 | 88.912 | 0.111 | 0.157 | 0.777 | 0.989 |
| Kepler | 6 | 88.678 | 0.113 | 0.134 | 0.768 | 0.994 |
| Kepler | 24 | 81.594 | 0.184 | 0.250 | 0.645 | 0.990 |
| Persistence | 1 | 75.431 | 0.246 | 0.426 | 0.539 | 0.987 |
| Persistence | 6 | 75.711 | 0.243 | 0.403 | 0.535 | 0.988 |
| Persistence | 24 | 75.724 | 0.243 | 0.412 | 0.542 | 0.988 |
| SGP4-prev | 1 | 97.572 | 0.024 | 0.045 | 0.921 | 0.999 |
| SGP4-prev | 6 | 97.589 | 0.024 | 0.046 | 0.923 | 0.999 |
| SGP4-prev | 24 | 97.534 | 0.025 | 0.047 | 0.923 | 0.999 |

## T6 conjunction screening
| task | lead_h | mode | n_pairs_screened | n_truth | n_pred | tle_age_med_h | recall | precision | tca_err_med_min | miss_err_med_km | candidate_reduction_pct | n_danger | recall_at_k | recall_at_2k |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T6 | 6 | Independent | 68849.000 | 47.000 | 290 | 13.652 | 0.915 | 0.148 | 0.000 | 0.101 | 0.000 |  |  |  |
| T6 | 6 | Chained-T5 (50% budget) | 34424.000 | 47.000 | 150 | 13.652 | 0.489 | 0.153 | 0.000 | 0.109 | 50.001 |  |  |  |
| T6 | 6 | Chained-T5 (25% budget) | 17212.000 | 47.000 | 111 | 13.652 | 0.383 | 0.162 | 0.000 | 0.107 | 75.000 |  |  |  |
| T6 | 6 | Chained-T5 (10% budget) | 6884.000 | 47.000 | 64 | 13.652 | 0.255 | 0.188 | 0.000 | 0.073 | 90.001 |  |  |  |
| T6 | 24 | Independent | 68849.000 | 47.000 | 290 | 36.458 | 0.915 | 0.148 | 0.000 | 0.207 | 0.000 |  |  |  |
| T6 | 24 | Chained-T5 (50% budget) | 34424.000 | 47.000 | 150 | 36.458 | 0.489 | 0.153 | 0.000 | 0.175 | 50.001 |  |  |  |
| T6 | 24 | Chained-T5 (25% budget) | 17212.000 | 47.000 | 111 | 36.458 | 0.383 | 0.162 | 0.000 | 0.165 | 75.000 |  |  |  |
| T6 | 24 | Chained-T5 (10% budget) | 6884.000 | 47.000 | 64 | 36.458 | 0.255 | 0.188 | 0.000 | 0.117 | 90.001 |  |  |  |

### Risk ranking with uncertainty
| lead_h | mode | n_pred | n_danger | recall_at_k | recall_at_2k |
|---|---|---|---|---|---|
| 6 | UQ-point | 290 | 43.000 | 0.884 | 1.000 |
| 6 | UQ-fixed | 290 | 43.000 | 0.884 | 1.000 |
| 6 | UQ-age | 290 | 43.000 | 0.837 | 0.977 |
| 24 | UQ-point | 290 | 43.000 | 0.814 | 0.953 |
| 24 | UQ-fixed | 290 | 43.000 | 0.837 | 0.953 |
| 24 | UQ-age | 290 | 43.000 | 0.837 | 0.977 |

## T6 CDM final-risk prediction (ESA dataset, 199 events)
| task | method | n | mae_log10 | rmse_log10 | highrisk_precision | highrisk_recall | highrisk_f2 | n_highrisk |
|---|---|---|---|---|---|---|---|---|
| T6-CDM | Last-estimate | 199 | 0.524 | 0.834 | 0.882 | 1.000 | 0.974 | 150 |
| T6-CDM | Ridge (5-fold) | 199 | 0.445 | 0.581 | 0.898 | 0.880 | 0.884 | 150 |
| T6-CDM | HistGB (5-fold) | 199 | 0.263 | 0.369 | 0.972 | 0.940 | 0.946 | 150 |

## T7 autonomous avoidance
| task | strategy | source | n | safety_rate | dv_mean_m_s | dv_max_m_s | thrust_s_mean | pc_post_median | final_dist_med_m |
|---|---|---|---|---|---|---|---|---|---|
| T7 | No-maneuver | cdm | 59 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 454.679 |
| T7 | No-maneuver | screening | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1248.089 |
| T7 | PPO | cdm | 59 | 1.000 | 45.000 | 45.000 | 150.000 | 0.000 | 3852.168 |
| T7 | PPO | screening | 4 | 1.000 | 45.000 | 45.000 | 150.000 | 0.000 | 4622.217 |
| T7 | Physics-Pc | cdm | 59 | 0.000 | 0.122 | 1.800 | 0.407 | 0.000 | 480.343 |
| T7 | Physics-Pc | screening | 4 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 1248.089 |
| T7 | Physics-max | cdm | 59 | 1.000 | 10.078 | 16.200 | 33.593 | 0.000 | 2017.328 |
| T7 | Physics-max | screening | 4 | 1.000 | 6.375 | 12.600 | 21.250 | 0.000 | 2062.335 |
| T7 | Proportional | cdm | 59 | 0.000 | 10.757 | 16.579 | 150.000 | 0.000 | 1720.976 |
| T7 | Proportional | screening | 4 | 0.500 | 7.029 | 13.342 | 150.000 | 0.000 | 1957.345 |
| T7 | No-maneuver | ALL | 63 | 0.000 | 0.000 | 0.000 | 0.000 | 0.000 | 489.150 |
| T7 | PPO | ALL | 63 | 1.000 | 45.000 | 45.000 | 150.000 | 0.000 | 3886.637 |
| T7 | Physics-Pc | ALL | 63 | 0.000 | 0.114 | 1.800 | 0.381 | 0.000 | 490.947 |
| T7 | Physics-max | ALL | 63 | 1.000 | 9.843 | 16.200 | 32.810 | 0.000 | 2019.055 |
| T7 | Proportional | ALL | 63 | 0.032 | 10.520 | 16.579 | 150.000 | 0.000 | 1727.201 |

## STARLINK END-TO-END: A0 (independent tasks) vs A1 (connected chain), MEME truth with operator covariance, test 2026-07-05 .. 08-23
| stage | metric | A0_raw_prev | PAF_alone | PAF_MTF_T2_no_gate | A1_PAF_MTF_gate | A1_T1aware_PAFlite_MTF |
|---|---|---|---|---|---|---|
| T4 forecast vs MEME (17 Starlink, test 2026-07-05..08-23) | 3D P50 km, +1 h | 0.75 | 0.900 | 0.9 | 0.770 | 0.76 |
| T4 forecast vs MEME (17 Starlink, test 2026-07-05..08-23) | 3D P90 km, +1 h | 3.85 | 31.710 | 31.71 | 4.060 | 3.87 |
| T4 forecast vs MEME (17 Starlink, test 2026-07-05..08-23) | 3D P50 km, +4 h | 0.99 | 1.150 | 1.16 | 1.020 | 0.99 |
| T4 forecast vs MEME (17 Starlink, test 2026-07-05..08-23) | 3D P90 km, +4 h | 5.16 | 38.670 | 38.67 | 5.440 | 5.24 |
| T4 forecast vs MEME (17 Starlink, test 2026-07-05..08-23) | 3D P50 km, +24 h | 5.32 | 5.980 | 6.01 | 5.520 | 5.42 |
| T4 forecast vs MEME (17 Starlink, test 2026-07-05..08-23) | 3D P90 km, +24 h | 27.21 | 110.990 | 110.99 | 27.310 | 27.28 |
| UQ along-track, 95%/68% coverage | fixed sigma on raw prev | age sigma | age x g(T2) | 0.811/0.684 |  | 0.580/0.443 |  | 0.656/0.579 |
| T6 warning lead 6 h (134 MEME approaches < 50 km, none with true Pc > 0) | recall < 50 km / precision | 0.067 / 0.015 |  | 0.082 / 0.018 |  | nan |
| T6 warning lead 24 h (134 MEME approaches < 50 km, none with true Pc > 0) | recall < 50 km / precision | 0.075 / 0.018 |  | 0.097 / 0.022 |  | nan |

## S1 Starlink forecast vs MEME (3D P50 km) by method and horizon
| split | method | 1.0 | 4.0 | 12.0 | 24.0 |
|---|---|---|---|---|---|
| all | +PAF +MTF | 0.904 | 1.166 | 2.568 | 6.009 |
| all | +PAF +MTF +T2 | 0.898 | 1.162 | 2.539 | 6.009 |
| all | +PAF +MTF +T2 +T1 | 0.895 | 1.155 | 2.532 | 5.988 |
| all | +PAF +MTF +gate | 0.773 | 1.016 | 2.225 | 5.515 |
| all | +PAF prev | 0.895 | 1.152 | 2.526 | 5.981 |
| all | +PAF-lite(T1-aware) +MTF | 0.761 | 0.994 | 2.198 | 5.422 |
| all | +PAF-lite(blind) +MTF | 0.763 | 0.997 | 2.200 | 5.374 |
| all | A0 raw prev | 0.748 | 0.986 | 2.166 | 5.316 |
| all | raw MTF | 0.759 | 0.992 | 2.203 | 5.464 |
| all | raw MTF +T2 | 0.752 | 0.989 | 2.190 | 5.415 |
| no-recent-thrust | +PAF +MTF | 1.721 | 1.686 | 2.589 | 6.253 |
| no-recent-thrust | +PAF +MTF +T2 | 1.721 | 1.686 | 2.589 | 6.253 |
| no-recent-thrust | +PAF +MTF +T2 +T1 | 1.721 | 1.686 | 2.589 | 6.253 |
| no-recent-thrust | +PAF +MTF +gate | 0.790 | 0.898 | 2.268 | 6.253 |
| no-recent-thrust | +PAF prev | 1.721 | 1.686 | 2.589 | 6.253 |
| no-recent-thrust | +PAF-lite(T1-aware) +MTF | 0.790 | 0.898 | 2.268 | 6.253 |
| no-recent-thrust | +PAF-lite(blind) +MTF | 0.790 | 0.898 | 2.268 | 6.253 |
| no-recent-thrust | A0 raw prev | 0.780 | 0.892 | 2.299 | 6.253 |
| no-recent-thrust | raw MTF | 0.790 | 0.898 | 2.268 | 6.253 |
| no-recent-thrust | raw MTF +T2 | 0.790 | 0.898 | 2.268 | 6.253 |
| post-thrust(48h) | +PAF +MTF | 0.901 | 1.166 | 2.568 | 6.009 |
| post-thrust(48h) | +PAF +MTF +T2 | 0.898 | 1.162 | 2.539 | 6.009 |
| post-thrust(48h) | +PAF +MTF +T2 +T1 | 0.895 | 1.155 | 2.532 | 5.988 |
| post-thrust(48h) | +PAF +MTF +gate | 0.773 | 1.016 | 2.225 | 5.515 |
| post-thrust(48h) | +PAF prev | 0.895 | 1.152 | 2.526 | 5.981 |
| post-thrust(48h) | +PAF-lite(T1-aware) +MTF | 0.761 | 0.994 | 2.198 | 5.422 |
| post-thrust(48h) | +PAF-lite(blind) +MTF | 0.763 | 0.997 | 2.200 | 5.374 |
| post-thrust(48h) | A0 raw prev | 0.748 | 0.986 | 2.166 | 5.316 |
| post-thrust(48h) | raw MTF | 0.759 | 0.992 | 2.203 | 5.464 |
| post-thrust(48h) | raw MTF +T2 | 0.752 | 0.989 | 2.190 | 5.415 |

## S1 per-satellite win rate vs A0
| h | method | n_sat | win_rate_vs_A0 | median_rel_change_pct |
|---|---|---|---|---|
| 1.000 | +PAF +MTF | 17 | 0.059 | 9.389 |
| 1.000 | +PAF +MTF +T2 | 17 | 0.059 | 7.202 |
| 1.000 | +PAF +MTF +T2 +T1 | 17 | 0.000 | 7.280 |
| 1.000 | +PAF +MTF +gate | 17 | 0.235 | 1.201 |
| 1.000 | +PAF prev | 17 | 0.000 | 7.583 |
| 1.000 | +PAF-lite(T1-aware) +MTF | 17 | 0.176 | 0.380 |
| 1.000 | +PAF-lite(blind) +MTF | 17 | 0.176 | 0.807 |
| 1.000 | raw MTF | 17 | 0.176 | 0.201 |
| 1.000 | raw MTF +T2 | 17 | 0.353 | 0.000 |
| 4.000 | +PAF +MTF | 17 | 0.000 | 11.671 |
| 4.000 | +PAF +MTF +T2 | 17 | 0.000 | 8.022 |
| 4.000 | +PAF +MTF +T2 +T1 | 17 | 0.000 | 3.377 |
| 4.000 | +PAF +MTF +gate | 17 | 0.118 | 4.721 |
| 4.000 | +PAF prev | 17 | 0.000 | 3.130 |
| 4.000 | +PAF-lite(T1-aware) +MTF | 17 | 0.059 | 1.161 |
| 4.000 | +PAF-lite(blind) +MTF | 17 | 0.059 | 1.514 |
| 4.000 | raw MTF | 17 | 0.294 | 1.514 |
| 4.000 | raw MTF +T2 | 17 | 0.235 | 0.393 |
| 12.000 | +PAF +MTF | 17 | 0.176 | 11.249 |
| 12.000 | +PAF +MTF +T2 | 17 | 0.176 | 8.773 |
| 12.000 | +PAF +MTF +T2 +T1 | 17 | 0.176 | 8.817 |
| 12.000 | +PAF +MTF +gate | 17 | 0.235 | 1.885 |
| 12.000 | +PAF prev | 17 | 0.000 | 6.041 |
| 12.000 | +PAF-lite(T1-aware) +MTF | 17 | 0.294 | 0.554 |
| 12.000 | +PAF-lite(blind) +MTF | 17 | 0.353 | 0.629 |
| 12.000 | raw MTF | 17 | 0.353 | 0.554 |
| 12.000 | raw MTF +T2 | 17 | 0.529 | -0.070 |
| 24.000 | +PAF +MTF | 17 | 0.294 | 10.333 |
| 24.000 | +PAF +MTF +T2 | 17 | 0.235 | 6.177 |
| 24.000 | +PAF +MTF +T2 +T1 | 17 | 0.235 | 6.940 |
| 24.000 | +PAF +MTF +gate | 17 | 0.353 | 2.067 |
| 24.000 | +PAF prev | 17 | 0.176 | 4.642 |
| 24.000 | +PAF-lite(T1-aware) +MTF | 17 | 0.235 | 0.012 |
| 24.000 | +PAF-lite(blind) +MTF | 17 | 0.235 | 0.012 |
| 24.000 | raw MTF | 17 | 0.353 | 0.077 |
| 24.000 | raw MTF +T2 | 17 | 0.294 | 0.012 |

## S1 residual correctors: orbit-only vs +T1 (SMA rate) vs +T1+T2 features
| base | corrector | h | n | base_p50_km | corrected_p50_km | base_p90_km | corrected_p90_km | t_abs_p50_base | t_abs_p50_corr |
|---|---|---|---|---|---|---|---|---|---|
| +PAF-lite(T1-aware) +MTF | orbit-only | 1.000 | 4473 | 0.761 | 3.570 | 3.866 | 23.644 | 0.731 | 3.547 |
| +PAF-lite(T1-aware) +MTF | orbit-only | 4.000 | 4473 | 0.994 | 4.494 | 5.235 | 24.438 | 0.970 | 4.475 |
| +PAF-lite(T1-aware) +MTF | orbit-only | 12.000 | 4473 | 2.198 | 6.519 | 11.331 | 38.257 | 2.186 | 6.495 |
| +PAF-lite(T1-aware) +MTF | orbit-only | 24.000 | 4473 | 5.422 | 13.422 | 27.280 | 79.978 | 5.412 | 13.400 |
| +PAF-lite(T1-aware) +MTF | +T1 (SMA rate, phase) | 1.000 | 4473 | 0.761 | 3.726 | 3.866 | 17.674 | 0.731 | 3.696 |
| +PAF-lite(T1-aware) +MTF | +T1 (SMA rate, phase) | 4.000 | 4473 | 0.994 | 4.230 | 5.235 | 18.613 | 0.970 | 4.213 |
| +PAF-lite(T1-aware) +MTF | +T1 (SMA rate, phase) | 12.000 | 4473 | 2.198 | 7.490 | 11.331 | 38.484 | 2.186 | 7.481 |
| +PAF-lite(T1-aware) +MTF | +T1 (SMA rate, phase) | 24.000 | 4473 | 5.422 | 15.639 | 27.280 | 69.703 | 5.412 | 15.619 |
| +PAF-lite(T1-aware) +MTF | +T1 +T2 | 1.000 | 4473 | 0.761 | 2.706 | 3.866 | 15.105 | 0.731 | 2.690 |
| +PAF-lite(T1-aware) +MTF | +T1 +T2 | 4.000 | 4473 | 0.994 | 2.913 | 5.235 | 16.069 | 0.970 | 2.889 |
| +PAF-lite(T1-aware) +MTF | +T1 +T2 | 12.000 | 4473 | 2.198 | 5.545 | 11.331 | 26.262 | 2.186 | 5.543 |
| +PAF-lite(T1-aware) +MTF | +T1 +T2 | 24.000 | 4473 | 5.422 | 14.470 | 27.280 | 68.966 | 5.412 | 14.449 |

## S1 uncertainty on Starlink (along-track): fixed / age / age x g(T2) / operator covariance alone
| product | model | split | n | cov68 | cov95 | nees | sigma_p50_km |
|---|---|---|---|---|---|---|---|
| A0 raw prev | sigma-fixed | all | 17892 | 0.684 | 0.811 | 777.063 | 3.400 |
| A0 raw prev | sigma-fixed | post-thrust(48h) | 17876 | 0.684 | 0.811 | 777.757 | 3.400 |
| A0 raw prev | sigma-age | all | 17892 | 0.496 | 0.641 | 273662197.651 | 2.961 |
| A0 raw prev | sigma-age | post-thrust(48h) | 17876 | 0.495 | 0.641 | 273906984.642 | 2.961 |
| A0 raw prev | sigma-age x g(T2) | all | 17892 | 0.624 | 0.715 | 227434935.811 | 5.079 |
| A0 raw prev | sigma-age x g(T2) | post-thrust(48h) | 17876 | 0.624 | 0.715 | 227638346.869 | 5.080 |
| A0 raw prev | operator covariance (MEME) alone | all | 17892 | 0.062 | 0.117 | 35084961.191 | 0.097 |
| A0 raw prev | operator covariance (MEME) alone | post-thrust(48h) | 17876 | 0.063 | 0.117 | 35116337.579 | 0.097 |
| +PAF +MTF | sigma-fixed | all | 17892 | 0.629 | 0.739 | 56526.357 | 3.816 |
| +PAF +MTF | sigma-fixed | post-thrust(48h) | 17876 | 0.629 | 0.739 | 56576.819 | 3.816 |
| +PAF +MTF | sigma-age | all | 17892 | 0.444 | 0.580 | 2398703.560 | 3.785 |
| +PAF +MTF | sigma-age | post-thrust(48h) | 17876 | 0.444 | 0.580 | 2400691.752 | 3.785 |
| +PAF +MTF | sigma-age x g(T2) | all | 17892 | 0.580 | 0.656 | 431125.417 | 6.620 |
| +PAF +MTF | sigma-age x g(T2) | post-thrust(48h) | 17876 | 0.580 | 0.656 | 431352.519 | 6.618 |
| +PAF +MTF | operator covariance (MEME) alone | all | 17892 | 0.056 | 0.106 | 28107104816.300 | 0.097 |
| +PAF +MTF | operator covariance (MEME) alone | post-thrust(48h) | 17876 | 0.056 | 0.106 | 28132239589.661 | 0.097 |
| +PAF +MTF +T2 | sigma-fixed | all | 17892 | 0.631 | 0.739 | 56247.146 | 3.826 |
| +PAF +MTF +T2 | sigma-fixed | post-thrust(48h) | 17876 | 0.631 | 0.739 | 56297.359 | 3.826 |
| +PAF +MTF +T2 | sigma-age | all | 17892 | 0.443 | 0.580 | 2084470.665 | 3.773 |
| +PAF +MTF +T2 | sigma-age | post-thrust(48h) | 17876 | 0.443 | 0.580 | 2086176.005 | 3.772 |
| +PAF +MTF +T2 | sigma-age x g(T2) | all | 17892 | 0.579 | 0.656 | 394867.567 | 6.568 |
| +PAF +MTF +T2 | sigma-age x g(T2) | post-thrust(48h) | 17876 | 0.579 | 0.656 | 395060.620 | 6.567 |
| +PAF +MTF +T2 | operator covariance (MEME) alone | all | 17892 | 0.056 | 0.106 | 28107095240.709 | 0.097 |
| +PAF +MTF +T2 | operator covariance (MEME) alone | post-thrust(48h) | 17876 | 0.056 | 0.106 | 28132230005.500 | 0.097 |

## S1 T2 detector vs MEME-derived thrust arcs
| pool | k | n_intervals | n_thrust | precision | recall | thrust_fraction |
|---|---|---|---|---|---|---|
| step6 | 2.000 | 1320 | 1083 | 0.844 | 0.370 | 0.820 |

## S2 Starlink conjunction warning: A0 vs A1 trajectories, true Pc from MEME covariance
| lead_h | chain | n_pred | n_matched | precision | recall_ref_lt50km | recall_ref_lt10km | recall_truePc_ge_1e6 | miss_err_med_km | spearman_pcorb_vs_truepc | spearman_pointmiss_vs_truepc | spearman_pcfixed_vs_truepc | recall_at_k_pc_orb | recall_at_k_pc_fixed | recall_at_k_point |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 6 | A0 raw prev | 581 | 9 | 0.015 | 0.067 | 0.000 |  | 16.918 |  |  |  |  |  |  |
| 6 | A1 PAF+MTF+T2 | 625 | 11 | 0.018 | 0.082 | 0.000 |  | 14.701 |  |  |  |  |  |  |
| 24 | A0 raw prev | 549 | 10 | 0.018 | 0.075 | 0.000 |  | 10.146 |  |  |  |  |  |  |
| 24 | A1 PAF+MTF+T2 | 591 | 13 | 0.022 | 0.097 | 0.000 |  | 12.002 |  |  |  |  |  |  |

## S3 Starlink avoidance on matched events (safety by true Pc)
| policy | n_events | safety_rate_true_pc | safety_rate_2km | dv_mean_m_s | dv_p90_m_s |
|---|---|---|---|---|---|
| No-maneuver | 13 | 1.000 | 1.000 | 0.000 | 0.000 |
| Distance rule (2 km) | 13 | 1.000 | 1.000 | 0.000 | 0.000 |
| Pc-gate OrbBench sigma | 13 | 1.000 | 1.000 | 0.000 | 0.000 |
| Pc-gate fixed sigma | 13 | 1.000 | 1.000 | 0.000 | 0.000 |
| Pc-gate operator covariance (oracle) | 13 | 1.000 | 1.000 | 0.000 | 0.000 |

## S4 data ablation on Starlink (1B)
| h | config | p50 | p90 |
|---|---|---|---|
| 4.000 | B5 TLE-only: raw prev, fixed sigma | 0.986 | 5.160 |
| 4.000 | PAF alone (official step6 pool, no gate) | 1.152 | 38.670 |
| 4.000 | B1 no maneuver awareness: PAF+MTF (no gate, no T1/T2) | 1.166 | 39.780 |
| 4.000 | B0 full: PAF+MTF + availability gate | 1.016 | 5.440 |
| 4.000 | B0' full: T1-aware PAF-lite + MTF | 0.994 | 5.240 |
| 24.000 | B5 TLE-only: raw prev, fixed sigma | 5.316 | 27.210 |
| 24.000 | PAF alone (official step6 pool, no gate) | 5.981 | 110.990 |
| 24.000 | B1 no maneuver awareness: PAF+MTF (no gate, no T1/T2) | 6.009 | 113.040 |
| 24.000 | B0 full: PAF+MTF + availability gate | 5.515 | 27.310 |
| 24.000 | B0' full: T1-aware PAF-lite + MTF | 5.422 | 27.280 |

## E2E FULL CHAIN T1->T7 on 1,500 LEO objects (300-2000 km), 3 d, 5 decision epochs, 20% screening budget: independent (A0) vs unified pipeline (A1); reference = Hermite-interpolated L2 trajectories; 30 injected dangerous conjunctions per epoch
| task | metric | A0 | A1 | note |
|---|---|---|---|---|
| T1 Phase | Macro-F1 vs rule labels (cohort, 3 seeds) | 0.666 | 0.666 | A1: + T2 maneuver-flag features |
| T2 Maneuver | event F1 / F2 on Sentinel logs (test 2024-25H1) | 0.433 / 0.538 | 0.443 / 0.543 | A1: threshold conditioned on T1 phase and T3 Ap |
| T3 Weather | MAE Ap / F10.7, 1-3 d | 6.6 / 3.2 | 6.6 / 3.2 | no upstream |
| T4 Forecast | 3D P50 / P95 at 24 h (km) vs L2 reference | 0.11 / 3.09 | 0.11 / 3.09 | reference = non-causal L2 product of the same TLEs (TLE-update consistency) |
| T4 Uncertainty | along-track 95% coverage / median NEES at 24 h | 0.92 / 0.04 | 0.93 / 0.03 | A0 fixed sigma; A1 sigma-age x g(T1,T2,T3) |
| T5 Evolution | TV of +24 h density vs reference | 0.016 | 0.016 | from the respective forecast |
| T6 Warning | recall of reference approaches < 10 km at 20% budget (all pairs) | 0.196 (0.961) | 0.375 | A0 random order; A1 T5 order |
| T6 Warning | alerts / precision vs reference Pc >= 1e-4 | 45 / 0.05 | 0 / nan | A0 alerts on predicted miss < 5 km; A1 on predicted Pc >= 1e-4 |
| T7 Avoidance | maneuvers / unnecessary (ref Pc < 1e-4) / total dv (m/s) | 45 / 43 / 45.9 | 0 / 0 / 0.0 | A0 distance rule; A1 Pc-gate |
| T7 Avoidance | natural dangerous events (ref Pc >= 1e-4): total / made safe / missed | 5 / 2 / 3 | 5 / 0 / 5 | reference geometry and covariance after maneuver |
| T6+T7 injected | injected dangerous conjunctions: total / in screening budget / alerted / safe after | 123 / 0.22 / 26 / 25 | 123 / 0.39 / 45 / 45 | 30/origin, ref miss 20-1000 m, same forecast error |
| T6+T7 injected | unnecessary maneuvers on injected non-dangerous / dv (m/s) | 4 / 414.0 | 0 / 274.8 | nan |
| T7 Avoidance | close events (ref miss < 1 km): total / separated / missed | 5 / 0 / 5 | 5 / 0 / 5 | distance view of the same events |
| Cost | screening time per origin (s) / pairs | 10.7 / 77132 (all: 50.8 / 385661) | 11.6 / 77132 | nan |

## E2E per-task removal on the same cohort and the same injected conjunctions
| config | T4_p50_24h | T4_cov95_t_24h | T4_nees_24h | T6_recall_close | T6_recall_dangerous_pc | T6_n_alerts | T6_alert_precision_pc | T7_n_maneuvers | T7_unnecessary_pc | T7_dv_total_m_s | INJ_n_dangerous | INJ_in_budget | INJ_alerted | INJ_safe_after | INJ_missed | INJ_unnecessary | INJ_dv_m_s | time_screen_s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 chained | 0.112 | 0.932 | 0.030 | 0.375 | 0.125 | 0.000 |  | 0.000 | 0.000 | 0.000 | 123.000 | 0.394 | 45.000 | 45.000 | 78.000 | 0.000 | 274.800 | 57.848 |
| A1 -T1 phase | 0.112 | 0.931 | 0.030 | 0.375 | 0.125 | 0.000 |  | 0.000 | 0.000 | 0.000 | 123.000 | 0.394 | 45.000 | 45.000 | 78.000 | 0.000 | 274.800 | 59.500 |
| A1 -T2 maneuver | 0.112 | 0.928 | 0.036 | 0.375 | 0.125 | 0.000 |  | 0.000 | 0.000 | 0.000 | 123.000 | 0.394 | 45.000 | 44.000 | 78.000 | 0.000 | 251.700 | 61.817 |
| A1 -T3 weather | 0.112 | 0.932 | 0.030 | 0.375 | 0.125 | 0.000 |  | 0.000 | 0.000 | 0.000 | 123.000 | 0.394 | 45.000 | 45.000 | 78.000 | 0.000 | 274.800 | 61.333 |
| A1 -T4 sigma (fixed) | 0.112 | 0.924 | 0.043 | 0.375 | 0.125 | 0.000 |  | 0.000 | 0.000 | 0.000 | 123.000 | 0.394 | 44.000 | 44.000 | 79.000 | 0.000 | 250.800 | 62.306 |
| A1 -T5 order (random) | 0.112 | 0.932 | 0.030 | 0.196 | 0.625 | 2.000 | 1.000 | 2.000 | 0.000 | 6.600 | 123.000 | 0.221 | 25.000 | 25.000 | 98.000 | 0.000 | 150.300 | 51.910 |
| A1 -T6 Pc (distance rule) | 0.112 | 0.932 | 0.030 | 0.375 | 0.125 | 108.000 | 0.000 | 108.000 | 108.000 | 42.900 | 123.000 | 0.394 | 47.000 | 46.000 | 76.000 | 8.000 | 760.500 | 60.019 |

## UC upstream connections T1->T2 and T3->T2 on Sentinel (maneuver logs as truth, test 2024-25H1)
| variant | precision | recall | f1 | f2 |
|---|---|---|---|---|
| HistGB (A0) | 0.275 | 0.140 | 0.185 | 0.155 |
| HistGB + T1 + T3 (A1) | 0.321 | 0.154 | 0.208 | 0.172 |
| SGP4-residual (A0) | 0.326 | 0.643 | 0.433 | 0.538 |
| SGP4-residual + T1 phase | 0.326 | 0.643 | 0.433 | 0.538 |
| SGP4-residual + T1 phase + T3 Ap (A1) | 0.339 | 0.640 | 0.443 | 0.543 |
| SGP4-residual + T3 Ap | 0.339 | 0.640 | 0.443 | 0.543 |

## UC upstream connection T1->T2 on Starlink (MEME thrust arcs as truth; 84% of intervals are thrust, so F1 is degenerate and MCC is reported)
| variant | k_stable | k_raise | k_lower | n_test | n_thrust | tp | fp | fn | tn | precision | recall | f1 | balanced_acc | mcc | coast_precision | coast_recall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| always thrust (trivial) |  |  |  | 1683 | 1413 | 1413 | 270 | 0 | 0 | 0.840 | 1.000 | 0.913 | 0.500 | 0.000 | 0.000 | 0.000 |
| SGP4-residual (A0) | 3.000 | 3.000 | 3.000 | 1683 | 1413 | 444 | 74 | 969 | 196 | 0.857 | 0.314 | 0.460 | 0.520 | 0.032 | 0.168 | 0.726 |
| SGP4-residual, threshold per T1 phase | 3.000 | 3.000 | 3.000 | 1683 | 1413 | 444 | 74 | 969 | 196 | 0.857 | 0.314 | 0.460 | 0.520 | 0.032 | 0.168 | 0.726 |
| T1 phase alone (non-stable = thrust) |  |  |  | 1683 | 1413 | 1281 | 230 | 132 | 40 | 0.848 | 0.907 | 0.876 | 0.527 | 0.066 | 0.233 | 0.148 |
| SGP4-residual AND T1 phase | 3.000 | 3.000 | 3.000 | 1683 | 1413 | 411 | 56 | 1002 | 214 | 0.880 | 0.291 | 0.437 | 0.542 | 0.068 | 0.176 | 0.793 |
| SGP4-residual OR T1 phase (A1) | 3.000 | 3.000 | 3.000 | 1683 | 1413 | 1314 | 248 | 99 | 22 | 0.841 | 0.930 | 0.883 | 0.506 | 0.016 | 0.182 | 0.081 |
|   phase=stable: A0 | 3.000 | 3.000 | 3.000 | 172 | 132 | 33 | 18 | 99 | 22 | 0.647 | 0.250 | 0.361 | 0.400 | -0.185 | 0.182 | 0.550 |
|   phase=stable: A1 | 3.000 | 3.000 | 3.000 | 172 | 132 | 33 | 18 | 99 | 22 | 0.647 | 0.250 | 0.361 | 0.400 | -0.185 | 0.182 | 0.550 |
|   phase=lower: A0 | 3.000 | 3.000 | 3.000 | 1505 | 1277 | 408 | 56 | 869 | 172 | 0.879 | 0.319 | 0.469 | 0.537 | 0.057 | 0.165 | 0.754 |
|   phase=lower: A1 | 3.000 | 3.000 | 3.000 | 1505 | 1277 | 408 | 56 | 869 | 172 | 0.879 | 0.319 | 0.469 | 0.537 | 0.057 | 0.165 | 0.754 |

## UC upstream connection T2->T1 on the cohort (rule labels, 3 object-level seeds)
| variant | macro_f1 | f1_STABLE_ORBIT | f1_SUSTAINED_RAISE | f1_SUSTAINED_DECAY |
|---|---|---|---|---|
| HistGB (A0) | 0.666 | 1.000 | 0.000 | 0.998 |
| HistGB + T2 flags (A1) | 0.666 | 1.000 | 0.000 | 0.999 |

## E51 A0 vs A1 and per-task ablations with paired bootstrap 95% CIs (X1 events; X2/X4 clustered by forecast origin; X3 by approach)
| example | variant | metric | value_ref | value_variant | diff | ci_lo | ci_hi | significant |
|---|---|---|---|---|---|---|---|---|
| X1 CDM warning->avoidance (199 events) | -T3 (no space weather) | total delta-v | 457.800 | 340.500 | -117.300 | -255.900 | -0.900 | True |
| X1 CDM warning->avoidance (199 events) | -T3 (no space weather) | safety rate (true Pc<1e-4) | 0.920 | 0.915 | -0.005 | -0.025 | 0.015 | False |
| X1 CDM warning->avoidance (199 events) | -T5 (no object/orbit attributes) | safety rate (true Pc<1e-4) | 0.920 | 0.905 | -0.015 | -0.040 | 0.010 | False |
| X1 CDM warning->avoidance (199 events) | -T3 -T5 (age/OD only) | safety rate (true Pc<1e-4) | 0.920 | 0.894 | -0.025 | -0.055 | 0.005 | False |
| X1 CDM warning->avoidance (199 events) | -T4 (fixed sigma, no uncertainty task) | unnecessary maneuvers | 16.000 | 25.000 | 9.000 | 2.000 | 17.000 | True |
| X1 CDM warning->avoidance (199 events) | -T4 (fixed sigma, no uncertainty task) | total delta-v | 457.800 | 816.300 | 358.500 | 124.155 | 580.822 | True |
| X1 CDM warning->avoidance (199 events) | -T4 (fixed sigma, no uncertainty task) | safety rate (true Pc<1e-4) | 0.920 | 0.940 | 0.020 | -0.010 | 0.055 | False |
| X1 CDM warning->avoidance (199 events) | -T6 (distance rule, ignores Pc) | missed dangerous events | 12.000 | 0.000 | -12.000 | -19.000 | -6.000 | True |
| X1 CDM warning->avoidance (199 events) | -T6 (distance rule, ignores Pc) | unnecessary maneuvers | 16.000 | 61.000 | 45.000 | 34.000 | 57.000 | True |
| X1 CDM warning->avoidance (199 events) | -T6 (distance rule, ignores Pc) | total delta-v | 457.800 | 1209.600 | 751.800 | 511.485 | 984.900 | True |
| X1 CDM warning->avoidance (199 events) | -T6 (distance rule, ignores Pc) | safety rate (true Pc<1e-4) | 0.920 | 0.990 | 0.070 | 0.035 | 0.111 | True |
| X1 CDM warning->avoidance (199 events) | A0 none (fixed sigma + distance rule) | missed dangerous events | 12.000 | 0.000 | -12.000 | -19.000 | -6.000 | True |
| X1 CDM warning->avoidance (199 events) | A0 none (fixed sigma + distance rule) | unnecessary maneuvers | 16.000 | 61.000 | 45.000 | 34.000 | 57.000 | True |
| X1 CDM warning->avoidance (199 events) | A0 none (fixed sigma + distance rule) | total delta-v | 457.800 | 1209.600 | 751.800 | 511.485 | 984.900 | True |
| X1 CDM warning->avoidance (199 events) | A0 none (fixed sigma + distance rule) | safety rate (true Pc<1e-4) | 0.920 | 0.990 | 0.070 | 0.035 | 0.111 | True |
| X2 Starlink triage (4 h) | -T3 | AUC | 0.733 | 0.710 | -0.023 | -0.030 | -0.016 | True |
| X2 Starlink triage (4 h) | -T3 | recall at 30% flag budget | 0.569 | 0.544 | -0.025 | -0.045 | -0.003 | True |
| X2 Starlink triage (4 h) | -T2 | AUC | 0.733 | 0.731 | -0.002 | -0.011 | 0.007 | False |
| X2 Starlink triage (4 h) | -T2 | recall at 30% flag budget | 0.569 | 0.593 | 0.025 | -0.001 | 0.051 | False |
| X2 Starlink triage (4 h) | -T1 | AUC | 0.733 | 0.704 | -0.029 | -0.047 | -0.014 | True |
| X2 Starlink triage (4 h) | -T1 | recall at 30% flag budget | 0.569 | 0.537 | -0.032 | -0.061 | 0.000 | False |
| X2 Starlink triage (4 h) | -T1 -T2 -T3 (A0: age only) | AUC | 0.733 | 0.718 | -0.015 | -0.037 | 0.006 | False |
| X2 Starlink triage (4 h) | -T1 -T2 -T3 (A0: age only) | recall at 30% flag budget | 0.569 | 0.558 | -0.011 | -0.048 | 0.025 | False |
| X2 Starlink triage (24 h) | -T3 | AUC | 0.643 | 0.651 | 0.007 | -0.002 | 0.017 | False |
| X2 Starlink triage (24 h) | -T3 | recall at 30% flag budget | 0.428 | 0.451 | 0.023 | 0.001 | 0.045 | True |
| X2 Starlink triage (24 h) | -T2 | AUC | 0.643 | 0.654 | 0.011 | 0.005 | 0.018 | True |
| X2 Starlink triage (24 h) | -T2 | recall at 30% flag budget | 0.428 | 0.451 | 0.023 | 0.009 | 0.044 | True |
| X2 Starlink triage (24 h) | -T1 | AUC | 0.643 | 0.584 | -0.059 | -0.078 | -0.038 | True |
| X2 Starlink triage (24 h) | -T1 | recall at 30% flag budget | 0.428 | 0.361 | -0.068 | -0.091 | -0.038 | True |
| X2 Starlink triage (24 h) | -T1 -T2 -T3 (A0: age only) | AUC | 0.643 | 0.602 | -0.041 | -0.062 | -0.021 | True |
| X2 Starlink triage (24 h) | -T1 -T2 -T3 (A0: age only) | recall at 30% flag budget | 0.428 | 0.369 | -0.060 | -0.085 | -0.031 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, post-maneuver) | A1 with perfect T2 (logged maneuvers) | T7 safety on dangerous conjunctions | 0.649 | 0.703 | 0.053 | 0.041 | 0.067 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, post-maneuver) | A1 with perfect T2 (logged maneuvers) | T6 alert recall | 0.679 | 0.707 | 0.029 | 0.019 | 0.039 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, post-maneuver) | A1 with perfect T2 (logged maneuvers) | total delta-v | 5302.200 | 6911.700 | 1609.500 | 1477.680 | 1749.915 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, post-maneuver) | -T2 (sigma-age, maneuver-blind) | T7 safety on dangerous conjunctions | 0.649 | 0.595 | -0.055 | -0.068 | -0.041 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, post-maneuver) | -T2 (sigma-age, maneuver-blind) | T6 alert recall | 0.679 | 0.637 | -0.042 | -0.054 | -0.030 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, post-maneuver) | -T2 (sigma-age, maneuver-blind) | total delta-v | 5302.200 | 4554.600 | -747.600 | -858.908 | -634.463 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, post-maneuver) | -T4-UQ (fixed sigma) | T7 safety on dangerous conjunctions | 0.649 | 0.663 | 0.014 | 0.005 | 0.023 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, post-maneuver) | -T4-UQ (fixed sigma) | T6 alert recall | 0.679 | 0.681 | 0.003 | -0.004 | 0.009 | False |
| X4 Sentinel T2->UQ->T6->T7 (4 h, post-maneuver) | -T4-UQ (fixed sigma) | total delta-v | 5302.200 | 5606.700 | 304.500 | 225.892 | 379.808 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, post-maneuver) | -T6 (distance rule on predicted miss) | T7 safety on dangerous conjunctions | 0.649 | 0.871 | 0.222 | 0.196 | 0.248 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, post-maneuver) | -T6 (distance rule on predicted miss) | T6 alert recall | 0.679 | 0.679 | 0.000 | 0.000 | 0.000 | False |
| X4 Sentinel T2->UQ->T6->T7 (4 h, post-maneuver) | -T6 (distance rule on predicted miss) | total delta-v | 5302.200 | 21049.200 | 15747.000 | 15119.573 | 16330.283 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, post-maneuver) | -T6 (distance rule on predicted miss) | unnecessary maneuvers | 128.000 | 884.000 | 756.000 | 707.000 | 804.025 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, post-maneuver) | A0 none (fixed sigma + distance rule) | T7 safety on dangerous conjunctions | 0.649 | 0.871 | 0.222 | 0.196 | 0.248 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, post-maneuver) | A0 none (fixed sigma + distance rule) | T6 alert recall | 0.679 | 0.681 | 0.003 | -0.004 | 0.009 | False |
| X4 Sentinel T2->UQ->T6->T7 (4 h, post-maneuver) | A0 none (fixed sigma + distance rule) | total delta-v | 5302.200 | 21049.200 | 15747.000 | 15119.573 | 16330.283 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, post-maneuver) | A0 none (fixed sigma + distance rule) | unnecessary maneuvers | 128.000 | 884.000 | 756.000 | 707.000 | 804.025 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, all) | A1 with perfect T2 (logged maneuvers) | T7 safety on dangerous conjunctions | 0.816 | 0.820 | 0.005 | -0.001 | 0.010 | False |
| X4 Sentinel T2->UQ->T6->T7 (4 h, all) | A1 with perfect T2 (logged maneuvers) | T6 alert recall | 0.830 | 0.829 | -0.000 | -0.005 | 0.004 | False |
| X4 Sentinel T2->UQ->T6->T7 (4 h, all) | A1 with perfect T2 (logged maneuvers) | total delta-v | 22557.000 | 22037.100 | -519.900 | -759.375 | -264.900 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, all) | -T2 (sigma-age, maneuver-blind) | T7 safety on dangerous conjunctions | 0.816 | 0.787 | -0.028 | -0.034 | -0.022 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, all) | -T2 (sigma-age, maneuver-blind) | T6 alert recall | 0.830 | 0.808 | -0.022 | -0.027 | -0.017 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, all) | -T2 (sigma-age, maneuver-blind) | total delta-v | 22557.000 | 19680.000 | -2877.000 | -3064.582 | -2698.133 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, all) | -T4-UQ (fixed sigma) | T7 safety on dangerous conjunctions | 0.816 | 0.831 | 0.015 | 0.010 | 0.020 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, all) | -T4-UQ (fixed sigma) | T6 alert recall | 0.830 | 0.838 | 0.009 | 0.005 | 0.012 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, all) | -T4-UQ (fixed sigma) | total delta-v | 22557.000 | 23513.400 | 956.400 | 806.588 | 1104.068 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, all) | -T6 (distance rule on predicted miss) | T7 safety on dangerous conjunctions | 0.816 | 0.957 | 0.141 | 0.130 | 0.153 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, all) | -T6 (distance rule on predicted miss) | T6 alert recall | 0.830 | 0.830 | 0.000 | 0.000 | 0.000 | False |
| X4 Sentinel T2->UQ->T6->T7 (4 h, all) | -T6 (distance rule on predicted miss) | total delta-v | 22557.000 | 78691.800 | 56134.800 | 55133.078 | 57136.223 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, all) | -T6 (distance rule on predicted miss) | unnecessary maneuvers | 417.000 | 3095.000 | 2678.000 | 2585.975 | 2764.000 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, all) | A0 none (fixed sigma + distance rule) | T7 safety on dangerous conjunctions | 0.816 | 0.957 | 0.141 | 0.130 | 0.153 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, all) | A0 none (fixed sigma + distance rule) | T6 alert recall | 0.830 | 0.838 | 0.009 | 0.005 | 0.012 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, all) | A0 none (fixed sigma + distance rule) | total delta-v | 22557.000 | 78691.800 | 56134.800 | 55133.078 | 57136.223 | True |
| X4 Sentinel T2->UQ->T6->T7 (4 h, all) | A0 none (fixed sigma + distance rule) | unnecessary maneuvers | 417.000 | 3095.000 | 2678.000 | 2585.975 | 2764.000 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, post-maneuver) | A1 with perfect T2 (logged maneuvers) | T7 safety on dangerous conjunctions | 0.670 | 0.691 | 0.021 | 0.010 | 0.035 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, post-maneuver) | A1 with perfect T2 (logged maneuvers) | T6 alert recall | 0.684 | 0.695 | 0.011 | 0.001 | 0.020 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, post-maneuver) | A1 with perfect T2 (logged maneuvers) | total delta-v | 6310.500 | 7959.900 | 1649.400 | 1490.663 | 1814.115 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, post-maneuver) | -T2 (sigma-age, maneuver-blind) | T7 safety on dangerous conjunctions | 0.670 | 0.664 | -0.005 | -0.014 | 0.003 | False |
| X4 Sentinel T2->UQ->T6->T7 (24 h, post-maneuver) | -T2 (sigma-age, maneuver-blind) | T6 alert recall | 0.684 | 0.674 | -0.010 | -0.018 | -0.003 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, post-maneuver) | -T2 (sigma-age, maneuver-blind) | total delta-v | 6310.500 | 5584.800 | -725.700 | -847.837 | -612.840 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, post-maneuver) | -T4-UQ (fixed sigma) | T7 safety on dangerous conjunctions | 0.670 | 0.666 | -0.003 | -0.009 | 0.003 | False |
| X4 Sentinel T2->UQ->T6->T7 (24 h, post-maneuver) | -T4-UQ (fixed sigma) | T6 alert recall | 0.684 | 0.679 | -0.005 | -0.011 | 0.001 | False |
| X4 Sentinel T2->UQ->T6->T7 (24 h, post-maneuver) | -T4-UQ (fixed sigma) | total delta-v | 6310.500 | 5751.600 | -558.900 | -659.437 | -464.700 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, post-maneuver) | -T6 (distance rule on predicted miss) | T7 safety on dangerous conjunctions | 0.670 | 0.790 | 0.120 | 0.098 | 0.141 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, post-maneuver) | -T6 (distance rule on predicted miss) | T6 alert recall | 0.684 | 0.684 | 0.000 | 0.000 | 0.000 | False |
| X4 Sentinel T2->UQ->T6->T7 (24 h, post-maneuver) | -T6 (distance rule on predicted miss) | total delta-v | 6310.500 | 19856.700 | 13546.200 | 12907.725 | 14148.390 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, post-maneuver) | -T6 (distance rule on predicted miss) | unnecessary maneuvers | 112.000 | 780.000 | 668.000 | 617.975 | 722.000 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, post-maneuver) | A0 none (fixed sigma + distance rule) | T7 safety on dangerous conjunctions | 0.670 | 0.790 | 0.120 | 0.098 | 0.141 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, post-maneuver) | A0 none (fixed sigma + distance rule) | T6 alert recall | 0.684 | 0.679 | -0.005 | -0.011 | 0.001 | False |
| X4 Sentinel T2->UQ->T6->T7 (24 h, post-maneuver) | A0 none (fixed sigma + distance rule) | total delta-v | 6310.500 | 19856.700 | 13546.200 | 12907.725 | 14148.390 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, post-maneuver) | A0 none (fixed sigma + distance rule) | unnecessary maneuvers | 112.000 | 780.000 | 668.000 | 617.975 | 722.000 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, all) | A1 with perfect T2 (logged maneuvers) | T7 safety on dangerous conjunctions | 0.761 | 0.761 | 0.001 | -0.005 | 0.006 | False |
| X4 Sentinel T2->UQ->T6->T7 (24 h, all) | A1 with perfect T2 (logged maneuvers) | T6 alert recall | 0.774 | 0.774 | 0.000 | -0.004 | 0.005 | False |
| X4 Sentinel T2->UQ->T6->T7 (24 h, all) | A1 with perfect T2 (logged maneuvers) | total delta-v | 23610.900 | 23242.800 | -368.100 | -608.130 | -97.627 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, all) | -T2 (sigma-age, maneuver-blind) | T7 safety on dangerous conjunctions | 0.761 | 0.753 | -0.008 | -0.012 | -0.003 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, all) | -T2 (sigma-age, maneuver-blind) | T6 alert recall | 0.774 | 0.768 | -0.006 | -0.010 | -0.002 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, all) | -T2 (sigma-age, maneuver-blind) | total delta-v | 23610.900 | 20867.700 | -2743.200 | -2940.315 | -2547.563 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, all) | -T2 (sigma-age, maneuver-blind) | unnecessary maneuvers | 418.000 | 400.000 | -18.000 | -29.000 | -7.000 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, all) | -T4-UQ (fixed sigma) | T7 safety on dangerous conjunctions | 0.761 | 0.747 | -0.014 | -0.019 | -0.010 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, all) | -T4-UQ (fixed sigma) | T6 alert recall | 0.774 | 0.770 | -0.004 | -0.007 | -0.001 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, all) | -T4-UQ (fixed sigma) | total delta-v | 23610.900 | 21279.300 | -2331.600 | -2499.383 | -2166.278 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, all) | -T4-UQ (fixed sigma) | unnecessary maneuvers | 418.000 | 395.000 | -23.000 | -33.000 | -12.000 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, all) | -T6 (distance rule on predicted miss) | T7 safety on dangerous conjunctions | 0.761 | 0.888 | 0.127 | 0.116 | 0.139 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, all) | -T6 (distance rule on predicted miss) | T6 alert recall | 0.774 | 0.774 | 0.000 | 0.000 | 0.000 | False |
| X4 Sentinel T2->UQ->T6->T7 (24 h, all) | -T6 (distance rule on predicted miss) | total delta-v | 23610.900 | 73244.100 | 49633.200 | 48519.210 | 50660.355 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, all) | -T6 (distance rule on predicted miss) | unnecessary maneuvers | 418.000 | 2868.000 | 2450.000 | 2359.000 | 2550.025 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, all) | A0 none (fixed sigma + distance rule) | T7 safety on dangerous conjunctions | 0.761 | 0.888 | 0.127 | 0.116 | 0.139 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, all) | A0 none (fixed sigma + distance rule) | T6 alert recall | 0.774 | 0.770 | -0.004 | -0.007 | -0.001 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, all) | A0 none (fixed sigma + distance rule) | total delta-v | 23610.900 | 73244.100 | 49633.200 | 48519.210 | 50660.355 | True |
| X4 Sentinel T2->UQ->T6->T7 (24 h, all) | A0 none (fixed sigma + distance rule) | unnecessary maneuvers | 418.000 | 2868.000 | 2450.000 | 2359.000 | 2550.025 | True |
| X3 LEO evolution->warning (139 approaches, 20% budget) | random order (A0) | recall of reference approaches | 0.352 | 0.121 | -0.231 | -0.319 | -0.143 | True |

## E52 Screening efficiency: all candidate pairs vs T5-ordered 20% budget (600 LEO objects, 3 d, 1-min grid)
| setting | n_objects | n_pairs | screening_time_s | recall |
|---|---|---|---|---|
| A0: all candidate pairs | 600.000 | 74766.000 | 27.700 | 1.000 |
| A1: T5-ordered, 20% budget | 600.000 | 14953.000 | 6.500 | 0.352 |
| A0: random 20% budget | 600.000 | 14953.000 | 6.500 | 0.121 |

## E53 Data ablation B0-B5 on decision-level metrics (assembled from X1/X2/X4, C1, T4, S4, X3)
| config | data_removed | example | metric | B0_full | ablated | rel_change_pct |
|---|---|---|---|---|---|---|
| B2 No-weather | space-weather indices | X1 CDM warning->avoidance | missed dangerous events (of 56) | 12.000 | 14.000 | 16.667 |
| B2 No-weather | space-weather indices | X1 CDM warning->avoidance | unnecessary maneuvers | 16.000 | 16.000 | 0.000 |
| B2 No-weather | space-weather indices | X1 CDM warning->avoidance | total delta-v (m/s) | 457.800 | 340.500 | -25.623 |
| B3 No-physical | object type, size, area-to-mass, orbit attributes | X1 CDM warning->avoidance | missed dangerous events (of 56) | 12.000 | 14.000 | 16.667 |
| B3 No-physical | object type, size, area-to-mass, orbit attributes | X1 CDM warning->avoidance | unnecessary maneuvers | 16.000 | 14.000 | -12.500 |
| B3 No-physical | object type, size, area-to-mass, orbit attributes | X1 CDM warning->avoidance | total delta-v (m/s) | 457.800 | 383.400 | -16.252 |
| B4 No-precise | operator covariances for sigma calibration | X1 CDM warning->avoidance | missed dangerous events (of 56) | 12.000 | 10.000 | -16.667 |
| B4 No-precise | operator covariances for sigma calibration | X1 CDM warning->avoidance | unnecessary maneuvers | 16.000 | 25.000 | 56.250 |
| B4 No-precise | operator covariances for sigma calibration | X1 CDM warning->avoidance | total delta-v (m/s) | 457.800 | 816.300 | 78.309 |
| B5 TLE-only | all auxiliary data | X1 CDM warning->avoidance | missed dangerous events (of 56) | 12.000 | 0.000 | -100.000 |
| B5 TLE-only | all auxiliary data | X1 CDM warning->avoidance | unnecessary maneuvers | 16.000 | 61.000 | 281.250 |
| B5 TLE-only | all auxiliary data | X1 CDM warning->avoidance | total delta-v (m/s) | 457.800 | 1209.600 | 164.220 |
| B2 No-weather | space-weather indices | X2 Starlink triage (4 h) | AUC | 0.733 | 0.710 | -3.114 |
| B1 No-maneuver | maneuver/phase information (T1, T2 inputs) | X2 Starlink triage (4 h) | AUC | 0.733 | 0.718 | -2.078 |
| B2 No-weather | space-weather indices | X2 Starlink triage (24 h) | AUC | 0.643 | 0.651 | 1.147 |
| B1 No-maneuver | maneuver/phase information (T1, T2 inputs) | X2 Starlink triage (24 h) | AUC | 0.643 | 0.602 | -6.374 |
| B0 with maneuver logs | (adds logged maneuvers to sigma) | X4 Sentinel post-maneuver (4 h) | T7 safety on dangerous conjunctions | 0.703 | 0.649 | -7.588 |
| B1 No-maneuver | maneuver awareness (detector and logs) | X4 Sentinel post-maneuver (4 h) | T7 safety on dangerous conjunctions | 0.649 | 0.595 | -8.421 |
| B1 No-maneuver | maneuver logs | T4-UQ Sentinel post-maneuver (24 h) | along-track 95% coverage | 0.796 | 0.747 | -6.145 |
| B1 No-maneuver | maneuver awareness | T4-UQ Sentinel post-maneuver (24 h) | along-track 95% coverage | 0.796 | 0.717 | -9.916 |
| B4 No-precise | POD for corrector training | T4 Sentinel forecast (1 h) | 3D P50 (km) | 0.311 | 0.571 | 83.852 |
| B4 No-precise | POD for corrector training | T4 Sentinel forecast (4 h) | 3D P50 (km) | 0.381 | 0.495 | 30.135 |
| B4 No-precise | POD for corrector training | T4 Sentinel forecast (24 h) | 3D P50 (km) | 0.525 | 0.457 | -12.992 |
| B4 No-precise | POD for sigma calibration | T4-UQ Sentinel (along-track) | NEES | 4.070 | 5.468 | 34.344 |
| B1 No-maneuver | maneuver awareness (gate / T1) | S4 Starlink forecast (4 h) | 3D P90 (km) | 5.440 | 39.780 | 631.250 |
| B5 TLE-only | all auxiliary data | S4 Starlink forecast (4 h) | 3D P90 (km) | 5.440 | 5.160 | -5.147 |
| B1 No-maneuver | maneuver awareness (gate / T1) | S4 Starlink forecast (24 h) | 3D P90 (km) | 27.310 | 113.040 | 313.914 |
| B5 TLE-only | all auxiliary data | S4 Starlink forecast (24 h) | 3D P90 (km) | 27.310 | 27.210 | -0.366 |
| B5 TLE-only | all auxiliary data | X3 LEO screening (20% budget) | recall of reference approaches | 0.383 | 0.136 | -64.516 |
| B5 TLE-only | all auxiliary data | X3 LEO screening (20% budget) | total delta-v (m/s) | 6.090 | 6.480 | 6.404 |

## E54a Same task on an existing dataset: T2 maneuver detection on the Shorten et al. TLE benchmark (15 satellites, curated maneuver timestamps)
| dataset | satellite | pool | setting | n_maneuvers | n_pred | precision | recall | f1 | f2 |
|---|---|---|---|---|---|---|---|---|---|
| Shorten TLE benchmark | POOLED-LEO | raw | transfer (Sentinel k) | 688.000 | 1853.000 | 0.263 | 0.708 | 0.383 | 0.529 |
| Shorten TLE benchmark | POOLED-ALL | raw | transfer (Sentinel k) | 887.000 | 2547.000 | 0.263 | 0.754 | 0.390 | 0.549 |
| Shorten TLE benchmark | POOLED-LEO | raw | best k on this dataset | 688.000 | 1396.000 | 0.323 | 0.656 | 0.433 | 0.544 |
| Shorten TLE benchmark | POOLED-ALL | raw | best k on this dataset | 887.000 | 1631.000 | 0.378 | 0.696 | 0.490 | 0.596 |
| Shorten TLE benchmark | POOLED-LEO | PAF-lite | transfer (Sentinel k) | 688.000 | 1693.000 | 0.308 | 0.759 | 0.438 | 0.587 |
| Shorten TLE benchmark | POOLED-ALL | PAF-lite | transfer (Sentinel k) | 887.000 | 2324.000 | 0.303 | 0.795 | 0.439 | 0.600 |
| Shorten TLE benchmark | POOLED-LEO | PAF-lite | best k on this dataset | 688.000 | 1045.000 | 0.457 | 0.695 | 0.552 | 0.629 |
| Shorten TLE benchmark | POOLED-ALL | PAF-lite | best k on this dataset | 887.000 | 1269.000 | 0.513 | 0.734 | 0.604 | 0.676 |
| Shorten TLE benchmark | POOLED-GEO | raw | transfer (Sentinel k) | 199.000 | 694.000 | 0.262 | 0.915 | 0.408 | 0.611 |
| Shorten TLE benchmark | POOLED-GEO | raw | best k on this dataset | 199.000 | 235.000 | 0.706 | 0.834 | 0.765 | 0.805 |
| Shorten TLE benchmark | POOLED-GEO | PAF-lite | transfer (Sentinel k) | 199.000 | 631.000 | 0.290 | 0.920 | 0.441 | 0.641 |
| Shorten TLE benchmark | POOLED-GEO | PAF-lite | best k on this dataset | 199.000 | 224.000 | 0.772 | 0.869 | 0.818 | 0.848 |
| OrbBench (Sentinel POD logs) | S1A | PAF | tuned on train years | 195.000 | 331.000 | 0.326 | 0.554 | 0.411 | 0.486 |
| OrbBench (Sentinel POD logs) | S2A | PAF | tuned on train years | 38.000 | 99.000 | 0.323 | 0.842 | 0.467 | 0.637 |
| OrbBench (Sentinel POD logs) | S3A | PAF | tuned on train years | 39.000 | 107.000 | 0.327 | 0.897 | 0.479 | 0.665 |
| OrbBench (Sentinel POD logs) | ALL | PAF | tuned on train years |  |  | 0.326 | 0.643 | 0.433 | 0.538 |

## E54b Same task on an existing dataset: T6 final-risk prediction on the full ESA Kelvins CDM dataset (competition protocol, train -> public test with private truth)
| dataset | method | mse_hr | f2 | precision_hr | recall_hr | score | n_events | n_high_risk |
|---|---|---|---|---|---|---|---|---|
| Kelvins public test (2167 events, competition protocol) | naive latest risk (>= 2 d before TCA) | 26.643 | 0.739 | 0.646 | 0.767 | 36.049 | 2167 | 150 |
| Kelvins test, OrbBench 199-event subset (n=199) | naive latest risk (>= 2 d before TCA) | 26.643 | 0.799 | 0.958 | 0.767 | 33.362 | 199 | 150 |
| Kelvins public test (2167 events, competition protocol) | Ridge signed-log, all events | 131.988 | 0.000 | 0.000 | 0.000 | 131988213600.936 | 2167 | 150 |
| Kelvins test, OrbBench 199-event subset (n=199) | Ridge signed-log, all events | 131.988 | 0.000 | 0.000 | 0.000 | 131988213600.936 | 199 | 150 |
| Kelvins public test (2167 events, competition protocol) | HistGB, all events | 157.471 | 0.008 | 0.500 | 0.007 | 18959.539 | 2167 | 150 |
| Kelvins test, OrbBench 199-event subset (n=199) | HistGB, all events | 157.471 | 0.008 | 1.000 | 0.007 | 18928.044 | 199 | 150 |
| Kelvins public test (2167 events, competition protocol) | Ridge signed-log, high-risk-focused (pre-risk >= -8, else naive) | 146.254 | 0.000 | 0.000 | 0.000 | 146253651316.665 | 2167 | 150 |
| Kelvins test, OrbBench 199-event subset (n=199) | Ridge signed-log, high-risk-focused (pre-risk >= -8, else naive) | 146.254 | 0.000 | 0.000 | 0.000 | 146253651316.665 | 199 | 150 |
| Kelvins public test (2167 events, competition protocol) | HistGB, high-risk-focused (pre-risk >= -8, else naive) | 145.122 | 0.000 | 0.000 | 0.000 | 145121926349.662 | 2167 | 150 |
| Kelvins test, OrbBench 199-event subset (n=199) | HistGB, high-risk-focused (pre-risk >= -8, else naive) | 145.122 | 0.000 | 0.000 | 0.000 | 145121926349.662 | 199 | 150 |
| Kelvins public test (2167 events, competition protocol) | naive + HistGB residual, high-risk-focused | 145.743 | 0.008 | 0.500 | 0.007 | 17547.399 | 2167 | 150 |
| Kelvins test, OrbBench 199-event subset (n=199) | naive + HistGB residual, high-risk-focused | 145.743 | 0.008 | 1.000 | 0.007 | 17518.251 | 199 | 150 |
| Kelvins public test (2167 events, competition protocol) | HistGB, geometry + risk-history features only (28), high-risk-focused | 160.729 | 0.000 | 0.000 | 0.000 | 160729244808.610 | 2167 | 150 |
| Kelvins test, OrbBench 199-event subset (n=199) | HistGB, geometry + risk-history features only (28), high-risk-focused | 160.729 | 0.000 | 0.000 | 0.000 | 160729244808.610 | 199 | 150 |
| Kelvins public test (2167 events, competition protocol) | naive + HistGB residual, geometry features, high-risk-focused | 160.928 | 0.000 | 0.000 | 0.000 | 160927618907.191 | 2167 | 150 |
| Kelvins test, OrbBench 199-event subset (n=199) | naive + HistGB residual, geometry features, high-risk-focused | 160.928 | 0.000 | 0.000 | 0.000 | 160927618907.191 | 199 | 150 |

## X4 TASK-LEVEL CHAIN: T2 maneuver detection -> T4 uncertainty -> T6 Pc alert -> T7 Pc-gated avoidance (Sentinel-1A/2A/3A POD truth, synthetic chasers, test 2024-25H1)
| split | chain | n_conj | n_dangerous | t6_alert_recall | t6_false_alert_rate | t7_safety_dangerous | t7_missed_dangerous | t7_unnecessary | dv_total_m_s |
|---|---|---|---|---|---|---|---|---|---|
| all | A1 full (T2 detector -> T4 sigma -> T6 Pc alert -> T7 Pc-gate) | 11652 | 4836 | 0.774 | 0.061 | 0.761 | 1093 | 418 | 23610.900 |
| post-maneuver (logged, 48 h) | A1 full (T2 detector -> T4 sigma -> T6 Pc alert -> T7 Pc-gate) | 3600 | 1511 | 0.684 | 0.054 | 0.670 | 477 | 112 | 6310.500 |
| no recent maneuver | A1 full (T2 detector -> T4 sigma -> T6 Pc alert -> T7 Pc-gate) | 8052 | 3325 | 0.815 | 0.065 | 0.802 | 616 | 306 | 17300.400 |
| all | A1 with perfect T2 (logged maneuvers) | 11652 | 4836 | 0.774 | 0.060 | 0.761 | 1092 | 410 | 23242.800 |
| post-maneuver (logged, 48 h) | A1 with perfect T2 (logged maneuvers) | 3600 | 1511 | 0.695 | 0.056 | 0.691 | 461 | 116 | 7959.900 |
| no recent maneuver | A1 with perfect T2 (logged maneuvers) | 8052 | 3325 | 0.810 | 0.062 | 0.793 | 631 | 294 | 15282.900 |
| all |   -T2 (sigma-age, maneuver-blind) | 11652 | 4836 | 0.768 | 0.059 | 0.753 | 1123 | 400 | 20867.700 |
| post-maneuver (logged, 48 h) |   -T2 (sigma-age, maneuver-blind) | 3600 | 1511 | 0.674 | 0.051 | 0.664 | 492 | 106 | 5584.800 |
| no recent maneuver |   -T2 (sigma-age, maneuver-blind) | 8052 | 3325 | 0.810 | 0.062 | 0.793 | 631 | 294 | 15282.900 |
| all |   -T4-UQ (fixed sigma) | 11652 | 4836 | 0.770 | 0.058 | 0.747 | 1113 | 395 | 21279.300 |
| post-maneuver (logged, 48 h) |   -T4-UQ (fixed sigma) | 3600 | 1511 | 0.679 | 0.052 | 0.666 | 485 | 109 | 5751.600 |
| no recent maneuver |   -T4-UQ (fixed sigma) | 8052 | 3325 | 0.811 | 0.061 | 0.783 | 628 | 286 | 15527.700 |
| all |   -T6 (distance rule on predicted miss) | 11652 | 4836 | 0.774 | 0.061 | 0.888 | 491 | 2868 | 73244.100 |
| post-maneuver (logged, 48 h) |   -T6 (distance rule on predicted miss) | 3600 | 1511 | 0.684 | 0.054 | 0.790 | 302 | 780 | 19856.700 |
| no recent maneuver |   -T6 (distance rule on predicted miss) | 8052 | 3325 | 0.815 | 0.065 | 0.933 | 189 | 2088 | 53387.400 |
| all | A0 none (fixed sigma + distance rule) | 11652 | 4836 | 0.770 | 0.058 | 0.888 | 491 | 2868 | 73244.100 |
| post-maneuver (logged, 48 h) | A0 none (fixed sigma + distance rule) | 3600 | 1511 | 0.679 | 0.052 | 0.790 | 302 | 780 | 19856.700 |
| no recent maneuver | A0 none (fixed sigma + distance rule) | 8052 | 3325 | 0.811 | 0.061 | 0.933 | 189 | 2088 | 53387.400 |
| all | perfect T4 forecast (no error) | 11652 | 4836 | 1.000 | 0.002 | 1.000 | 0 | 11 | 15567.600 |
| post-maneuver (logged, 48 h) | perfect T4 forecast (no error) | 3600 | 1511 | 1.000 | 0.000 | 1.000 | 0 | 1 | 4896.000 |
| no recent maneuver | perfect T4 forecast (no error) | 8052 | 3325 | 1.000 | 0.002 | 1.000 | 0 | 10 | 10671.600 |
| all | no maneuver | 11652 | 4836 | 0.770 | 0.058 | 0.000 | 4836 | 0 | 0.000 |
| post-maneuver (logged, 48 h) | no maneuver | 3600 | 1511 | 0.679 | 0.052 | 0.000 | 1511 | 0 | 0.000 |
| no recent maneuver | no maneuver | 8052 | 3325 | 0.811 | 0.061 | 0.000 | 3325 | 0 | 0.000 |

## X3 TASK-LEVEL CHAIN: T5 evolution ordering -> T6 screening under a 20% pair budget -> T7 (600 LEO objects, 3 d)
| chain | budget_fraction | n_reference_approaches | approaches_reached | recall | dv_per_event_m_s | dv_total_m_s | events_with_pc_ge_gate | dv_per_reached_dangerous |
|---|---|---|---|---|---|---|---|---|
| A1 full (T5 order -> T6 Pc -> T7 Pc-gate) | 0.200 | 182 | 69.700 | 0.383 | 0.087 | 6.090 | 2 |  |
|   -T5 (random order) | 0.200 | 182 | 24.700 | 0.136 | 0.087 | 2.160 | 2 |  |
|   -T6 sigma (distance rule) | 0.200 | 182 | 69.700 | 0.383 | 0.262 | 18.260 | 2 |  |
| A0 none (random order + distance rule) | 0.200 | 182 | 24.700 | 0.136 | 0.262 | 6.480 | 2 |  |
| oracle ordering + Pc-gate | 0.200 | 182 | 182.000 | 1.000 | 0.087 | 15.900 | 2 |  |

## X2 TASK-LEVEL CHAIN: T1 phase + T2 score + T3 weather -> T4 forecast-reliability -> T6 screening triage (17 Starlinks, MEME truth)
| h | threshold_km | chain | n_test | auc | avg_precision | recall_at_30pct_flag | precision_at_30pct_flag |
|---|---|---|---|---|---|---|---|
| 4.000 | 3.000 | A1 full (T1+T2+T3 -> T4 reliability -> T6 triage) | 4473 | 0.733 | 0.414 | 0.569 | 0.362 |
| 4.000 | 3.000 |   -T3 | 4473 | 0.710 | 0.385 | 0.544 | 0.346 |
| 4.000 | 3.000 |   -T2 | 4473 | 0.731 | 0.398 | 0.593 | 0.377 |
| 4.000 | 3.000 |   -T1 | 4473 | 0.704 | 0.374 | 0.537 | 0.342 |
| 4.000 | 3.000 |   -T1 -T2 -T3 (A0: age only) | 4473 | 0.718 | 0.394 | 0.558 | 0.355 |
| 24.000 | 10.000 | A1 full (T1+T2+T3 -> T4 reliability -> T6 triage) | 4473 | 0.643 | 0.456 | 0.428 | 0.453 |
| 24.000 | 10.000 |   -T3 | 4473 | 0.651 | 0.457 | 0.451 | 0.477 |
| 24.000 | 10.000 |   -T2 | 4473 | 0.654 | 0.480 | 0.451 | 0.477 |
| 24.000 | 10.000 |   -T1 | 4473 | 0.584 | 0.401 | 0.361 | 0.382 |
| 24.000 | 10.000 |   -T1 -T2 -T3 (A0: age only) | 4473 | 0.602 | 0.400 | 0.369 | 0.390 |

## X1 TASK-LEVEL CHAIN: T3 weather + T5 object attributes -> T4 chaser-sigma model -> T6 Pc -> T7 Pc-gate (199 ESA CDMs, operator covariance as truth)
| chain | n_events | n_dangerous | safety_all | missed_dangerous | unnecessary_maneuvers | dv_total_m_s | sigma_t_within_x2 |
|---|---|---|---|---|---|---|---|
| A1 full (T3+T5+T4 -> T6 Pc -> T7 Pc-gate) | 199 | 56 | 0.920 | 12 | 16 | 457.800 | 0.417 |
|   -T3 (no space weather) | 199 | 56 | 0.915 | 14 | 16 | 340.500 | 0.447 |
|   -T5 (no object/orbit attributes) | 199 | 56 | 0.905 | 14 | 14 | 383.400 | 0.487 |
|   -T3 -T5 (age/OD only) | 199 | 56 | 0.894 | 17 | 16 | 310.800 | 0.432 |
|   -T4 (fixed sigma, no uncertainty task) | 199 | 56 | 0.940 | 10 | 25 | 816.300 | 0.291 |
|   -T6 (distance rule, ignores Pc) | 199 | 56 | 0.990 | 0 | 61 | 1209.600 | 0.417 |
| A0 none (fixed sigma + distance rule) | 199 | 56 | 0.990 | 0 | 61 | 1209.600 | 0.291 |
| oracle (operator per-event covariance) | 199 | 56 | 0.995 | 0 | 0 | 227.400 | 1.000 |
| no maneuver | 199 | 56 | 0.719 | 56 | 0 | 0.000 | 0.291 |

## X1 ranking quality of the T6 Pc alert (dangerous = true Pc >= 1e-4)
| chain | spearman_pcpred_vs_pctrue | recall_at_k | recall_at_2k | auc_dangerous |
|---|---|---|---|---|
| A1 full (T3+T5+T4 -> T6 Pc -> T7 Pc-gate) | 0.707 | 0.732 | 0.911 | 0.875 |
|   -T3 (no space weather) | 0.700 | 0.750 | 0.893 | 0.869 |
|   -T5 (no object/orbit attributes) | 0.736 | 0.750 | 0.911 | 0.880 |
|   -T3 -T5 (age/OD only) | 0.706 | 0.714 | 0.929 | 0.882 |
|   -T4 (fixed sigma, no uncertainty task) | 0.700 | 0.607 | 0.982 | 0.880 |
| oracle (operator per-event covariance) | 1.000 | 1.000 | 1.000 | 1.000 |
| point miss (no T4/T6 uncertainty) | 0.751 | 0.732 | 1.000 | 0.918 |

## CHAIN A0 (independent) vs A1 (chained): where each edge pays off
| stage | metric | A0 | A0_value | A1 | A1_value |
|---|---|---|---|---|---|
| Trajectory product (T4) | kinematic jump at TLE switch, P90 (m) | nearest TLE | 137.5 | MTF | 13.7 |
| Trajectory product (T4) | share of switches with jump > 50 m | nearest TLE | 0.252 | MTF | 0.01 |
| Uncertainty (T2 -> UQ), post-maneuver (logged) | along-track 95% cov / NEES | sigma-age only | 0.780 / 10.28 | sigma-age x g(T2 score) | 0.816 / 7.63 |
| Uncertainty (events -> UQ), post-maneuver (logged) | along-track 95% cov / NEES | sigma-age only | 0.780 / 10.28 | sigma-age x g(T2 + maneuver log) | 0.867 / 3.92 |
| Uncertainty (T2 -> UQ), all | along-track 95% cov / NEES | sigma-age only | 0.885 / 4.80 | sigma-age x g(T2 score) | 0.915 / 3.57 |
| Uncertainty (events -> UQ), all | along-track 95% cov / NEES | sigma-age only | 0.885 / 4.80 | sigma-age x g(T2 + maneuver log) | 0.931 / 2.42 |
| Warning ranking (UQ -> T6), lead 6 h | recall@K of true approaches | point miss | 0.167 | Pc(sigma-age x T2) | 0.167 |
| Warning ranking (UQ -> T6), lead 24 h | recall@K of true approaches | point miss | 0.155 | Pc(sigma-age x T2) | 0.167 |
| Screening budget (T5 -> T6), 20% of pairs | recall of reference approaches | random order | 0.136 | T5 shell density (min) | 0.383 |
| Screening budget (T5 -> T6), 50% of pairs | recall of reference approaches | random order | 0.488 | T5 shell density (min) | 0.71 |
| Screening budget (T5 -> T6) | AUC of recall-budget curve | random order | 0.494 | T5 shell density (min) | 0.651 |
| Avoidance (UQ -> T7), 117 CDM events | mean delta-v (m/s) [safety by true Pc<1e-4] | distance rule (2 km) | 10.34 [1.00] | Pc-gate, per-event calibrated sigma | 0.30 [1.00] |
| Avoidance (UQ -> T7), 117 CDM events | mean delta-v (m/s) [safety by true Pc<1e-4] | Pc-gate, one fixed sigma | 0.00 [0.75] | Pc-gate, per-event calibrated sigma | 0.30 [1.00] |

## C1 T2 -> UQ: maneuver-aware sigma (Sentinel POD truth, test 2024-25H1)
| model | split | n | cov68 | cov95 | nees | err_p50_km | sigma_p50_km |
|---|---|---|---|---|---|---|---|
| sigma-age | all | 17478 | 0.569 | 0.885 | 4.805 | 0.453 | 0.522 |
| sigma-age | post-maneuver (logged) | 5400 | 0.531 | 0.780 | 10.279 | 0.494 | 0.523 |
| sigma-age | no recent maneuver | 12078 | 0.586 | 0.932 | 2.357 | 0.440 | 0.520 |
| sigma-age | T2 score>=6 | 8826 | 0.574 | 0.891 | 4.488 | 0.452 | 0.520 |
| sigma-age | T2 score<3 | 5358 | 0.582 | 0.888 | 5.474 | 0.438 | 0.523 |
| sigma-age x g(T2 score) | all | 17478 | 0.658 | 0.915 | 3.569 | 0.453 | 0.612 |
| sigma-age x g(T2 score) | post-maneuver (logged) | 5400 | 0.594 | 0.816 | 7.628 | 0.494 | 0.607 |
| sigma-age x g(T2 score) | no recent maneuver | 12078 | 0.686 | 0.960 | 1.754 | 0.440 | 0.614 |
| sigma-age x g(T2 score) | T2 score>=6 | 8826 | 0.689 | 0.926 | 3.125 | 0.452 | 0.618 |
| sigma-age x g(T2 score) | T2 score<3 | 5358 | 0.616 | 0.902 | 4.686 | 0.438 | 0.566 |
| sigma-age x g(logged maneuver) | all | 17478 | 0.626 | 0.912 | 2.839 | 0.453 | 0.546 |
| sigma-age x g(logged maneuver) | post-maneuver (logged) | 5400 | 0.714 | 0.867 | 3.917 | 0.494 | 0.848 |
| sigma-age x g(logged maneuver) | no recent maneuver | 12078 | 0.586 | 0.932 | 2.357 | 0.440 | 0.520 |
| sigma-age x g(logged maneuver) | T2 score>=6 | 8826 | 0.657 | 0.914 | 2.744 | 0.452 | 0.590 |
| sigma-age x g(logged maneuver) | T2 score<3 | 5358 | 0.611 | 0.921 | 3.136 | 0.438 | 0.537 |
| sigma-age x g(both) | all | 17478 | 0.694 | 0.931 | 2.422 | 0.453 | 0.674 |
| sigma-age x g(both) | post-maneuver (logged) | 5400 | 0.714 | 0.867 | 3.917 | 0.494 | 0.848 |
| sigma-age x g(both) | no recent maneuver | 12078 | 0.686 | 0.960 | 1.754 | 0.440 | 0.614 |
| sigma-age x g(both) | T2 score>=6 | 8826 | 0.742 | 0.937 | 2.254 | 0.452 | 0.795 |
| sigma-age x g(both) | T2 score<3 | 5358 | 0.641 | 0.928 | 2.892 | 0.438 | 0.580 |

## C2 UQ -> T6: ranking predicted approaches by Pc vs point miss
| lead_h | ranking | target | n_pred | n_positive | recall_at_k | recall_at_2k | precision_at_10 | spearman_score_vs_neg_truemiss |
|---|---|---|---|---|---|---|---|---|
| 6 | point miss | matched (true approach < 10 km) | 1306 | 84 | 0.167 | 0.286 | 0.200 | 0.052 |
| 6 | point miss | dangerous (true miss < 5 km) | 1306 | 16 | 0.062 | 0.125 | 0.000 | 0.052 |
| 6 | Pc sigma-fixed | matched (true approach < 10 km) | 1306 | 84 | 0.167 | 0.298 | 0.200 | 0.053 |
| 6 | Pc sigma-fixed | dangerous (true miss < 5 km) | 1306 | 16 | 0.062 | 0.125 | 0.000 | 0.053 |
| 6 | Pc sigma-age | matched (true approach < 10 km) | 1306 | 84 | 0.167 | 0.298 | 0.200 | -0.030 |
| 6 | Pc sigma-age | dangerous (true miss < 5 km) | 1306 | 16 | 0.000 | 0.188 | 0.000 | -0.030 |
| 6 | Pc sigma-age x T2 | matched (true approach < 10 km) | 1306 | 84 | 0.167 | 0.298 | 0.200 | -0.030 |
| 6 | Pc sigma-age x T2 | dangerous (true miss < 5 km) | 1306 | 16 | 0.000 | 0.188 | 0.000 | -0.030 |
| 6 | Pc sigma-object (LOO p50) | matched (true approach < 10 km) | 1306 | 84 | 0.095 | 0.167 | 0.400 | -0.101 |
| 6 | Pc sigma-object (LOO p50) | dangerous (true miss < 5 km) | 1306 | 16 | 0.062 | 0.062 | 0.100 | -0.101 |
| 6 | Pc sigma-object (LOO p90) | matched (true approach < 10 km) | 1306 | 84 | 0.119 | 0.167 | 0.300 | -0.124 |
| 6 | Pc sigma-object (LOO p90) | dangerous (true miss < 5 km) | 1306 | 16 | 0.062 | 0.125 | 0.100 | -0.124 |
| 6 | miss / sigma-object | matched (true approach < 10 km) | 1306 | 84 | 0.095 | 0.167 | 0.300 | -0.118 |
| 6 | miss / sigma-object | dangerous (true miss < 5 km) | 1306 | 16 | 0.000 | 0.062 | 0.000 | -0.118 |
| 24 | point miss | matched (true approach < 10 km) | 1305 | 84 | 0.155 | 0.333 | 0.200 | 0.046 |
| 24 | point miss | dangerous (true miss < 5 km) | 1305 | 16 | 0.000 | 0.125 | 0.000 | 0.046 |
| 24 | Pc sigma-fixed | matched (true approach < 10 km) | 1305 | 84 | 0.167 | 0.333 | 0.200 | 0.046 |
| 24 | Pc sigma-fixed | dangerous (true miss < 5 km) | 1305 | 16 | 0.000 | 0.188 | 0.000 | 0.046 |
| 24 | Pc sigma-age | matched (true approach < 10 km) | 1305 | 84 | 0.167 | 0.286 | 0.200 | -0.028 |
| 24 | Pc sigma-age | dangerous (true miss < 5 km) | 1305 | 16 | 0.000 | 0.125 | 0.000 | -0.028 |
| 24 | Pc sigma-age x T2 | matched (true approach < 10 km) | 1305 | 84 | 0.167 | 0.286 | 0.200 | -0.028 |
| 24 | Pc sigma-age x T2 | dangerous (true miss < 5 km) | 1305 | 16 | 0.000 | 0.125 | 0.000 | -0.028 |
| 24 | Pc sigma-object (LOO p50) | matched (true approach < 10 km) | 1305 | 84 | 0.083 | 0.167 | 0.300 | -0.059 |
| 24 | Pc sigma-object (LOO p50) | dangerous (true miss < 5 km) | 1305 | 16 | 0.000 | 0.000 | 0.000 | -0.059 |
| 24 | Pc sigma-object (LOO p90) | matched (true approach < 10 km) | 1305 | 84 | 0.119 | 0.167 | 0.100 | -0.081 |
| 24 | Pc sigma-object (LOO p90) | dangerous (true miss < 5 km) | 1305 | 16 | 0.062 | 0.062 | 0.000 | -0.081 |
| 24 | miss / sigma-object | matched (true approach < 10 km) | 1305 | 84 | 0.095 | 0.167 | 0.100 | -0.067 |
| 24 | miss / sigma-object | dangerous (true miss < 5 km) | 1305 | 16 | 0.000 | 0.000 | 0.000 | -0.067 |

## C3 UQ -> T7: avoidance delta-v vs decision sigma (117 CDM events, true Pc criterion)
| policy | n_events | pre_pc_ge_1e4_frac | safety_rate_pc | safety_rate_2km | dv_mean_m_s | dv_p90_m_s | min_dist_p50_m | pc_post_p90 | decision_sigma_p50_m | true_sigma_p50_m |
|---|---|---|---|---|---|---|---|---|---|---|
| No-maneuver | 117 | 0.248 | 0.752 | 0.000 | 0.000 | 0.000 | 470.712 | 0.001 |  | 725.714 |
| Distance-max (2 km) | 117 | 0.248 | 1.000 | 1.000 | 10.338 | 15.000 | 2016.671 | 0.000 |  | 725.714 |
| Pc-gate, calibrated per-event sigma (CDM) | 117 | 0.248 | 1.000 | 0.000 | 0.303 | 1.200 | 501.996 | 0.000 | 725.714 | 725.714 |
| Pc-gate, per-event sigma x0.5 (overconfident) | 117 | 0.248 | 0.752 | 0.000 | 0.244 | 0.900 | 507.519 | 0.000 | 362.857 | 725.714 |
| Pc-gate, per-event sigma x2 (underconfident) | 117 | 0.248 | 0.915 | 0.000 | 0.326 | 1.500 | 501.333 | 0.000 | 1451.429 | 725.714 |
| Pc-gate, per-event sigma x4 | 117 | 0.248 | 0.846 | 0.000 | 0.226 | 0.480 | 490.844 | 0.000 | 2902.858 | 725.714 |
| Pc-gate, one fixed sigma (median of events) | 117 | 0.248 | 0.752 | 0.000 | 0.000 | 0.000 | 470.712 | 0.001 | 725.714 | 725.714 |
| Pc-gate, OrbBench TLE sigma-age (24 h, not event-specific) | 117 | 0.248 | 0.752 | 0.000 | 0.000 | 0.000 | 470.712 | 0.001 | 648.522 | 725.714 |
| Pc-gate, conservative 2 km sigma | 117 | 0.248 | 0.752 | 0.000 | 0.000 | 0.000 | 470.712 | 0.001 | 2000.000 | 725.714 |

## C4 T5 -> T6: recall of reference approaches vs screening budget (600 LEO objects)
| ordering | 0.02 | 0.05 | 0.1 | 0.15 | 0.2 | 0.3 | 0.4 | 0.5 | 0.75 | 1.0 | AUC | n_truth | n_pairs | n_objects |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| T5 shell density (min) | 0.074 | 0.111 | 0.204 | 0.302 | 0.383 | 0.512 | 0.593 | 0.710 | 0.889 | 1.000 | 0.651 | 182 | 74766 | 600 |
| T5 shell density (product) | 0.074 | 0.105 | 0.160 | 0.241 | 0.364 | 0.463 | 0.574 | 0.648 | 0.889 | 1.000 | 0.622 | 182 | 74766 | 600 |
| oracle | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 1.000 | 0.999 | 182 | 74766 | 600 |
| random | 0.025 | 0.025 | 0.049 | 0.099 | 0.136 | 0.296 | 0.395 | 0.488 | 0.759 | 1.000 | 0.494 | 182 | 74766 | 600 |
| same-shell first | 0.056 | 0.099 | 0.284 | 0.352 | 0.432 | 0.586 | 0.667 | 0.778 | 0.932 | 1.000 | 0.694 | 182 | 74766 | 600 |

## P1 Sentinel TLE thinning with POD truth (3D P50/P90 km by kept fraction and method)
| split | method | 0.1 | 0.2 | 0.33 | 0.5 | 1.0 |
|---|---|---|---|---|---|---|
| all | MTF | 0.583 | 0.535 | 0.523 | 0.517 | 0.514 |
| all | MTF-adaptive | 0.726 | 0.580 | 0.538 | 0.523 | 0.515 |
| all | MTF-adaptive+T2-gate | 0.731 | 0.596 | 0.562 | 0.542 | 0.528 |
| all | MTF-adaptive+oracle-gate | 0.723 | 0.598 | 0.552 | 0.532 | 0.519 |
| all | MTF-wide | 0.584 | 0.536 | 0.525 | 0.518 | 0.514 |
| all | blend-2 | 0.696 | 0.569 | 0.537 | 0.524 | 0.517 |
| all | nearest | 0.583 | 0.535 | 0.524 | 0.518 | 0.516 |
| maneuver+-24h | MTF | 0.650 | 0.579 | 0.552 | 0.542 | 0.534 |
| maneuver+-24h | MTF-adaptive | 0.829 | 0.632 | 0.571 | 0.550 | 0.532 |
| maneuver+-24h | MTF-adaptive+T2-gate | 0.861 | 0.673 | 0.625 | 0.600 | 0.570 |
| maneuver+-24h | MTF-adaptive+oracle-gate | 0.940 | 0.706 | 0.625 | 0.580 | 0.541 |
| maneuver+-24h | MTF-wide | 0.651 | 0.581 | 0.554 | 0.543 | 0.531 |
| maneuver+-24h | blend-2 | 0.764 | 0.612 | 0.565 | 0.548 | 0.535 |
| maneuver+-24h | nearest | 0.650 | 0.579 | 0.554 | 0.544 | 0.540 |
| quiet | MTF | 0.569 | 0.526 | 0.517 | 0.512 | 0.510 |
| quiet | MTF-adaptive | 0.707 | 0.569 | 0.530 | 0.517 | 0.511 |
| quiet | MTF-adaptive+T2-gate | 0.707 | 0.580 | 0.548 | 0.530 | 0.519 |
| quiet | MTF-adaptive+oracle-gate | 0.669 | 0.562 | 0.527 | 0.515 | 0.509 |
| quiet | MTF-wide | 0.570 | 0.526 | 0.518 | 0.513 | 0.510 |
| quiet | blend-2 | 0.683 | 0.560 | 0.530 | 0.518 | 0.513 |
| quiet | nearest | 0.569 | 0.526 | 0.517 | 0.512 | 0.510 |

## P1 relative change vs nearest (day-block bootstrap 95% CI)
| keep | method | nearest_p50 | method_p50 | rel_change_pct | ci_lo | ci_hi | win_rate |
|---|---|---|---|---|---|---|---|
| 0.100 | MTF | 0.583 | 0.583 | -0.009 | -0.027 | 0.006 | 0.156 |
| 0.100 | MTF-adaptive | 0.583 | 0.726 | 24.525 | 23.348 | 26.300 | 0.423 |
| 0.100 | MTF-adaptive+T2-gate | 0.583 | 0.731 | 25.310 | 23.846 | 26.679 | 0.397 |
| 0.100 | MTF-adaptive+oracle-gate | 0.583 | 0.723 | 24.077 | 21.856 | 25.740 | 0.252 |
| 0.100 | MTF-wide | 0.583 | 0.584 | 0.154 | 0.060 | 0.260 | 0.267 |
| 0.100 | blend-2 | 0.583 | 0.696 | 19.402 | 18.246 | 20.562 | 0.457 |
| 0.200 | MTF | 0.535 | 0.535 | -0.013 | -0.053 | 0.012 | 0.206 |
| 0.200 | MTF-adaptive | 0.535 | 0.580 | 8.419 | 7.676 | 9.219 | 0.478 |
| 0.200 | MTF-adaptive+T2-gate | 0.535 | 0.596 | 11.406 | 10.412 | 12.429 | 0.449 |
| 0.200 | MTF-adaptive+oracle-gate | 0.535 | 0.598 | 11.771 | 10.664 | 12.896 | 0.294 |
| 0.200 | MTF-wide | 0.535 | 0.536 | 0.158 | 0.037 | 0.294 | 0.406 |
| 0.200 | blend-2 | 0.535 | 0.569 | 6.332 | 5.704 | 7.022 | 0.507 |
| 0.330 | MTF | 0.524 | 0.523 | -0.115 | -0.168 | -0.060 | 0.287 |
| 0.330 | MTF-adaptive | 0.524 | 0.538 | 2.602 | 2.222 | 3.026 | 0.505 |
| 0.330 | MTF-adaptive+T2-gate | 0.524 | 0.562 | 7.200 | 6.515 | 8.033 | 0.470 |
| 0.330 | MTF-adaptive+oracle-gate | 0.524 | 0.552 | 5.238 | 4.532 | 5.981 | 0.316 |
| 0.330 | MTF-wide | 0.524 | 0.525 | 0.113 | -0.058 | 0.293 | 0.487 |
| 0.330 | blend-2 | 0.524 | 0.537 | 2.431 | 2.009 | 2.885 | 0.523 |
| 0.500 | MTF | 0.518 | 0.517 | -0.180 | -0.245 | -0.112 | 0.388 |
| 0.500 | MTF-adaptive | 0.518 | 0.523 | 0.869 | 0.651 | 1.122 | 0.512 |
| 0.500 | MTF-adaptive+T2-gate | 0.518 | 0.542 | 4.573 | 4.007 | 5.252 | 0.484 |
| 0.500 | MTF-adaptive+oracle-gate | 0.518 | 0.532 | 2.719 | 2.180 | 3.261 | 0.327 |
| 0.500 | MTF-wide | 0.518 | 0.518 | 0.003 | -0.201 | 0.177 | 0.510 |
| 0.500 | blend-2 | 0.518 | 0.524 | 1.071 | 0.808 | 1.370 | 0.527 |
| 1.000 | MTF | 0.516 | 0.514 | -0.361 | -0.480 | -0.221 | 0.499 |
| 1.000 | MTF-adaptive | 0.516 | 0.515 | -0.238 | -0.421 | 0.005 | 0.515 |
| 1.000 | MTF-adaptive+T2-gate | 0.516 | 0.528 | 2.384 | 2.021 | 2.873 | 0.492 |
| 1.000 | MTF-adaptive+oracle-gate | 0.516 | 0.519 | 0.465 | 0.054 | 0.879 | 0.334 |
| 1.000 | MTF-wide | 0.516 | 0.514 | -0.365 | -0.542 | -0.153 | 0.517 |
| 1.000 | blend-2 | 0.516 | 0.517 | 0.217 | 0.014 | 0.470 | 0.531 |

## P1 error by local TLE gap (pooled over kept fractions)
| gap_bin | method | n | p50 | p90 |
|---|---|---|---|---|
| <6h | MTF | 62003 | 0.585 | 0.960 |
| <6h | MTF-adaptive | 62003 | 0.585 | 0.960 |
| <6h | MTF-adaptive+T2-gate | 62003 | 0.585 | 0.961 |
| <6h | MTF-adaptive+oracle-gate | 46290 | 0.571 | 0.961 |
| <6h | MTF-wide | 62003 | 0.583 | 0.961 |
| <6h | blend-2 | 62003 | 0.584 | 0.959 |
| <6h | nearest | 62003 | 0.586 | 0.959 |
| 6-12h | MTF | 226481 | 0.512 | 0.927 |
| 6-12h | MTF-adaptive | 226481 | 0.511 | 0.928 |
| 6-12h | MTF-adaptive+T2-gate | 226481 | 0.516 | 0.938 |
| 6-12h | MTF-adaptive+oracle-gate | 158183 | 0.519 | 0.945 |
| 6-12h | MTF-wide | 226481 | 0.511 | 0.929 |
| 6-12h | blend-2 | 226481 | 0.511 | 0.927 |
| 6-12h | nearest | 226481 | 0.513 | 0.928 |
| 12-24h | MTF | 286082 | 0.499 | 0.928 |
| 12-24h | MTF-adaptive | 286082 | 0.501 | 0.939 |
| 12-24h | MTF-adaptive+T2-gate | 286082 | 0.511 | 0.975 |
| 12-24h | MTF-adaptive+oracle-gate | 195831 | 0.510 | 0.968 |
| 12-24h | MTF-wide | 286082 | 0.499 | 0.932 |
| 12-24h | blend-2 | 286082 | 0.500 | 0.935 |
| 12-24h | nearest | 286082 | 0.500 | 0.929 |
| 24-48h | MTF | 310902 | 0.526 | 0.983 |
| 24-48h | MTF-adaptive | 310902 | 0.543 | 1.071 |
| 24-48h | MTF-adaptive+T2-gate | 310902 | 0.570 | 1.454 |
| 24-48h | MTF-adaptive+oracle-gate | 212553 | 0.566 | 1.308 |
| 24-48h | MTF-wide | 310902 | 0.529 | 0.999 |
| 24-48h | blend-2 | 310902 | 0.538 | 1.046 |
| 24-48h | nearest | 310902 | 0.526 | 0.983 |
| 48-96h | MTF | 261519 | 0.537 | 1.043 |
| 48-96h | MTF-adaptive | 261519 | 0.595 | 1.614 |
| 48-96h | MTF-adaptive+T2-gate | 261519 | 0.618 | 3.095 |
| 48-96h | MTF-adaptive+oracle-gate | 167418 | 0.636 | 3.062 |
| 48-96h | MTF-wide | 261519 | 0.540 | 1.058 |
| 48-96h | blend-2 | 261519 | 0.580 | 1.449 |
| 48-96h | nearest | 261519 | 0.537 | 1.043 |
| >96h | MTF | 227426 | 0.625 | 3.482 |
| >96h | MTF-adaptive | 227426 | 0.929 | 7.092 |
| >96h | MTF-adaptive+T2-gate | 227426 | 1.005 | 18.818 |
| >96h | MTF-adaptive+oracle-gate | 131133 | 0.956 | 12.613 |
| >96h | MTF-wide | 227426 | 0.625 | 3.480 |
| >96h | blend-2 | 227426 | 0.880 | 6.588 |
| >96h | nearest | 227426 | 0.625 | 3.482 |

## P2 fault injection with POD truth: raw vs PAF-lite pools
| pool | method | 0.0 | 0.02 | 0.05 | 0.1 |
|---|---|---|---|---|---|
| PAF-lite | MTF | 0.943 | 0.948 | 0.959 | 0.984 |
| PAF-lite | MTF-adaptive | 0.948 | 0.954 | 0.966 | 0.998 |
| PAF-lite | nearest | 0.943 | 0.945 | 0.950 | 0.961 |
| raw | MTF | 0.940 | 0.964 | 1.013 | 1.169 |
| raw | MTF-adaptive | 0.944 | 0.974 | 1.040 | 1.289 |
| raw | nearest | 0.940 | 0.948 | 0.962 | 0.990 |

## P2 PAF-lite detection of injected faults
| p | n_tle | n_fault | fault_removed | clean_removed | recall_n_jump | recall_bstar_neg | recall_duplicate | fault_recall | false_removal_rate |
|---|---|---|---|---|---|---|---|---|---|
| 0.000 | 10640 | 0 | 0 | 140 | 0.000 | 0.000 | 0.000 | 0.000 | 0.013 |
| 0.020 | 32036 | 576 | 442 | 530 | 0.667 | 1.000 | 0.671 | 0.767 | 0.017 |
| 0.050 | 32220 | 1504 | 1137 | 703 | 0.624 | 1.000 | 0.700 | 0.756 | 0.023 |
| 0.100 | 32527 | 3105 | 2283 | 993 | 0.595 | 1.000 | 0.677 | 0.735 | 0.034 |

## P3 cohort: effect of a PAF-removed nearest neighbour on the LOO residual
| scope | n_neighbour_removed | n_neighbour_kept | nearest_raw_p50_when_nb_removed | nearest_PAF_p50_when_nb_removed | PAF_win_rate_when_nb_removed | nearest_raw_p90_when_nb_removed | nearest_PAF_p90_when_nb_removed | nearest_raw_p50_when_nb_kept | MTF_raw_p50_when_nb_removed | MTF_PAF_p50_when_nb_removed |
|---|---|---|---|---|---|---|---|---|---|---|
| ALL | 7205 | 120032 | 0.209 | 0.330 | 0.336 | 0.921 | 1.175 | 0.075 | 0.209 | 0.329 |
| DEB|GEO | 32 | 26 | 0.284 | 0.400 | 0.594 | 2.913 | 1.888 | 0.441 | 0.284 | 0.400 |
| DEB|HEO | 36 | 438 | 0.341 | 0.600 | 0.333 | 1.479 | 2.516 | 0.700 | 0.335 | 0.600 |
| DEB|LEO | 396 | 60254 | 0.163 | 0.246 | 0.298 | 0.961 | 1.306 | 0.088 | 0.160 | 0.245 |
| DEB|MEO | 221 | 2437 | 0.309 | 0.618 | 0.299 | 2.012 | 4.241 | 0.398 | 0.306 | 0.625 |
| PAY|GEO | 2409 | 3696 | 0.278 | 0.405 | 0.353 | 1.198 | 1.309 | 0.411 | 0.277 | 0.405 |
| PAY|HEO | 1072 | 2144 | 0.245 | 0.432 | 0.296 | 0.935 | 1.318 | 0.391 | 0.244 | 0.432 |
| PAY|LEO | 342 | 23642 | 0.035 | 0.049 | 0.260 | 0.110 | 0.148 | 0.030 | 0.033 | 0.049 |
| PAY|MEO | 337 | 5208 | 0.118 | 0.193 | 0.326 | 0.468 | 0.728 | 0.152 | 0.117 | 0.193 |
| R/B|GEO | 1020 | 1279 | 0.171 | 0.265 | 0.368 | 0.651 | 0.690 | 0.258 | 0.175 | 0.265 |
| R/B|HEO | 678 | 1972 | 0.224 | 0.342 | 0.345 | 0.807 | 1.058 | 0.343 | 0.224 | 0.338 |
| R/B|LEO | 120 | 12557 | 0.046 | 0.064 | 0.350 | 0.170 | 0.264 | 0.037 | 0.046 | 0.064 |
| R/B|MEO | 541 | 5782 | 0.221 | 0.339 | 0.348 | 0.829 | 1.052 | 0.227 | 0.224 | 0.334 |

## P3 cohort TLE sparsity around held-out epochs
| scope | n_epochs | n_obj | pct_ge2_within_12h | pct_ge2_within_24h | pct_ge1_within_12h | median_neighbour_gap_h | pct_neighbour_removed_by_paf |
|---|---|---|---|---|---|---|---|
| ALL | 127237 | 3542 | 18.419 | 71.496 | 63.436 | 10.051 | 5.663 |
| DEB|HEO | 474 | 13 | 11.603 | 50.422 | 37.131 | 22.799 | 7.595 |
| DEB|LEO | 60650 | 1590 | 15.115 | 69.923 | 62.706 | 10.251 | 0.653 |
| DEB|MEO | 2658 | 89 | 7.148 | 34.951 | 37.810 | 17.789 | 8.315 |
| PAY|GEO | 6105 | 215 | 4.685 | 53.071 | 36.200 | 16.395 | 39.459 |
| PAY|HEO | 3216 | 113 | 3.762 | 43.035 | 31.157 | 21.252 | 33.333 |
| PAY|LEO | 23984 | 658 | 32.246 | 93.579 | 85.223 | 7.189 | 1.426 |
| PAY|MEO | 5545 | 154 | 3.066 | 30.352 | 24.418 | 22.027 | 6.078 |
| PAY|VLEO | 400 | 10 | 34.750 | 94.750 | 85.500 | 7.783 | 0.250 |
| R/B|GEO | 2299 | 81 | 5.481 | 64.985 | 40.322 | 13.535 | 44.367 |
| R/B|HEO | 2650 | 87 | 6.264 | 41.358 | 34.113 | 21.581 | 25.585 |
| R/B|LEO | 12677 | 350 | 34.598 | 94.131 | 86.850 | 6.966 | 0.947 |
| R/B|MEO | 6323 | 175 | 13.063 | 55.733 | 49.391 | 12.048 | 8.556 |
| R/B|VLEO | 118 | 3 | 42.373 | 95.763 | 90.678 | 6.286 | 0.000 |

## P4 storm-aware sigma (T3 -> T4-UQ chaining), Sentinel test window
| model | split | n | cov68 | cov95 | nees | err_p50_km | sigma_p50_km |
|---|---|---|---|---|---|---|---|
| sigma-age | all | 26370 | 0.670 | 0.925 | 4.070 | 0.456 | 0.647 |
| sigma-age | storm(Ap>=40) | 1992 | 0.656 | 0.901 | 6.076 | 0.442 | 0.649 |
| sigma-age | quiet | 24378 | 0.671 | 0.927 | 3.906 | 0.457 | 0.647 |
| sigma-age x g(Ap obs lag1) | all | 26370 | 0.668 | 0.924 | 4.090 | 0.456 | 0.645 |
| sigma-age x g(Ap obs lag1) | storm(Ap>=40) | 1992 | 0.659 | 0.899 | 6.048 | 0.442 | 0.649 |
| sigma-age x g(Ap obs lag1) | quiet | 24378 | 0.669 | 0.927 | 3.930 | 0.457 | 0.645 |
| sigma-age x g(Ap T3-forecast) | all | 26370 | 0.669 | 0.925 | 4.086 | 0.456 | 0.646 |
| sigma-age x g(Ap T3-forecast) | storm(Ap>=40) | 1992 | 0.662 | 0.900 | 6.026 | 0.442 | 0.650 |
| sigma-age x g(Ap T3-forecast) | quiet | 24378 | 0.669 | 0.927 | 3.928 | 0.457 | 0.646 |
| sigma-age x g(Ap perfect) | all | 26370 | 0.668 | 0.924 | 4.264 | 0.456 | 0.646 |
| sigma-age x g(Ap perfect) | storm(Ap>=40) | 1992 | 0.661 | 0.902 | 5.972 | 0.442 | 0.663 |
| sigma-age x g(Ap perfect) | quiet | 24378 | 0.669 | 0.926 | 4.125 | 0.457 | 0.644 |

## P5 trajectory continuity at TLE switch epochs (1-min cadence, Sentinels)
| method | n_switches | jump_p50_m | jump_p90_m | jump_max_m | bg_jump_p50_m | frac_jump_gt_50m | frac_jump_gt_200m | err_step_km | err_change_at_switch_km |
|---|---|---|---|---|---|---|---|---|---|
| MTF | 576 | 7.469 | 13.660 | 7190.351 | 4.310 | 0.010 | 0.009 | 0.538 | 0.130 |
| MTF-adaptive | 576 | 7.551 | 14.115 | 382.903 | 4.280 | 0.005 | 0.003 | 0.546 | 0.109 |
| blend-2 | 576 | 7.262 | 12.705 | 17.841 | 4.310 | 0.000 | 0.000 | 0.549 | 0.108 |
| nearest | 576 | 24.497 | 137.522 | 7190.351 | 4.287 | 0.252 | 0.057 | 0.532 | 0.169 |

## E5 cohort thinning (truth-free LOO): nearest vs MTF by removal fraction
| frac | n | nearest_p50 | mtf_p50 | nearest_p90 | mtf_p90 | mtf_win_rate | age_nearest_med_h |
|---|---|---|---|---|---|---|---|
| 0.000 | 6877 | 0.164 | 0.159 | 0.900 | 0.897 | 0.096 | 11.789 |
| 0.100 | 34385 | 0.171 | 0.168 | 0.957 | 0.954 | 0.078 | 11.994 |
| 0.300 | 34385 | 0.191 | 0.189 | 1.103 | 1.101 | 0.048 | 14.671 |
| 0.500 | 34385 | 0.231 | 0.230 | 1.426 | 1.425 | 0.027 | 21.694 |
| 0.700 | 34385 | 0.309 | 0.309 | 2.268 | 2.269 | 0.010 | 24.961 |

## E5 per-object best-method flips across removal fractions
| n_obj | objects_with_flip | flip_rate | best_at_0 | best_at_0p5 | flip_rate_DEB|HEO | flip_rate_DEB|LEO | flip_rate_DEB|MEO | flip_rate_PAY|GEO | flip_rate_PAY|HEO | flip_rate_PAY|LEO | flip_rate_PAY|MEO | flip_rate_PAY|VLEO | flip_rate_R/B|GEO | flip_rate_R/B|HEO | flip_rate_R/B|LEO | flip_rate_R/B|MEO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 430 | 147 | 0.342 | {'nearest': 300, 'MTF': 130} | {'nearest': 375, 'MTF': 55} | 0.308 | 0.375 | 0.250 | 0.250 | 0.225 | 0.650 | 0.150 | 0.300 | 0.275 | 0.225 | 0.600 | 0.400 |

## E6 best MTF setting per type x regime and loss of a single global setting
| type | regime | window_h | tau_h | k | res_p50 | res_p50_at_global_setting | loss_of_global_pct |
|---|---|---|---|---|---|---|---|
| DEB | HEO | 24.000 | 4.000 | 3 | 0.536 | 0.536 | 0.000 |
| DEB | LEO | 24.000 | 4.000 | 2 | 0.098 | 0.100 | 1.714 |
| DEB | MEO | 24.000 | 4.000 | 5 | 0.294 | 0.294 | 0.000 |
| PAY | GEO | 24.000 | 4.000 | 3 | 0.288 | 0.288 | 0.105 |
| PAY | HEO | 24.000 | 4.000 | 3 | 0.273 | 0.273 | 0.000 |
| PAY | LEO | 24.000 | 4.000 | 5 | 0.023 | 0.023 | 0.000 |
| PAY | MEO | 24.000 | 4.000 | 5 | 0.105 | 0.105 | 0.000 |
| PAY | VLEO | 24.000 | 2.000 | 3 | 0.101 | 0.117 | 16.279 |
| R/B | GEO | 24.000 | 4.000 | 3 | 0.170 | 0.170 | 0.000 |
| R/B | HEO | 24.000 | 4.000 | 3 | 0.263 | 0.263 | 0.000 |
| R/B | LEO | 24.000 | 4.000 | 5 | 0.026 | 0.026 | 0.000 |
| R/B | MEO | 24.000 | 4.000 | 2 | 0.185 | 0.185 | 0.021 |

## E7 error census on the full cohort archive (2006 -> 2025), % of TLEs hit by each rule
| scope | n_obj | n_tle | half_period_pct | bstar_neg_pct | isolated_n_pct | any_rule_pct | objects_with_bstar_neg_pct | median_gap_h |
|---|---|---|---|---|---|---|---|---|
| ALL | 3603 | 32191037 | 3.087 | 10.401 | 0.033 | 13.455 | 75.853 | 16.243 |
| DEB|GEO | 2 | 14532 | 28.227 | 0.000 | 0.048 | 28.255 | 0.000 | 21.219 |
| DEB|HEO | 13 | 55243 | 5.956 | 1.345 | 0.005 | 7.197 | 38.462 | 23.969 |
| DEB|LEO | 1613 | 14547368 | 0.455 | 8.248 | 0.013 | 8.691 | 88.407 | 16.765 |
| DEB|MEO | 94 | 535847 | 2.959 | 18.264 | 0.050 | 20.832 | 92.553 | 22.569 |
| DEB|VLEO | 2 | 20614 | 0.475 | 3.415 | 0.005 | 3.886 | 100.000 | 15.958 |
| PAY|GEO | 215 | 1411909 | 25.111 | 0.000 | 0.221 | 25.279 | 0.000 | 23.789 |
| PAY|HEO | 117 | 667700 | 22.716 | 3.709 | 0.072 | 26.123 | 17.949 | 24.037 |
| PAY|LEO | 674 | 7439813 | 0.431 | 15.434 | 0.011 | 15.834 | 95.549 | 13.878 |
| PAY|MEO | 157 | 1008876 | 5.888 | 5.285 | 0.184 | 11.239 | 20.382 | 22.524 |
| PAY|VLEO | 10 | 121473 | 0.542 | 8.604 | 0.002 | 9.114 | 100.000 | 12.717 |
| R/B|GEO | 81 | 494526 | 24.719 | 0.000 | 0.060 | 24.766 | 0.000 | 23.682 |
| R/B|HEO | 87 | 514277 | 18.371 | 5.763 | 0.063 | 23.786 | 36.782 | 24.011 |
| R/B|LEO | 359 | 4061114 | 0.413 | 17.065 | 0.010 | 17.438 | 99.443 | 13.718 |
| R/B|MEO | 176 | 1261326 | 5.729 | 6.843 | 0.088 | 12.420 | 65.341 | 21.068 |
| R/B|VLEO | 3 | 36419 | 0.409 | 8.386 | 0.000 | 8.789 | 66.667 | 12.765 |

## E7 LOO residual of PAF-removed TLEs by rule vs kept TLEs
| method | scope | rule | n | res3d_p50 | res3d_p90 | ratio_p50_vs_kept | ratio_p90_vs_kept | frac_gt_1km | kept_frac_gt_1km |
|---|---|---|---|---|---|---|---|---|---|
| MTF-raw | ALL | bstar_neg | 8134 | 0.049 | 0.479 | 0.634 | 0.913 | 0.045 | 0.043 |
| MTF-raw | ALL | half_period | 6126 | 0.186 | 0.786 | 2.409 | 1.499 | 0.068 | 0.043 |
| MTF-raw | ALL | kept | 127475 | 0.077 | 0.524 | 1.000 | 1.000 | 0.043 | 0.043 |
| nearest-raw | ALL | bstar_neg | 8134 | 0.051 | 0.487 | 0.639 | 0.922 | 0.045 | 0.043 |
| nearest-raw | ALL | half_period | 6126 | 0.191 | 0.803 | 2.416 | 1.520 | 0.071 | 0.043 |
| nearest-raw | ALL | kept | 127475 | 0.079 | 0.528 | 1.000 | 1.000 | 0.043 | 0.043 |

## E8 truth availability
| cohort_objects | objects_with_pod_truth | pct_objects_with_truth | starlink_ephemeris_objects | starlink_ephemeris_overlaps_tle_archive | tle_epochs_cohort_2006plus | note |
|---|---|---|---|---|---|---|
| 3603 | 5 | 0.139 | 10653 | False | 32191037 | POD truth exists only for the Sentinel satellites; Starlink ephemerides start 2026-06, after the TLE archive ends (2025-12-31). |

## E3-LOO Truth-free reconstruction consistency on the whole cohort (held-out TLE as reference)
| scope | method | n | n_obj | res3d_p50_km | res3d_p90_km | res3d_p95_km | res_t_p50_km | spread_med_km |
|---|---|---|---|---|---|---|---|---|
| ALL | MTF-PAF | 127475 | 3548 | 0.079 | 0.551 | 0.930 | 0.062 | 0.000 |
| ALL | MTF-raw | 127475 | 3548 | 0.077 | 0.524 | 0.897 | 0.061 | 0.000 |
| ALL | nearest-PAF | 127475 | 3548 | 0.081 | 0.554 | 0.932 | 0.063 | 0.000 |
| ALL | nearest-raw | 127475 | 3548 | 0.079 | 0.528 | 0.902 | 0.062 | 0.000 |
| ALL | product (MTF+PAF, in-sample) | 20143 | 565 | 0.001 | 0.015 | 0.140 | 0.001 |  |
| ALL | reference TLE_k itself | 161 | 5 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| DEB|LEO | MTF-PAF | 60650 | 1590 | 0.087 | 0.430 | 0.738 | 0.071 | 0.000 |
| DEB|LEO | MTF-raw | 60650 | 1590 | 0.087 | 0.428 | 0.733 | 0.071 | 0.000 |
| DEB|LEO | nearest-PAF | 60650 | 1590 | 0.088 | 0.433 | 0.742 | 0.072 | 0.000 |
| DEB|LEO | nearest-raw | 60650 | 1590 | 0.088 | 0.431 | 0.737 | 0.072 | 0.000 |
| DEB|LEO | product (MTF+PAF, in-sample) | 11100 | 294 | 0.002 | 0.015 | 0.051 | 0.001 |  |
| PAY|LEO | MTF-PAF | 24222 | 664 | 0.029 | 0.091 | 0.141 | 0.019 | 0.000 |
| PAY|LEO | MTF-raw | 24222 | 664 | 0.029 | 0.090 | 0.140 | 0.019 | 0.000 |
| PAY|LEO | nearest-PAF | 24222 | 664 | 0.031 | 0.094 | 0.144 | 0.021 | 0.000 |
| PAY|LEO | nearest-raw | 24222 | 664 | 0.030 | 0.093 | 0.142 | 0.021 | 0.000 |
| PAY|LEO | product (MTF+PAF, in-sample) | 4112 | 114 | 0.001 | 0.013 | 249.734 | 0.001 |  |
| PAY|LEO | reference TLE_k itself | 161 | 5 | 0.000 | 0.000 | 0.000 | 0.000 |  |
| PAY|VLEO | MTF-PAF | 400 | 10 | 0.106 | 0.422 | 0.864 | 0.099 | 0.000 |
| PAY|VLEO | MTF-raw | 400 | 10 | 0.106 | 0.422 | 0.864 | 0.099 | 0.000 |
| PAY|VLEO | nearest-PAF | 400 | 10 | 0.106 | 0.407 | 0.864 | 0.098 | 0.000 |
| PAY|VLEO | nearest-raw | 400 | 10 | 0.106 | 0.407 | 0.864 | 0.098 | 0.000 |
| R/B|LEO | MTF-PAF | 12677 | 350 | 0.035 | 0.131 | 0.194 | 0.026 | 0.000 |
| R/B|LEO | MTF-raw | 12677 | 350 | 0.035 | 0.131 | 0.193 | 0.026 | 0.000 |
| R/B|LEO | nearest-PAF | 12677 | 350 | 0.037 | 0.136 | 0.198 | 0.028 | 0.000 |
| R/B|LEO | nearest-raw | 12677 | 350 | 0.037 | 0.136 | 0.197 | 0.028 | 0.000 |
| R/B|LEO | product (MTF+PAF, in-sample) | 2333 | 63 | 0.001 | 0.008 | 0.024 | 0.001 |  |
| R/B|VLEO | MTF-PAF | 118 | 3 | 0.086 | 0.267 | 0.331 | 0.073 | 0.000 |
| R/B|VLEO | MTF-raw | 118 | 3 | 0.086 | 0.267 | 0.331 | 0.073 | 0.000 |
| R/B|VLEO | nearest-PAF | 118 | 3 | 0.082 | 0.308 | 0.370 | 0.070 | 0.000 |
| R/B|VLEO | nearest-raw | 118 | 3 | 0.082 | 0.308 | 0.370 | 0.070 | 0.000 |
| source=multi | MTF-PAF | 20482 | 565 | 0.066 | 0.360 | 0.647 | 0.049 |  |
| source=multi | MTF-raw | 20482 | 565 | 0.065 | 0.348 | 0.619 | 0.048 |  |
| source=multi | nearest-PAF | 20482 | 565 | 0.067 | 0.360 | 0.643 | 0.050 |  |
| source=multi | nearest-raw | 20482 | 565 | 0.067 | 0.349 | 0.618 | 0.050 |  |
| source=multi | product (MTF+PAF, in-sample) | 20143 | 565 | 0.001 | 0.015 | 0.140 | 0.001 |  |
| source=pruned | MTF-PAF | 106755 | 2977 | 0.083 | 0.585 | 0.975 | 0.065 |  |
| source=pruned | MTF-raw | 106755 | 2977 | 0.081 | 0.557 | 0.939 | 0.064 |  |
| source=pruned | nearest-PAF | 106755 | 2977 | 0.085 | 0.588 | 0.977 | 0.067 |  |
| source=pruned | nearest-raw | 106755 | 2977 | 0.083 | 0.561 | 0.943 | 0.066 |  |

## E3-LOO paired per-object comparison vs nearest-raw
| method | n_obj | win_rate_vs_nearest_raw | median_rel_change_pct | median_res3d_km | nearest_raw_median_km |
|---|---|---|---|---|---|
| MTF-PAF | 3548 | 0.424 | 0.000 | 0.087 | 0.088 |
| MTF-raw | 3548 | 0.427 | 0.000 | 0.086 | 0.088 |
| nearest-PAF | 3548 | 0.054 | 0.000 | 0.089 | 0.088 |
| product (MTF+PAF, in-sample) | 565 | 0.961 | -97.911 | 0.001 | 0.072 |
| reference TLE_k itself | 5 | 1.000 | -100.000 | 0.000 | 0.022 |

## E3-LOO residual of TLEs removed vs kept by PAF-lite
| k_removed_by_paf | n | res3d_p50_km | res3d_p90_km |
|---|---|---|---|
| False | 127475 | 0.079 | 0.528 |
| True | 14260 | 0.099 | 0.664 |

## E3-LOO proxy validation on Sentinels (LOO residual vs POD error)
| method | n | loo_res3d_p50_km | pod_err3d_p50_km | spearman_loo_vs_pod |
|---|---|---|---|---|
| MTF-PAF | 163 | 0.020 | 0.507 | 0.044 |
| MTF-raw | 163 | 0.020 | 0.503 | 0.008 |
| nearest-PAF | 163 | 0.025 | 0.520 | 0.061 |
| nearest-raw | 163 | 0.025 | 0.503 | 0.026 |
| reference TLE_k itself | 163 | 0.000 | 0.501 |  |

## E4 delivered OAD proxies vs per-object LOO error
| loo_method | n_obj | spearman_overlap | spearman_spread |
|---|---|---|---|
| nearest-raw | 482 | 0.617 | 0.606 |
| MTF-PAF | 482 | 0.612 | 0.604 |
| product (MTF+PAF, in-sample) | 482 | 0.123 | 0.104 |

## E1 Task chaining: independent vs chained
| exp | edge | h | independent | chained | metric | rel_change_pct |
|---|---|---|---|---|---|---|
| E1 | maneuver→forecast (T2→T4) | 1 | 0.571 | 0.571 | 3D P50 km | -0.009 |
| E1 | phase+maneuver→forecast (T1,T2→T4) | 1 | 0.571 | 0.571 | 3D P50 km | -0.009 |
| E1 | weather→forecast (T3→T4) | 1 | 0.795 | 0.773 | 3D P50 km | -2.708 |
| E1 | maneuver→forecast, P95 | 1 | 1.219 | 1.222 | 3D P95 km | 0.230 |
| E1 | maneuver→forecast (T2→T4) | 4 | 0.495 | 0.496 | 3D P50 km | 0.161 |
| E1 | phase+maneuver→forecast (T1,T2→T4) | 4 | 0.495 | 0.496 | 3D P50 km | 0.161 |
| E1 | weather→forecast (T3→T4) | 4 | 0.641 | 0.622 | 3D P50 km | -3.044 |
| E1 | maneuver→forecast, P95 | 4 | 1.695 | 1.695 | 3D P95 km | 0.000 |
| E1 | maneuver→forecast (T2→T4) | 24 | 0.457 | 0.455 | 3D P50 km | -0.362 |
| E1 | phase+maneuver→forecast (T1,T2→T4) | 24 | 0.457 | 0.455 | 3D P50 km | -0.362 |
| E1 | weather→forecast (T3→T4) | 24 | 0.489 | 0.497 | 3D P50 km | 1.623 |
| E1 | maneuver→forecast, P95 | 24 | 4.216 | 4.246 | 3D P95 km | 0.719 |
| E1 | maneuver→forecast [maneuver windows] | 1 | 0.414 | 0.415 | 3D P50 km | 0.103 |
| E1 | maneuver→forecast [quiet windows] | 1 | 0.597 | 0.595 | 3D P50 km | -0.262 |
| E1 | maneuver→forecast [maneuver windows] | 24 | 0.969 | 0.954 | 3D P50 km | -1.555 |
| E1 | maneuver→forecast [quiet windows] | 24 | 0.429 | 0.429 | 3D P50 km | -0.013 |

## E2 Data ablation
| exp | config | h | forecast_p50_km | corrected_p50_km |
|---|---|---|---|---|
| E2 | B0 Full (MTF+T2, Ridge-weather+T3, σ-age) | 1 | 0.571 | 0.773 |
| E2 | B1 No-maneuver labels | 1 | 0.571 | 0.773 |
| E2 | B2 No-weather | 1 | 0.571 | 0.792 |
| E2 | B4 No-precise (no POD-trained corrector) | 1 | 0.571 |  |
| E2 | B5 TLE-only (SGP4-prev) | 1 | 0.571 |  |
| E2 | B0 Full (MTF+T2, Ridge-weather+T3, σ-age) | 4 | 0.496 | 0.622 |
| E2 | B1 No-maneuver labels | 4 | 0.495 | 0.622 |
| E2 | B2 No-weather | 4 | 0.496 | 0.644 |
| E2 | B4 No-precise (no POD-trained corrector) | 4 | 0.496 |  |
| E2 | B5 TLE-only (SGP4-prev) | 4 | 0.495 |  |
| E2 | B0 Full (MTF+T2, Ridge-weather+T3, σ-age) | 24 | 0.455 | 0.497 |
| E2 | B1 No-maneuver labels | 24 | 0.457 | 0.497 |
| E2 | B2 No-weather | 24 | 0.455 | 0.482 |
| E2 | B4 No-precise (no POD-trained corrector) | 24 | 0.455 |  |
| E2 | B5 TLE-only (SGP4-prev) | 24 | 0.455 |  |

## E3 Trajectory quality vs POD (C0 raw / C1 PAF-lite / C2 PAF-lite+MTF)
| config | sat | split | n | n_tle | coverage | age_med_h | e3d_p50_km | e3d_p90_km | e3d_p95_km | abs_r_p50_km | abs_t_p50_km | abs_n_p50_km |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| C0 raw + nearest | S2A | all | 3030 | 774.000 | 1.000 | -0.028 | 0.502 | 0.924 | 1.026 | 0.069 | 0.468 | 0.097 |
| C1 PAF-lite + nearest | S2A | all | 3030 | 774.000 | 1.000 | -0.028 | 0.502 | 0.924 | 1.026 | 0.069 | 0.468 | 0.097 |
| C2 PAF-lite + MTF | S2A | all | 3030 | 774.000 | 1.000 | 2.170 | 0.509 | 0.924 | 1.042 | 0.069 | 0.476 | 0.097 |
| C0 raw + nearest | S3B | all | 24918 | 361.000 | 1.000 | -0.011 | 0.513 | 0.920 | 1.059 | 0.071 | 0.473 | 0.098 |
| C1 PAF-lite + nearest | S3B | all | 24918 | 361.000 | 1.000 | -0.011 | 0.513 | 0.920 | 1.059 | 0.071 | 0.473 | 0.098 |
| C2 PAF-lite + MTF | S3B | all | 24918 | 361.000 | 1.000 | 3.560 | 0.510 | 0.918 | 1.054 | 0.071 | 0.471 | 0.098 |
| C0 raw + nearest | S1A | all | 25056 | 780.000 | 1.000 | -0.003 | 0.531 | 0.967 | 1.071 | 0.070 | 0.491 | 0.103 |
| C1 PAF-lite + nearest | S1A | all | 25056 | 736.000 | 1.000 | -0.002 | 0.530 | 0.965 | 1.071 | 0.070 | 0.491 | 0.103 |
| C2 PAF-lite + MTF | S1A | all | 25056 | 736.000 | 1.000 | 2.365 | 0.528 | 0.969 | 1.074 | 0.070 | 0.488 | 0.103 |
| C0 raw + nearest | S3A | all | 24918 | 793.000 | 1.000 | -0.006 | 0.507 | 0.890 | 0.971 | 0.067 | 0.470 | 0.095 |
| C1 PAF-lite + nearest | S3A | all | 24918 | 792.000 | 1.000 | -0.006 | 0.507 | 0.890 | 0.971 | 0.067 | 0.470 | 0.095 |
| C2 PAF-lite + MTF | S3A | all | 24918 | 792.000 | 1.000 | 2.231 | 0.507 | 0.891 | 0.973 | 0.067 | 0.469 | 0.095 |
| C0 raw + nearest | S2B | all | 25056 | 765.000 | 1.000 | 0.019 | 0.510 | 0.901 | 0.995 | 0.069 | 0.470 | 0.095 |
| C1 PAF-lite + nearest | S2B | all | 25056 | 765.000 | 1.000 | 0.019 | 0.510 | 0.901 | 0.995 | 0.069 | 0.470 | 0.095 |
| C2 PAF-lite + MTF | S2B | all | 25056 | 765.000 | 1.000 | 2.284 | 0.508 | 0.901 | 0.998 | 0.069 | 0.469 | 0.095 |
| C0 raw + nearest | ALL | all | 102978 |  | 1.000 |  | 0.510 | 0.920 | 1.026 |  |  |  |
| C1 PAF-lite + nearest | ALL | all | 102978 |  | 1.000 |  | 0.510 | 0.920 | 1.026 |  |  |  |
| C2 PAF-lite + MTF | ALL | all | 102978 |  | 1.000 |  | 0.509 | 0.918 | 1.042 |  |  |  |

## E4 Uncertainty
| exp | stage | model | component | cov68 | cov95 | nees | spearman | lead_h | recall_at_k | recall_at_2k | safety_rate | dv_mean_m_s | pc_post_median |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E4 | T4 σ calibration | σ-fixed | r | 0.810 | 0.997 | 0.546 |  |  |  |  |  |  |  |
| E4 | T4 σ calibration | σ-fixed | t | 0.674 | 0.923 | 5.468 |  |  |  |  |  |  |  |
| E4 | T4 σ calibration | σ-fixed | n | 0.900 | 0.995 | 0.480 |  |  |  |  |  |  |  |
| E4 | T4 σ calibration | σ-age | r | 0.795 | 0.989 | 0.604 | 0.073 |  |  |  |  |  |  |
| E4 | T4 σ calibration | σ-age | t | 0.670 | 0.925 | 4.070 | 0.069 |  |  |  |  |  |  |
| E4 | T4 σ calibration | σ-age | n | 0.898 | 0.994 | 0.484 | 0.018 |  |  |  |  |  |  |
| E4 | T4 σ calibration | σ-oad | r | 0.801 | 0.995 | 0.573 | 0.058 |  |  |  |  |  |  |
| E4 | T4 σ calibration | σ-oad | t | 0.647 | 0.923 | 4.703 | 0.091 |  |  |  |  |  |  |
| E4 | T4 σ calibration | σ-oad | n | 0.883 | 0.993 | 0.538 | 0.047 |  |  |  |  |  |  |
| E4 | T6 risk ranking | UQ-point | nan |  |  |  |  | 6.000 | 0.884 | 1.000 |  |  |  |
| E4 | T6 risk ranking | UQ-fixed | nan |  |  |  |  | 6.000 | 0.884 | 1.000 |  |  |  |
| E4 | T6 risk ranking | UQ-age | nan |  |  |  |  | 6.000 | 0.837 | 0.977 |  |  |  |
| E4 | T6 risk ranking | UQ-point | nan |  |  |  |  | 24.000 | 0.814 | 0.953 |  |  |  |
| E4 | T6 risk ranking | UQ-fixed | nan |  |  |  |  | 24.000 | 0.837 | 0.953 |  |  |  |
| E4 | T6 risk ranking | UQ-age | nan |  |  |  |  | 24.000 | 0.837 | 0.977 |  |  |  |
| E4 | T7 avoidance | No-maneuver | nan |  |  |  |  |  |  |  | 0.000 | 0.000 | 0.000 |
| E4 | T7 avoidance | PPO | nan |  |  |  |  |  |  |  | 1.000 | 45.000 | 0.000 |
| E4 | T7 avoidance | Physics-Pc | nan |  |  |  |  |  |  |  | 0.000 | 0.114 | 0.000 |
| E4 | T7 avoidance | Physics-max | nan |  |  |  |  |  |  |  | 1.000 | 9.843 | 0.000 |
| E4 | T7 avoidance | Proportional | nan |  |  |  |  |  |  |  | 0.032 | 10.520 | 0.000 |

## Figures
- `figures/fig_c1_coverage.pdf`
- `figures/fig_c4_budget.pdf`
- `figures/fig_chain_ablation.pdf`
- `figures/fig_e1_chaining.pdf`
- `figures/fig_e3_quality.pdf`
- `figures/fig_p1_thinning.pdf`
- `figures/fig_p2_injection.pdf`
- `figures/fig_s1_starlink_forecast.pdf`
- `figures/fig_t4_forecast_error.pdf`
- `figures/fig_t4_uq_coverage.pdf`

## Key findings
### Evidence for Challenge 1 (a unified dataset is hard to build)
1. **TLE cadence is too heterogeneous for one fusion setting.** Around held-out epochs, only 18% of cohort epochs have >=2 TLEs within +-12 h (LEO payloads 32%, GEO payloads 5%, MEO payloads 3%); the median gap to the nearest neighbour is 10 h overall but 16-22 h in GEO/HEO (P3 sparsity). A fixed 12-h fusion window therefore activates fusion for fewer than one epoch in five, and MTF degenerates to nearest-TLE for most objects.
2. **The best reconstruction method depends on density, and the proxy-optimal setting differs from the truth-optimal one.** In the truth-free cohort test the best method flips between nearest and MTF for 34% of objects when TLEs are thinned (65% of LEO payloads, E5); the LOO proxy prefers the widest window (24 h) in every type x regime cell (E6), whereas with POD truth every wider or adaptive window is worse than the plain 12-h MTF (P1). Choosing a propagation setting without truth is therefore not possible in general.
### Evidence for Challenge 2 (data quality is hard to control)
3. **Errors are numerous and regime-dependent** (E7 census, 32.2 M TLEs of 3,603 objects, 2006-2025): 13.5% of records trigger at least one PAF rule; B* < 0 affects 10.4% of records and 76% of objects (LEO payloads 15%, LEO rocket bodies 17%); half-period re-releases are 0.4% of LEO records but 25-28% of GEO/HEO records.
4. **Removal rules have opposite effects in different regimes.** Superseded (half-period) records are 1.9-2.4x less consistent with the rest of the archive in LEO, i.e. the revision corrected them, but 0.6x *more* consistent in GEO/HEO, where half a period is 12 h and the rule deletes legitimate consecutive TLEs (E7 by rule). B* < 0 records are not less consistent than kept ones (0.64x median residual), so a hard delete is not justified by trajectory consistency. Removing neighbours also thins the pool: MTF on the PAF-lite pool is 2% worse than on the raw pool in the LOO test.
5. **On precisely tracked payloads, every 'smarter' fusion is worse than the simple one** (P1, POD truth, 5 Sentinels, 2024-2025H1): adaptive windows +24.5%, bracketing blend +19.4%, T2-gated segments +25.3% and even oracle maneuver gating +24.1% at 10% TLE density; at full density the penalties are 0.2-2.4%. The error floor (~0.5 km) is SGP4 model bias shared by all TLEs, which fusion cannot remove.
6. **Quality cannot be quantified from truth for 99.9% of objects.** 5 of 3,608 cohort objects (0.14%) have any independent reference (E8). The TLE-consistency residual does not track the true POD error at the sample level (Spearman 0.04, E3 Sentinel validation) but does at the object level (0.61 with the delivered OAD proxies, E4).
### Positive evidence for the OrbBench methods
7. **PAF removes faults that would corrupt trajectories** (P2, POD truth): with faults injected at the archive's own type mix and magnitudes, PAF-lite recovers 74-77% of them at a 1.3-3.4% removal rate on clean records; at a 10% fault rate the P90 error of MTF trajectories drops from 1.17 to 0.98 km (-16%) and the P50 from 0.554 to 0.528 km; the P99 tail is cut from 934 to 326 km for MTF and from 14.1 to 5.8 km for nearest. Fusion amplifies undetected faults (raw MTF P99 934 km vs raw nearest 14 km), so PAF is a precondition for MTF rather than an alternative to it.
8. **MTF gives smoother products, not lower error.** Against POD, MTF beats nearest by 0.36% [95% CI 0.22-0.48%] at full density and by 2.5% in the truth-free cohort test; its main benefit is continuity: the kinematic jump at TLE switch epochs falls from 24.5 m (P50) / 138 m (P90) with nearest to 7.5 / 14 m with MTF, and the share of switches with jumps above 50 m from 25% to 1% (P5).
9. **Calibrated, age-dependent uncertainty is better than a fixed one**: at 24 h NEES 3.1 vs 4.7 and 95% coverage 0.90 vs 0.88 (T4-UQ). The delivered object-level OAD proxies correlate 0.61 (Spearman) with reconstruction error on 482 objects.
10. **High-precision references have measurable value**: a corrector trained on 2019-2023 POD reduces the forecast error of SGP4-MTF by 38% (0.5 h), 46% (1 h), 23% (4 h) and 58% (8 h), and by 48% in post-maneuver windows at 4 h, but is neutral at 12 h and 15% worse at 24 h with heavier tails (T4, long window).
11. **Maneuvers matter and detection works**: forecast error is 49-77% higher in the 48 h after a logged maneuver (T4 split); the SGP4-consistency detector reaches recall 0.64 / F2 0.54 on 272 logged maneuvers while a supervised model does not transfer across years (F2 0.16).
### The connected architecture: which edges pay off (C1-C5, `chain_a0_a1.csv`)
12. **Events -> uncertainty (T2 -> UQ) is the edge that matters for maneuvering objects.** Selecting TLEs by maneuver segment changes forecast error by <0.2% (the post-maneuver TLE already carries the maneuver), but feeding the maneuver signal into sigma does: on Sentinel POD (test 2024-25H1, 3 satellites with logs) an age-only sigma covers 78.0% of along-track errors at 95% nominal in the 48 h after a maneuver (NEES 10.3); scaling sigma by the T2 consistency score raises coverage to 81.6% (NEES 7.6) with no truth used, and using the OrbBench maneuver records to 86.7% (NEES 3.9); overall NEES falls from 4.8 to 2.4 while quiet-period coverage stays at 0.96. Uncertainty, not the point estimate, is where event knowledge should enter.
13. **Population evolution -> screening (T5 -> T6) works across shells, not within one.** On 600 LEO objects (300-2000 km, 3 d, 139 reference approaches < 10 km), ordering candidate pairs by T5-predicted shell density recovers 34.8% of approaches with 20% of the pairs and 65.2% with 50% (random: 20.5% / 53.8%; AUC 0.66 vs 0.52). A simple 'same shell first' ordering does even better (47.7% / 81.1%, AUC 0.71), so the gain comes from altitude structure that T5 predicts, and prioritisation only pays when the cohort spans several shells (inside one 700-1000 km shell it lost recall).
14. **Uncertainty -> avoidance (UQ -> T7) is worth a factor 34 in propellant, but only with per-event calibration.** On 117 ESA CDM events (miss < 2 km), a 2-km distance rule is safe on 100% with mean delta-v 10.3 m/s; a Pc-gated policy using the per-event covariance is safe on 100% (true Pc < 1e-4) with 0.30 m/s. Halving that sigma drops safety to 75%; a single fleet-wide sigma, the TLE-derived sigma-age or a conservative 2-km sigma never trigger and leave 25% of events unsafe. A dataset that supplies calibrated per-object uncertainty is therefore a prerequisite for probability-based avoidance.
15. **Uncertainty -> warning ranking (UQ -> T6) is not demonstrated** in the 700-1000 km debris shell: predicted-approach errors (~100 km, TLE age 14-37 h) swamp any covariance, and recall@K of true approaches stays at 0.16-0.17 for point-miss, fixed, age or object-level sigma. Space-weather chaining (T3 -> T4/UQ) shows no effect at Sentinel altitudes (storms raise 24-h error by 4%).
16. **End-to-end (A0 independent vs A1 chained)**: MTF product P90 switch jump 138 -> 14 m; post-maneuver along-track 95% coverage 0.78 -> 0.87; screening recall at 20% budget 0.20 -> 0.35; avoidance delta-v at equal safety 10.3 -> 0.30 m/s. Every edge with a positive result carries information through *uncertainty or priority*, not by editing the point trajectory.

### Starlink end-to-end test (17 satellites, operator MEME ephemerides with 6x6 covariance as truth; raw + official step6 TLEs to 2026-08-24; test 2026-07-05..08-23, 3-h origins; `starlink_a0_a1.csv`, `s1_*`, `s2_*`)
17. **A filtering task run on its own is harmful on maneuvering satellites; connected to a data-availability gate it is safe.** Starlinks thrust 82% of the time. The official step6/PAF pool removes 25% of their TLEs (retention 26-35% on three satellites) and those are the maneuver TLEs, so causal forecasts from the filtered pool alone reach P90 31.7 km at 1 h and 111 km at 24 h against 3.8 / 27.2 km from the raw archive (P50 +20% / +13%); on the worst satellites the 24-h P90 is 3,900-5,700 km. Falling back to the raw pool whenever the filtered pool has no TLE within 24 h (18% of origins) restores P90 to 4.1 / 27.3 km and P50 to within 3-4% of raw; a T1-aware PAF-lite that suspends the element-jump and B* rules inside detected raise/lower arcs (retention 82-98%) is equally safe. This is the clearest 'independent task bad, connected chain good' case in the study and the Starlink counterpart of the Sentinel result that PAF is a precondition for MTF.
18. **Thrust, not TLE age, sets the Starlink forecast error.** 24-h P50 is 2.7 km when no thrust falls in the horizon and 5.4 km when it does (T1 phase stable 2.7 km vs lowering 5.6 km). 32% of 24-h forecasts miss by > 10 km and 12% of along-track errors exceed 3 sigma, so Gaussian sigma models under-cover (fixed sigma 68/81% at 68/95% nominal; age sigma 50/64%; age x g(T2) 62/72%). The operator covariance describes the ephemeris (sigma_T 0.1-0.2 km) and covers 6% of the TLE error. Residual correctors trained on 3 weeks of MEME make forecasts 3-5x worse because the station-keeping direction alternates between months; T1/T2 features reduce the damage (1 h: 3.6 -> 2.7 km) but do not remove it.
19. **Maneuver detection transfers to Starlink**: the SGP4-consistency detector reaches precision 0.84 at recall 0.37 against MEME-derived thrust arcs (threshold tuned on the first 3 weeks), flagging thrust change-points rather than whole arcs. Gating the fusion on it changes forecast error by < 1%, consistent with the Sentinel result: maneuver knowledge belongs in sigma and in the filter, not in TLE selection.
20. **No dangerous conjunctions occur among the 17 Starlinks in 50 days** (134 approaches < 50 km, closest 17 km, true Pc from the operator covariances = 0), so UQ -> ranking and warning -> avoidance cannot be tested on this cohort; recall of the < 50 km approaches from TLE trajectories is 7-10% (A1 +22-29% relative to A0) at 2% precision, because TLE errors (5-27 km) are comparable to the approach distances.

### Task-level chain ablations: full chain vs the same chain with one upstream task removed (X1-X4, `x*_*.csv`, `fig_chain_ablation.pdf`)
21. **X3 Evolution -> warning -> avoidance (600 LEO objects, 3 d, 139 reference approaches < 10 km, 20% screening budget).** Full chain T5 -> T6 -> T7 reaches 51.6 of 139 approaches (recall 0.371) and spends 0 m/s because no reached approach has Pc >= 1e-4 under the T6 sigma; removing T5 (random pair order) reaches 28.4 (recall 0.205, -45%); removing the T6 sigma (2-km distance rule) keeps the 51.6 approaches but spends 14.0 m/s on events whose Pc is 0; A0 (neither) reaches 28.4 and spends 7.7 m/s. Oracle ordering reaches all 139.
22. **X2 Phase + maneuver detection -> forecast reliability -> screening triage (17 Starlinks, MEME truth, 4,473 test forecasts at 24 h, 31.7% miss by > 10 km).** T4 predicts which 24-h forecasts will be unreliable so that T6 does not screen on them. With the T1 phase/SMA-rate and T2 score as inputs the flag reaches AUC 0.651 and recall 0.451 at a 30% flag budget; removing T1 drops it to AUC 0.584 / recall 0.361 (below the age-only A0 baseline 0.602 / 0.369), removing T2 leaves it unchanged (0.654 / 0.451), and adding the T3 Ap/F10.7 series does not help at 24 h (0.643 / 0.428) but does at 4 h (AUC 0.733 with T3 vs 0.710 without; -T1 0.704; A0 0.718). Phase knowledge is the input that carries the maneuver information the forecast lacks.
23. **X1 Weather + object attributes -> chaser-uncertainty model -> Pc alert -> Pc-gated avoidance (199 ESA CDMs, operator covariances as truth, 56 dangerous events with true Pc >= 1e-4).** T4 predicts the chaser RTN sigma from the OD age plus T5 object/orbit attributes and T3 weather (5-fold CV; 42% of along-track sigmas within a factor 2 vs 29% for a single fixed sigma), T6 alerts at predicted Pc >= 1e-4 and T7 maneuvers down to 1e-5. Full chain: 12 of 56 dangerous events missed (safety on dangerous 0.714), 16 unnecessary maneuvers, 458 m/s; removing T3: 14 missed (0.696); removing T5: 14 missed (0.661); removing both (age-only sigma): 17 missed (0.625). Removing T4 (fixed sigma) misses 10 but fires 25 unnecessary maneuvers and spends 816 m/s; removing T6 (2-km distance rule) misses none but fires 61 unnecessary maneuvers and spends 1,210 m/s. The operator covariance (oracle) misses none with 0 unnecessary and 227 m/s, so the remaining gap is the sigma model, not the chain.
24. **X4 Maneuver detection -> uncertainty -> Pc alert -> avoidance on POD truth (Sentinel-1A/2A/3A, test 2024-25H1, 2,913 forecast origins x 4 synthetic chasers per horizon: miss log-uniform 0.05-10 km, chaser sigma 0.2 km).** In the 48 h after a logged maneuver (1,463 dangerous conjunctions at 4 h) the full chain alerts on 67.9% of dangerous conjunctions and T7 leaves 64.9% of them safe; removing T2 (maneuver-blind sigma) gives 63.7% / 59.5% (+13% missed dangerous), a perfect T2 (logged maneuvers) 70.7% / 70.3%. Over all 4,799 dangerous conjunctions at 4 h the chain is safe on 81.6% (-T2 78.7%) with 22.6 km/s total delta-v, the distance rule on 95.7% with 78.7 km/s and 7.4x the unnecessary maneuvers; a fixed sigma is equivalent to the chain here (83.1%, +4% delta-v). With a perfect forecast every dangerous conjunction is handled at 15.3 km/s, so the forecast error, not the controller, is the limit.
25. **Reading across X1-X4**: every removal of an upstream *task output* degrades the downstream metric where the physics says it should (T5 ordering -> screening recall; T1 phase -> forecast reliability; T3/T5 attributes -> chaser sigma -> missed dangerous events; T2 -> post-maneuver safety), while removing the *uncertainty* task is neutral-to-harmful in propellant rather than in safety, and removing the *Pc alert* (distance rule) buys safety at 2.6-3.5x the delta-v. Space weather (T3) only pays off at short horizons.

### Significance, efficiency, and data ablation for Sec. 5.2-5.3 (E51-E53)
26. **Which chain effects survive a paired bootstrap (95% CI).** Removing the Pc alert (T6, distance rule) is significant everywhere: on the 199 CDMs it adds 45 [34, 57] unnecessary maneuvers and 752 [511, 985] m/s; on the Sentinel conjunctions it adds 2,678 [2,586, 2,764] unnecessary maneuvers and 56 [55, 57] km/s at 4 h while raising safety from 0.82 to 0.96. Removing the uncertainty task (T4, fixed sigma) is significant in propellant, not in safety: +9 [2, 17] unnecessary maneuvers and +359 [124, 581] m/s on the CDMs, +956 [807, 1,104] m/s on the Sentinels. Removing the maneuver detector (T2) is significant on the Sentinel post-maneuver conjunctions: safety 0.649 -> 0.595 [-0.068, -0.041] and alert recall 0.679 -> 0.637 [-0.054, -0.030]; a perfect detector (logged maneuvers) would give 0.703 [+0.041, +0.067]. Removing the phase task (T1) is significant on Starlink triage at 24 h: AUC 0.643 -> 0.584 [-0.078, -0.038], recall at 30% budget 0.428 -> 0.361 [-0.091, -0.038]; removing T2 or T3 there is neutral or slightly positive, and at 4 h T3 (weather) and T1 are both significant (AUC -0.023 [-0.030, -0.016] and -0.029 [-0.047, -0.014]). On the CDMs the T3/T5 inputs to the sigma model change missed dangerous events by +2 each, not significant (CI [-1, 6] and [-2, 6]) with 56 dangerous events; only their effect on delta-v (T3: -117 [-256, -1] m/s) reaches significance. T5 ordering vs random at a 20% screening budget: recall 0.371 vs 0.205 (X3). **Efficiency (E52):** screening all 58,825 candidate pairs of 600 LEO objects over 3 d takes 22.4 s on the 1-min grid; the T5-ordered 20% budget (11,765 pairs) takes 5.0 s and reaches 38.8% of the reference approaches, against 20.1% for a random 20% budget at the same cost.
27. **Data ablation on decision metrics (E53, `e53_data_ablation.csv`).** B1 No-maneuver: Starlink 4-h/24-h P90 5.4 -> 39.8 km / 27 -> 113 km when the filtered pool is used without maneuver awareness; Sentinel post-maneuver safety 0.649 -> 0.595 and along-track 95% coverage 0.796 -> 0.717; Starlink triage AUC 0.643 -> 0.602. B2 No-weather: CDM missed dangerous 12 -> 14 (n.s.), Starlink 4-h triage AUC 0.733 -> 0.710 (sig.), 24 h unchanged. B3 No-physical: CDM missed dangerous 12 -> 14 (n.s.), sigma within x2 0.42 -> 0.49. B4 No-precise: without POD-trained correctors the 1-h/4-h forecast P50 is 0.31 -> 0.57 km / 0.38 -> 0.50 km (24 h: 0.53 -> 0.46, correctors hurt); without covariance-calibrated sigma the CDM chain fires 25 vs 16 unnecessary maneuvers and 816 vs 458 m/s; NEES 4.1 -> 5.5. B5 TLE-only: screening recall at 20% budget 0.371 -> 0.205 and 7.7 m/s spent on zero-Pc events; CDM chain 61 unnecessary maneuvers and 1,210 m/s. The pattern matches Sec. 3.4: auxiliary data pay off through uncertainty and priority, and only maneuver awareness changes point-forecast error, and only on maneuvering satellites.

### Same tasks on existing datasets (E54, `e54a_shorten_t2.csv`, `e54b_kelvins_t6.csv`, `e54_dataset_relation.json`)
28. **T2 on the Shorten et al. TLE benchmark (15 satellites, 887 curated maneuvers in the TLE-covered span).** The OrbBench SGP4-consistency detector transfers with the threshold tuned on Sentinel: pooled F1 0.39 / F2 0.55 on their raw TLE files, 0.44 / 0.60 after PAF-lite; with the best threshold on their data 0.49 -> 0.60 (F1) and GEO satellites 0.77 -> 0.82. PAF raises precision at equal or higher recall on every pool (raw 0.38 -> PAF 0.51 at recall 0.70 -> 0.73), i.e. OrbBench's quality control improves the same task on an external dataset. The Sentinel reference numbers (F1 0.43 pooled) sit inside the range of the external satellites, so the OrbBench T2 evaluation is not tuned to its own data.
29. **T6 on the full ESA Kelvins CDM dataset (13,154 training events, 2,167 test events, private final risk as truth).** OrbBench's 199 CDM rows are exactly rows of the Kelvins private test file: the final CDM of 199 test events, comprising all 150 events with final risk >= -6 plus 49 low-risk ones, so the OrbBench T6-CDM/T7 experiments are on the high-risk subset of the Kelvins test set, not on independent data (`e54_dataset_relation.json`). Under the competition protocol (features from the last CDM at least 2 d before TCA; score = MSE on high-risk events / F2), the persistence baseline (latest risk) reaches F2 0.74 with MSE 26.6, whereas every regressor we fit, Ridge or HistGB, on all events or only on events with pre-risk >= -8, with all 105 features or 28 geometry and risk-history features, collapses to F2 <= 0.01: squared-error training over events whose risk mostly vanishes by TCA shrinks high-risk predictions toward -30, and the metric only scores the high-risk events. Within the training set (5-fold CV) the same models also lose to persistence (MSE 53 vs 3.5). This reproduces the competition's own conclusion that the latest-risk baseline is hard to beat, and it is why OrbBench evaluates T6/T7 on decision metrics (missed dangerous events, unnecessary maneuvers, delta-v at true-Pc safety) computed with the CDM covariances rather than on final-risk regression.

### Full chain T1->T7 and upstream connections (E2E, UC; `e2e_*.csv`, `uc_*.csv`)
30. **Staging fix (2026-09-16).** The L2 trajectory products carry a fixed per-object sub-minute timestamp offset (uniform 0-60 s). Stage 0 previously snapped each 1-min sample to the nearest grid minute, i.e. a 50-225 km along-track shift of every cohort reference trajectory. Staging now interpolates position and velocity with cubic Hermite; against SGP4 of the object's own nearest TLE the re-staged references agree to 0-150 m. All cohort-reference results (T5, T6 shell screening, C4, X3, E2E) were rerun; Sentinel, Starlink and CDM results use independent truth and were unaffected. Changes: C4 T5-ordering AUC 0.656 -> 0.651 with 182 (was 139) reference approaches on the 600-object cohort, random 0.519 -> 0.494; X3 recall at 20% budget T5 0.371 -> 0.383 vs random 0.205 -> 0.136 (paired CI on the difference [-0.32, -0.14]); T6 shell recall of independent screening 0.915 (was ~0.17 because the reference itself was displaced), UQ ranking recall@K 0.81-0.88 for all sigma modes (was 0.16-0.17); T5 SGP4-prev density overlap 97.6% (was lower), TV 0.024.
31. **Full chain, 1,500 LEO objects.** Tasks without upstream (T1, T2, T3, T5) and the point forecast of T4 are identical in both modes (24-h P50 0.11 km, P95 3.09 km against the non-causal L2 product of the same TLEs); differences appear only in uncertainty, warning and avoidance. With a 20% screening budget the T5 ordering finds 37.5% of the 953 reference approaches vs 19.6% for random order at the same cost (11.6 vs 10.7 s per epoch; all 385,661 pairs: 96.1% in 50.8 s). Natural conjunctions with true Pc >= 1e-4 are rare (5 in 3 d), so 30 dangerous conjunctions per epoch are injected by shifting one object's reference and forecast trajectories by the same vector (reference miss 20-1000 m, forecast error unchanged): of 123 dangerous injected events the unified pipeline has 39% inside its budget, alerts on 45 and makes 45 safe with 0 unnecessary maneuvers and 275 m/s; independent execution 22% / 26 / 25 / 4 / 414 m/s. On natural approaches the distance rule fires 45 times, 43 of them on events with true Pc < 1e-4; the Pc gate fires 0 times and no natural event reaches the gate. Per-task removal on the same cohort: -T5 -> 22% in budget, 25 alerted (back to independent); -T6 Pc -> 47 alerted but 8 unnecessary on injected + 108 on natural events, 761 m/s (2.8x); -T4 sigma -> 44 alerted, NEES 0.030 -> 0.043; -T1/-T3 no change (quiet week: storm factor never triggered, 26 non-stable objects); -T2 44 safe. The cohort reference is pseudo-truth generated from the same TLEs, so sigma is conservative against it (NEES << 1); the independent-truth sub-chains (X1, X2, X4) carry the significance tests.
32. **Upstream connections outside the task graph.** T3 -> T2 (raise the consistency threshold on days with Ap >= 40): Sentinel storm-day false alarms 70 -> 46, pooled F1 0.433 -> 0.443 at unchanged recall. T1 -> T2 (phase-specific thresholds): no change on Sentinel (99.8% of epochs stable); on Starlink the detection task itself degenerates (84% of TLE intervals inside thrust arcs; 'always thrust' F1 0.91, every detector MCC <= 0.07). T2 -> T1 (T2 flags as phase features): Macro-F1 0.666 unchanged (cohort labels are only STABLE/DECAY). Reading: whether a connection helps is decided by whether the data contain the variation it detects.

## Notes and caveats
- Reference trajectories for T5/T6 are OrbBench L2 trajectories (MTF / nearest SGP4), i.e. pseudo-truth; only T2/T4/E3 use independent POD truth.
- All methods are causal at the forecast origin; Ridge correctors and σ models are fit on 2024-07 → 2024-12 and frozen.
- Deep-learning rows marked 'reported' come from the existing 8-model experiment (2025-H2, pseudo-truth) and are not re-run here (CPU only).
- T1 silver labels derive from rule thresholds calibrated on Starlink; HistGB re-learns the rules (near-perfect ARI) and should be read as an upper bound, the unsupervised rows are the informative ones.
