# Table 4 assembled from the released result files

## T1 Phase classification

| Method | Macro-F1 | ARI |
|---|---|---|
| Constant stable | 0.315 | 0 |
| PCA + KMeans | 0.701 | 0.794 |
| PELT + KMeans | 0.471 | 0.660 |
| Transformer + KMeans | 0.324 | -0.017 |

_source CSV: tables/table4_t1_assembled.csv_

## T2 Maneuver detection (F1 up, false alarms down)

| Metric | Mode | Sentinel-1A | Sentinel-2A | Sentinel-3A | Overall |
|---|---|---|---|---|---|
| F1 | Independent | 0.411 | 0.467 | 0.479 | 0.433 |
| F1 | Unified | 0.420 | 0.492 | 0.479 | 0.443 |
| FA | Independent | 223 | 67 | 72 | 362 |
| FA | Unified | 207 | 60 | 72 | 339 |

_source CSV: tables/table4_t2_assembled.csv_

## T3 Space weather forecasting (Chronos-2 backtest MAE)

| History (days) | 3d F10.7 | 3d Kp | 3d Ap | 7d F10.7 | 7d Kp | 7d Ap |
|---|---|---|---|---|---|---|
| 30 | 11.763 | 1.033 | 15.302 | 9.839 | 1.337 | 14.336 |
| 60 | 12.001 | 0.911 | 14.028 | 10.244 | 0.980 | 12.003 |
| 90 | 11.225 | 0.878 | 13.550 | 9.548 | 0.861 | 10.956 |

_source CSV: tables/table4_t3_assembled.csv_

## T4 Orbit forecasting: PICP95 of the 24 h along-track interval

| Metric | Mode | Sentinel-1A | Sentinel-2A | Sentinel-2B | Sentinel-3A | Sentinel-3B | Starlink-1047 | Starlink-1017 | Starlink-1046 | All Sentinel | All Starlink | All cohort |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| PICP95 | Independent | 0.748 | 0.825 | 0.880 | 0.881 | 0.846 | 0.933 | 0.934 | 0.721 | 0.835 | 0.857 | 0.924 |
| PICP95 | Unified | 0.797 | 0.846 | 0.887 | 0.897 | 0.861 | 0.982 | 0.958 | 0.775 | 0.858 | 0.909 | 0.932 |

_source CSV: tables/table4_t4_picp_assembled.csv_

## T4 Orbit forecasting: 3D position error P50 at 1 h (km)

| Position error 1h | Sentinel-1A | Sentinel-2A | Sentinel-2B | Sentinel-3A | Sentinel-3B | Starlink-1047 | Starlink-1017 | Starlink-1046 | All Sentinel | All Starlink | All cohort |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LSTM | 88.647 | 21.627 | 21.659 | 26.762 | 26.512 | 2009.427 | 1965.683 | 2869.293 | 31.636 | 2018.519 | 160.984 |
| GRU | 27.286 | 44.803 | 45.004 | 43.807 | 43.907 | 411.298 | 399.177 | 804.892 | 42.350 | 422.608 | 123.782 |
| DLinear | 18.071 | 10.542 | 10.419 | 11.752 | 11.735 | 3249.082 | 3051.496 | 9867.511 | 11.720 | 3276.384 | 44.444 |
| TCN | 79.420 | 40.265 | 40.469 | 44.526 | 44.687 | 1288.822 | 1281.033 | 1915.670 | 43.235 | 1291.933 | 49.232 |
| Informer | 296.501 | 254.178 | 254.172 | 255.715 | 255.742 | 671.598 | 639.646 | 1306.081 | 257.394 | 696.297 | 286.600 |
| FEDformer | 39.918 | 26.443 | 27.015 | 16.590 | 16.654 | 3540.919 | 3356.242 | 7937.932 | 24.058 | 3566.941 | 36.049 |
| Chronos | 129.663 | 191.513 | 186.256 | 45.709 | 260.070 | 139.971 | 74.080 | 217.221 | 186.188 | 103.896 | 142.558 |
| Chronos-2 | 479.979 | 489.365 | 489.482 | 456.101 | 457.092 | 344.221 | 330.436 | 401.696 | 476.574 | 347.830 | 258.736 |

_source CSV: tables/table4_t4_pos_1h_assembled.csv_

## T4 Orbit forecasting: 3D position error P50 at 24 h (km)

| Position error 24h | Sentinel-1A | Sentinel-2A | Sentinel-2B | Sentinel-3A | Sentinel-3B | Starlink-1047 | Starlink-1017 | Starlink-1046 | All Sentinel | All Starlink | All cohort |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LSTM | 575.248 | 224.676 | 224.254 | 177.668 | 186.167 | 9639.718 | 9318.109 | 15834.304 | 257.652 | 9686.395 | 449.323 |
| GRU | 916.149 | 184.316 | 193.838 | 214.773 | 214.649 | 22019.927 | 21394.010 | 30983.511 | 223.738 | 22103.326 | 293.023 |
| DLinear | 187.780 | 88.077 | 89.069 | 100.162 | 100.293 | 7150.091 | 6821.234 | 11333.298 | 107.229 | 7184.446 | 199.569 |
| TCN | 256.573 | 92.900 | 93.065 | 90.884 | 89.849 | 12447.020 | 12224.256 | 12447.729 | 110.584 | 12328.151 | 144.866 |
| Informer | 87.174 | 276.365 | 276.337 | 299.563 | 299.464 | 13746.393 | 13717.269 | 11678.227 | 279.481 | 13704.173 | 276.766 |
| FEDformer | 144.287 | 87.894 | 87.880 | 81.615 | 83.585 | 10064.868 | 9601.867 | 14926.270 | 94.728 | 10104.342 | 158.670 |
| Chronos | 1427.241 | 211.025 | 199.739 | 369.834 | 358.007 | 3101.884 | 3308.573 | 635.969 | 300.684 | 2911.751 | 543.662 |
| Chronos-2 | 3573.512 | 994.368 | 994.455 | 1093.004 | 1094.698 | 3988.130 | 4123.373 | 1735.686 | 1080.559 | 3588.281 | 949.812 |

_source CSV: tables/table4_t4_pos_24h_assembled.csv_

## T5 Population forecasting (+1 h)

| Method | TV | Regional relative MAE |
|---|---|---|
| Last-frame persistence | 0.668 | 1.003 |
| Kepler + CIC | 0.172 | 0.200 |
| Physics + U-Net | 0.184 | 0.222 |

_source CSV: tables/table4_t5_assembled.csv_

## T6 Collision warning (1,500-object LEO cohort, 20 % screening budget)

| Metric | Mode | Payload | Rocket body | Debris | 300-800 | 800-1200 | 1200-2000 | Overall |
|---|---|---|---|---|---|---|---|---|
| ACR (%) | Independent | 18.100 | 17.800 | 21 | 22.800 | 18 | 20.900 | 19.600 |
| ACR (%) | Unified | 47.800 | 36.100 | 31.700 | 28.700 | 39.600 | 40.300 | 37.500 |
| DCR (%) | Independent | 27.500 | 26.667 | 17.647 | 20 | 19.178 | 30 | 21.951 |
| DCR (%) | Unified | 50 | 26.667 | 35.294 | 55.000 | 32.877 | 43.333 | 39.024 |
| DAC | Independent | 11 | 4 | 11 | 4 | 13 | 9 | 26 |
| DAC | Unified | 18 | 4 | 23 | 10 | 23 | 12 | 45 |

_source CSV: tables/table4_t6_assembled.csv_

## T7 Collision avoidance (same cohort)

| Metric | Mode | Payload | Rocket body | Debris | 300-800 | 800-1200 | 1200-2000 | Overall |
|---|---|---|---|---|---|---|---|---|
| DRC | Independent | 11 | 4 | 10 | 4 | 12 | 9 | 25 |
| DRC | Unified | 18 | 4 | 23 | 10 | 23 | 12 | 45 |
| dv_total (m/s) | Independent | 198.300 | 69.300 | 146.400 | 58.800 | 189.000 | 166.200 | 414.000 |
| dv_total (m/s) | Unified | 119.400 | 23.400 | 132.000 | 52.500 | 123.300 | 99.000 | 274.800 |

_source CSV: tables/table4_t7_assembled.csv_
