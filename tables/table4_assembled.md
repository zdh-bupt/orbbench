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
| LSTM | 88.942 | 21.419 | 21.461 | 26.718 | 18.930 | 2007.609 | 1963.256 | 2871.720 | 29.124 | 2010.034 | 129.663 |
| GRU | 25.083 | 44.977 | 45.589 | 44.089 | 43.996 | 411.353 | 400.067 | 811.944 | 42.685 | 418.648 | 99.557 |
| DLinear | 15.965 | 10.060 | 11.138 | 11.853 | 11.455 | 3244.360 | 3063.197 | 10328.512 | 11.560 | 3236.496 | 35.296 |
| TCN | 82.268 | 42.017 | 42.244 | 44.844 | 43.570 | 1288.096 | 1281.024 | 1956.958 | 43.143 | 1290.259 | 48.308 |
| Informer | 296.501 | 255.261 | 254.700 | 255.797 | 253.384 | 674.205 | 636.097 | 1337.514 | 256.683 | 684.050 | 287.112 |
| FEDformer | 39.443 | 25.759 | 25.761 | 16.391 | 18.587 | 3536.812 | 3355.523 | 8164.914 | 22.217 | 3530.191 | 30.453 |
| Chronos | 129.663 | 191.513 | 186.256 | 45.709 | 260.070 | 139.971 | 74.080 | 217.221 | 186.188 | 103.896 | 142.558 |
| Chronos-2 | 484.020 | 474.045 | 476.265 | 451.584 | 512.119 | 344.543 | 325.266 | 395.722 | 473.998 | 344.634 | 273.448 |

_source CSV: tables/table4_t4_pos_1h_assembled.csv_

## T4 Orbit forecasting: 3D position error P50 at 24 h (km)

| Position error 24h | Sentinel-1A | Sentinel-2A | Sentinel-2B | Sentinel-3A | Sentinel-3B | Starlink-1047 | Starlink-1017 | Starlink-1046 | All Sentinel | All Starlink | All cohort |
|---|---|---|---|---|---|---|---|---|---|---|---|
| LSTM | 575.328 | 208.891 | 224.008 | 188.598 | 139.050 | 9635.310 | 9327.503 | 15968.133 | 236.420 | 9624.623 | 417.439 |
| GRU | 917.621 | 215.036 | 162.357 | 209.774 | 226.064 | 22001.951 | 21446.485 | 31024.115 | 225.426 | 21986.723 | 298.215 |
| DLinear | 184.959 | 92.131 | 73.880 | 105.709 | 97.900 | 7099.564 | 6820.660 | 11345.237 | 105.560 | 7097.774 | 178.023 |
| TCN | 254.418 | 93.404 | 89.270 | 92.417 | 80.060 | 12426.882 | 12250.217 | 12373.909 | 104.503 | 12313.127 | 150.069 |
| Informer | 83.544 | 277.353 | 277.464 | 300.291 | 294.699 | 13747.272 | 13717.729 | 11631.776 | 279.860 | 13711.951 | 272.207 |
| FEDformer | 127.100 | 88.173 | 77.896 | 88.948 | 79.666 | 10002.783 | 9573.873 | 14906.681 | 89.883 | 9993.543 | 149.162 |
| Chronos | 1427.241 | 211.025 | 199.739 | 369.834 | 358.007 | 3101.884 | 3308.573 | 635.969 | 300.684 | 2911.751 | 543.662 |
| Chronos-2 | 3633.291 | 979.958 | 1009.395 | 1086.481 | 1205.078 | 4000.450 | 4103.069 | 1678.614 | 1080.794 | 3718.477 | 975.733 |

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
