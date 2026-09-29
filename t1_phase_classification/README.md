# TLE Observable-State Use Case Package

This directory collects the publication figures, retrospective clustering
results, and the code involved in the TLE observable-state use case.

## Contents

- `figures/tle_state_use_case/`: three final figures in PDF, SVG, and PNG,
  their plotting scripts, the Markdown/LaTeX result tables, and small figure
  configuration CSV files.
- `reports/observable_regime_feature_labels_v2/retest/`: aggregate and
  per-seed evaluation outputs. Row-level `predictions.csv` files are excluded.
- `reports/observable_regime_feature_labels_v2/retest.log`: execution log.
- `scripts/`, `satlife_observable/`, and `tests/`: label-rule, retrospective
  evaluation, plotting, core metric/model, and test code.
- `STARLINK_MOTION_PHASE_TEST_CASE.md`: the observable-state test standard.

## Scope and interpretation

The state rules relabel TLE-observable trajectory shapes inside inherited
legacy candidate segments. They do not perform fully legacy-independent,
end-to-end segmentation. The reported test scores are a retrospective
evaluation after label reorganization, not a new unseen frozen final test.
The labels describe observable trajectory morphology and must not be
interpreted as operational intent or the cause of a maneuver.

As requested, this package contains no label dataset, raw/derived TLE dataset,
row-level prediction file, or model weight/artifact (`.pt`, `.joblib`, `.npz`,
`.npy`). Those external inputs are required to rerun label construction or
model inference. Figure 3 and the result tables can be regenerated from the
packaged aggregate result CSV files; Figures 1 and 2 require the omitted
feature-label dataset.

## Main result files

- `figures/tle_state_use_case/RESULTS_TABLE.md`
- `figures/tle_state_use_case/TABLE_clustering_results.tex`
- `reports/observable_regime_feature_labels_v2/retest/RESULTS.md`
- `reports/observable_regime_feature_labels_v2/retest/summary_by_system.csv`
- `reports/observable_regime_feature_labels_v2/retest/per_class_summary.csv`

The recorded retrospective run used Python 3.13.11, PyTorch 2.11.0+cu128,
NumPy 2.5.2, pandas 3.0.5, and scikit-learn 1.9.0. Plotting also requires
Matplotlib; evaluation imports SciPy, joblib, and ruptures.
