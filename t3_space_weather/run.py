"""Run the corrected Chronos-2 space-weather experiment in an isolated folder.

The source daily Parquet has the F10.7 observed/adjusted labels reversed.  This
runner leaves the source file untouched and maps the source ``f107_adj`` column
to the scientifically intended ``f107_obs`` target before forecasting.
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


import os

SCRIPT_DIR = Path(__file__).resolve().parent
RUN_DIR = SCRIPT_DIR                      # outputs (forecasts/, backtests/, metrics/, figures/) are written next to this script
RELEASE_ROOT = Path(os.environ.get("ORBBENCH_RELEASE_ROOT", str(SCRIPT_DIR.parent)))
# Inputs: the daily space-weather table and the flux measurement table (shipped in <release>/data/space_weather/) and the
# GFZ definitive Kp/Ap/F10.7 rows for 2025-06-25..07-01 used by the formal-forecast validation. Override with env vars.
DATA_PATH = Path(os.environ.get("ORBBENCH_SW_PARQUET", str(RELEASE_ROOT / "data" / "space_weather" / "space_weather_daily.parquet")))
FLUX_PATH = Path(os.environ.get("ORBBENCH_SW_FLUX_PARQUET", str(RELEASE_ROOT / "data" / "space_weather" / "solar_flux_measurements.parquet")))
TRUTH_PATH = Path(os.environ.get("ORBBENCH_GFZ_TRUTH", str(SCRIPT_DIR / "GFZ_true_20250625_20250701.txt")))

FORECAST_DIR = RUN_DIR / "forecasts"
BACKTEST_DIR = RUN_DIR / "backtests"
METRICS_DIR = RUN_DIR / "metrics"
FIG_COMBINED_DIR = RUN_DIR / "figures" / "combined"
FIG_INDIVIDUAL_DIR = RUN_DIR / "figures" / "individual"
CHECK_DIR = RUN_DIR / "data_checks"
LOG_DIR = RUN_DIR / "logs"

MODEL_NAME = "amazon/chronos-2"
DEVICE = "cpu"
CONTEXTS = [30, 60, 90]
HORIZONS = [3, 7]
VARIABLES = ["f107_obs", "kp_daily_avg", "ap_avg"]
QUANTILES = [0.1, 0.25, 0.5, 0.75, 0.9]
LABELS = {
    "f107_obs": ("F10.7 observed", "sfu", "f107"),
    "kp_daily_avg": ("Kp daily mean", "index", "kp"),
    "ap_avg": ("Ap daily mean", "index", "ap"),
}


def setup_logging() -> None:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(LOG_DIR / "rerun.log", mode="w", encoding="utf-8"),
    ]
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", handlers=handlers)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_directories() -> None:
    for directory in [FORECAST_DIR, BACKTEST_DIR, METRICS_DIR, FIG_COMBINED_DIR,
                      FIG_INDIVIDUAL_DIR, CHECK_DIR, LOG_DIR]:
        directory.mkdir(parents=True, exist_ok=True)


def load_corrected_data() -> pd.DataFrame:
    raw = pd.read_parquet(DATA_PATH)
    obs = raw.loc[raw["data_section"].eq("observed")].copy()
    kp_cols = [
        "kp_00_03", "kp_03_06", "kp_06_09", "kp_09_12",
        "kp_12_15", "kp_15_18", "kp_18_21", "kp_21_24",
    ]
    obs["kp_daily_avg"] = obs[kp_cols].mean(axis=1)
    obs["date"] = pd.to_datetime(obs["date"])

    # Confirmed source-label defect: daily.f107_adj contains the observed flux.
    obs["f107_obs_corrected"] = obs["f107_adj"]
    clean = (
        obs[["date", "f107_obs_corrected", "kp_daily_avg", "ap_avg"]]
        .rename(columns={"f107_obs_corrected": "f107_obs"})
        .dropna()
        .sort_values("date")
        .drop_duplicates("date", keep="last")
        .reset_index(drop=True)
    )
    if clean["date"].duplicated().any():
        raise ValueError("Corrected daily data contains duplicate dates")
    if not clean["date"].is_monotonic_increasing:
        raise ValueError("Corrected daily data is not chronological")
    logging.info("Corrected data: %d rows, %s through %s", len(clean), clean.date.min().date(), clean.date.max().date())
    return clean


def write_field_audit(clean: pd.DataFrame) -> None:
    raw = pd.read_parquet(DATA_PATH)
    obs = raw.loc[raw["data_section"].eq("observed"), ["date", "f107_obs", "f107_adj"]].copy()
    obs["date"] = pd.to_datetime(obs["date"])
    obs = obs.dropna().sort_values("date").drop_duplicates("date", keep="last")

    flux = pd.read_parquet(FLUX_PATH)[["date", "fluxobsflux", "fluxadjflux"]].copy()
    flux["date"] = pd.to_datetime(flux["date"])
    flux = flux.dropna().sort_values("date")
    # The measurement table usually has three observations per date.  A daily
    # record should match any corresponding measurement, not an arbitrary last
    # row of that date.
    joined_all = obs.merge(flux, on="date", how="inner", validate="one_to_many")
    tol = 1e-6
    joined_all["daily_obs_matches_flux_adjusted"] = np.isclose(joined_all["f107_obs"], joined_all["fluxadjflux"], atol=tol)
    joined_all["daily_adj_matches_flux_observed"] = np.isclose(joined_all["f107_adj"], joined_all["fluxobsflux"], atol=tol)
    joined_all["swap_pair_matches"] = joined_all["daily_obs_matches_flux_adjusted"] & joined_all["daily_adj_matches_flux_observed"]
    match_flags = joined_all.groupby("date", as_index=False)[
        ["daily_obs_matches_flux_adjusted", "daily_adj_matches_flux_observed", "swap_pair_matches"]
    ].any()
    joined = obs.merge(match_flags, on="date", how="inner", validate="one_to_one")
    joined["corrected_f107_obs"] = joined["f107_adj"]
    joined.to_csv(CHECK_DIR / "field_mapping_validation.csv", index=False, encoding="utf-8-sig")

    summary = {
        "source_daily_parquet": str(DATA_PATH),
        "source_flux_parquet": str(FLUX_PATH),
        "daily_parquet_sha256": sha256(DATA_PATH),
        "flux_parquet_sha256": sha256(FLUX_PATH),
        "overlap_rows": int(len(joined)),
        "swap_pair_matches": int(joined["swap_pair_matches"].sum()),
        "swap_pair_match_rate": float(joined["swap_pair_matches"].mean()),
        "applied_mapping": "model f107_obs <- source daily f107_adj",
        "source_files_modified": False,
        "model_input_rows": int(len(clean)),
        "model_input_start": clean.date.min().date().isoformat(),
        "model_input_end": clean.date.max().date().isoformat(),
    }
    (CHECK_DIR / "field_mapping_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    logging.info("Field audit: %d/%d overlapping rows match the swapped pair", summary["swap_pair_matches"], summary["overlap_rows"])


def to_long(data: pd.DataFrame) -> pd.DataFrame:
    return (
        data.melt(id_vars=["date"], var_name="item_id", value_name="target")
        .rename(columns={"date": "timestamp"})
        .sort_values(["item_id", "timestamp"])
        .reset_index(drop=True)
    )


def predict(pipeline, context_long: pd.DataFrame, prediction_length: int) -> pd.DataFrame:
    return pipeline.predict_df(
        context_long,
        prediction_length=prediction_length,
        quantile_levels=QUANTILES,
        id_column="item_id",
        timestamp_column="timestamp",
        target="target",
        cross_learning=True,
        freq="D",
    )


def save_forecast(pred: pd.DataFrame, path: Path) -> pd.DataFrame:
    wide = pred.pivot(index="timestamp", columns="item_id", values="0.5").reset_index()
    wide.columns.name = None
    for variable in VARIABLES:
        quantile_frame = pred.loc[pred["item_id"].eq(variable), ["timestamp"] + [str(q) for q in QUANTILES]].copy()
        quantile_frame = quantile_frame.rename(columns={str(q): f"{variable}_p{int(q * 100)}" for q in QUANTILES})
        wide = wide.merge(quantile_frame, on="timestamp", how="left", validate="one_to_one")
    wide.to_csv(path, index=False, encoding="utf-8-sig")
    return wide


def plot_forecast(history: pd.DataFrame, pred: pd.DataFrame, context: int, horizon: int, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(15, 11))
    fig.suptitle(f"Chronos 2 corrected space-weather forecast: context {context} days, horizon {horizon} days")
    for ax, variable in zip(axes, VARIABLES):
        label, unit, _ = LABELS[variable]
        hist = history.loc[history["date"].ge(history.date.max() - pd.Timedelta(days=context * 2)), ["date", variable]]
        series = pred.loc[pred["item_id"].eq(variable)].sort_values("timestamp")
        ax.plot(hist["date"], hist[variable], color="#4C78A8", linewidth=1, label="history")
        ax.plot(series["timestamp"], series["0.5"], color="#F58518", marker="o", label="p50 forecast")
        ax.fill_between(series["timestamp"], series["0.1"], series["0.9"], color="#F58518", alpha=.14, label="p10-p90")
        ax.set_title(label)
        ax.set_ylabel(unit)
        ax.grid(alpha=.25)
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def run_forecasts(pipeline, data: pd.DataFrame) -> pd.DataFrame:
    data_long = to_long(data)
    final_wide = None
    for context in CONTEXTS:
        context_start = data.date.max() - pd.Timedelta(days=context - 1)
        context_long = data_long.loc[data_long["timestamp"].ge(context_start)]
        for horizon in HORIZONS:
            tag = f"ctx{context}_pred{horizon}"
            logging.info("Formal forecast %s", tag)
            pred = predict(pipeline, context_long, horizon)
            wide = save_forecast(pred, FORECAST_DIR / f"forecast_{tag}.csv")
            plot_forecast(data, pred, context, horizon, FORECAST_DIR / f"forecast_{tag}.png")
            if context == 90 and horizon == 7:
                final_wide = wide
                wide.to_csv(FORECAST_DIR / "space_weather_forecast_final.csv", index=False, encoding="utf-8-sig")
                plot_forecast(data, pred, context, horizon, FORECAST_DIR / "forecast_final_recommended.png")
    if final_wide is None:
        raise RuntimeError("Recommended forecast was not generated")
    return final_wide


def summarize(data: pd.DataFrame, group_columns: list[str]) -> pd.DataFrame:
    records = []
    for keys, group in data.groupby(group_columns, sort=True):
        keys = keys if isinstance(keys, tuple) else (keys,)
        record = dict(zip(group_columns, keys))
        record.update({
            "n": len(group),
            "mae": group["abs_error"].mean(),
            "mse": group["squared_error"].mean(),
            "rmse": np.sqrt(group["squared_error"].mean()),
            "bias": group["error"].mean(),
            "median_abs_error": group["abs_error"].median(),
            "p90_abs_error": group["abs_error"].quantile(.9),
            "mape_percent": group["ape_percent"].mean(),
            "coverage_50": ((group["true"] >= group["p25"]) & (group["true"] <= group["p75"])).mean(),
            "coverage_80": ((group["true"] >= group["p10"]) & (group["true"] <= group["p90"])).mean(),
        })
        records.append(record)
    result = pd.DataFrame(records)
    numeric = result.select_dtypes(include=[np.number]).columns
    result[numeric] = result[numeric].round(6)
    return result


def run_backtests(pipeline, data: pd.DataFrame) -> pd.DataFrame:
    data_long = to_long(data)
    all_dates = sorted(data["date"].unique())
    frames = []
    nested_metrics = {}
    for context in CONTEXTS:
        for horizon in HORIZONS:
            tag = f"ctx{context}_pred{horizon}"
            logging.info("Backtest %s", tag)
            test_ends = sorted([all_dates[len(all_dates) - 1 - i * horizon] for i in range(10)])
            rows = []
            window_metrics = {v: [] for v in VARIABLES}
            for window, test_end_raw in enumerate(test_ends):
                test_end = pd.Timestamp(test_end_raw)
                test_start = test_end - pd.Timedelta(days=horizon - 1)
                context_end = test_start - pd.Timedelta(days=1)
                context_start = context_end - pd.Timedelta(days=context - 1)
                context_long = data_long.loc[data_long["timestamp"].between(context_start, context_end)]
                pred = predict(pipeline, context_long, horizon)
                truth = data.loc[data["date"].between(test_start, test_end)]
                for variable in VARIABLES:
                    joined = pred.loc[pred["item_id"].eq(variable)].merge(
                        truth[["date", variable]], left_on="timestamp", right_on="date", how="inner", validate="one_to_one"
                    )
                    if len(joined) != horizon:
                        raise ValueError(f"{tag} window {window} {variable}: expected {horizon} truth rows, got {len(joined)}")
                    for lead, (_, row) in enumerate(joined.sort_values("timestamp").iterrows(), 1):
                        true = float(row[variable])
                        predicted = float(row["0.5"])
                        error = predicted - true
                        rows.append({
                            "window": window, "lead": lead, "variable": variable,
                            "date": row["timestamp"], "true": true, "pred": predicted,
                            "error": error, "abs_error": abs(error), "squared_error": error ** 2,
                            "ape_percent": abs(error) / abs(true) * 100 if true != 0 else np.nan,
                            "p10": float(row["0.1"]), "p25": float(row["0.25"]),
                            "p50": predicted, "p75": float(row["0.75"]), "p90": float(row["0.9"]),
                        })
                    errors = joined["0.5"] - joined[variable]
                    window_metrics[variable].append({
                        "mae": float(errors.abs().mean()),
                        "rmse": float(np.sqrt((errors ** 2).mean())),
                        "coverage_50": float(((joined[variable] >= joined["0.25"]) & (joined[variable] <= joined["0.75"])).mean()),
                        "coverage_80": float(((joined[variable] >= joined["0.1"]) & (joined[variable] <= joined["0.9"])).mean()),
                    })
            frame = pd.DataFrame(rows)
            frame.to_csv(BACKTEST_DIR / f"backtest_predictions_{tag}.csv", index=False, encoding="utf-8-sig")
            frames.append(frame.assign(config=tag, context_days=context, horizon_days=horizon))
            nested_metrics[tag] = {}
            for variable, values in window_metrics.items():
                vm = pd.DataFrame(values)
                nested_metrics[tag][variable] = {
                    "MAE_mean": round(float(vm.mae.mean()), 6), "MAE_std": round(float(vm.mae.std(ddof=0)), 6),
                    "RMSE_mean": round(float(vm.rmse.mean()), 6), "RMSE_std": round(float(vm.rmse.std(ddof=0)), 6),
                    "50%CI_Coverage_mean": round(float(vm.coverage_50.mean()), 6),
                    "80%CI_Coverage_mean": round(float(vm.coverage_80.mean()), 6),
                }
    (METRICS_DIR / "backtest_metrics_summary.json").write_text(
        json.dumps(nested_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    trace = pd.concat(frames, ignore_index=True)
    trace = trace[["config", "context_days", "horizon_days", "window", "lead", "variable", "date",
                   "true", "pred", "error", "abs_error", "squared_error", "ape_percent",
                   "p10", "p25", "p50", "p75", "p90"]]
    trace = trace.sort_values(["horizon_days", "context_days", "window", "variable", "date"]).reset_index(drop=True)
    trace.round(6).to_csv(BACKTEST_DIR / "chronos_config_trace.csv", index=False, encoding="utf-8-sig")
    metrics = summarize(trace, ["config", "context_days", "horizon_days", "variable"])
    metrics.to_csv(METRICS_DIR / "chronos_config_metrics.csv", index=False, encoding="utf-8-sig")
    return trace


def load_gfz_truth() -> pd.DataFrame:
    names = (["year", "month", "day", "days", "days_mid", "bsr", "bsr_day"]
             + [f"kp_{i}" for i in range(1, 9)] + [f"ap_{i}" for i in range(1, 9)]
             + ["ap_daily", "sunspot_number", "f107_observed", "f107_adjusted", "status"])
    truth = pd.read_csv(TRUTH_PATH, sep=r"\s+", header=None, names=names)
    truth["date"] = pd.to_datetime(truth[["year", "month", "day"]])
    truth["kp_daily_avg"] = truth[[f"kp_{i}" for i in range(1, 9)]].mean(axis=1)
    truth = truth.rename(columns={"f107_observed": "f107_obs", "ap_daily": "ap_avg"})
    if not truth["status"].eq(2).all():
        raise ValueError("GFZ validation file contains non-definitive values")
    return truth


def validate_formal_forecast(final_wide: pd.DataFrame) -> None:
    truth = load_gfz_truth()
    forecast = final_wide.copy()
    forecast["timestamp"] = pd.to_datetime(forecast["timestamp"])
    joined = forecast.merge(truth[["date", "f107_obs", "kp_daily_avg", "ap_avg", "status"]],
                            left_on="timestamp", right_on="date", how="inner", suffixes=("_pred", "_true"), validate="one_to_one")
    if len(joined) != len(forecast):
        raise ValueError("Formal forecast dates do not fully match GFZ truth")
    rows = []
    for variable in VARIABLES:
        for lead, (_, row) in enumerate(joined.sort_values("timestamp").iterrows(), 1):
            true = float(row[f"{variable}_true"])
            predicted = float(row[f"{variable}_pred"])
            error = predicted - true
            rows.append({
                "config": "ctx90_pred7", "context_days": 90, "horizon_days": 7, "lead": lead,
                "date": row["timestamp"], "variable": variable, "true": true, "pred": predicted,
                "error": error, "abs_error": abs(error), "squared_error": error ** 2,
                "ape_percent": abs(error) / abs(true) * 100 if true != 0 else np.nan,
                "p10": float(row[f"{variable}_p10"]), "p25": float(row[f"{variable}_p25"]),
                "p50": float(row[f"{variable}_p50"]), "p75": float(row[f"{variable}_p75"]),
                "p90": float(row[f"{variable}_p90"]), "truth_status": int(row["status"]),
            })
    validation = pd.DataFrame(rows)
    validation.round(6).to_csv(FORECAST_DIR / "formal_forecast_validation.csv", index=False, encoding="utf-8-sig")
    summarize(validation, ["config", "variable"]).to_csv(
        METRICS_DIR / "formal_forecast_metrics.csv", index=False, encoding="utf-8-sig"
    )


def plot_distributions(trace: pd.DataFrame) -> None:
    colors = {30: "#4C78A8", 60: "#F58518", 90: "#54A24B"}
    fig, axes = plt.subplots(3, 4, figsize=(19, 12))
    for row_index, variable in enumerate(VARIABLES):
        label, unit, stem = LABELS[variable]
        for horizon_index, horizon in enumerate(HORIZONS):
            subset = trace.loc[(trace.variable == variable) & (trace.horizon_days == horizon)]
            x_grid = np.linspace(0, max(1e-9, subset.abs_error.max() * 1.05), 300)
            for plot_type_index, plot_type in enumerate(["pdf", "cdf"]):
                ax = axes[row_index, horizon_index * 2 + plot_type_index]
                individual_fig, individual_ax = plt.subplots(figsize=(7.2, 5.2))
                for context in CONTEXTS:
                    values = subset.loc[subset.context_days == context, "abs_error"].dropna().to_numpy()
                    if plot_type == "pdf":
                        from scipy.stats import gaussian_kde
                        y = gaussian_kde(values)(x_grid)
                        ax.plot(x_grid, y, lw=2, color=colors[context], label=f"ctx{context}")
                        individual_ax.plot(x_grid, y, lw=2.2, color=colors[context], label=f"ctx{context}")
                    else:
                        x = np.sort(values)
                        y = np.arange(1, len(x) + 1) / len(x)
                        ax.step(x, y, where="post", lw=2, color=colors[context], label=f"ctx{context}")
                        individual_ax.step(x, y, where="post", lw=2.2, color=colors[context], label=f"ctx{context}")
                title = f"{label}, {horizon}-day horizon: {plot_type.upper()}"
                for target_ax in [ax, individual_ax]:
                    target_ax.set_title(title)
                    target_ax.set_xlabel(f"Absolute error ({unit})")
                    target_ax.set_ylabel("Density" if plot_type == "pdf" else "Cumulative probability")
                    if plot_type == "cdf":
                        target_ax.set_ylim(0, 1.02)
                    target_ax.grid(alpha=.25)
                    target_ax.spines[["top", "right"]].set_visible(False)
                individual_ax.legend(title="Context length", frameon=False)
                individual_fig.tight_layout()
                individual_fig.savefig(FIG_INDIVIDUAL_DIR / f"{stem}_pred{horizon}_{plot_type}.png",
                                       dpi=300, bbox_inches="tight", facecolor="white")
                plt.close(individual_fig)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.suptitle("Chronos 2 corrected absolute-error distributions", y=.995)
    fig.legend(handles, legend_labels, loc="upper center", bbox_to_anchor=(.5, .965),
               ncol=3, frameon=False, title="Context length")
    fig.subplots_adjust(left=.055, right=.99, bottom=.06, top=.89, wspace=.22, hspace=.35)
    fig.savefig(FIG_COMBINED_DIR / "chronos_error_cdf_pdf.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def verify_outputs(trace: pd.DataFrame) -> None:
    expected_trace_rows = sum(3 * 10 * h * 3 for h in HORIZONS)
    checks = {
        "trace_rows_expected": expected_trace_rows,
        "trace_rows_actual": int(len(trace)),
        "trace_row_count_ok": len(trace) == expected_trace_rows,
        "trace_missing_values": int(trace.isna().sum().sum()),
        "trace_duplicate_keys": int(trace.duplicated(["config", "window", "lead", "variable"]).sum()),
        "formal_validation_rows": int(len(pd.read_csv(FORECAST_DIR / "formal_forecast_validation.csv"))),
        "individual_figure_count": len(list(FIG_INDIVIDUAL_DIR.glob("*.png"))),
        "forecast_csv_count": len(list(FORECAST_DIR.glob("forecast_ctx*_pred*.csv"))),
        "backtest_csv_count": len(list(BACKTEST_DIR.glob("backtest_predictions_ctx*_pred*.csv"))),
    }
    checks["all_checks_pass"] = (
        checks["trace_row_count_ok"] and checks["trace_missing_values"] == 0
        and checks["trace_duplicate_keys"] == 0 and checks["formal_validation_rows"] == 21
        and checks["individual_figure_count"] == 12 and checks["forecast_csv_count"] == 6
        and checks["backtest_csv_count"] == 6
    )
    (CHECK_DIR / "output_verification.json").write_text(
        json.dumps(checks, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not checks["all_checks_pass"]:
        raise RuntimeError(f"Output verification failed: {checks}")
    logging.info("Output verification passed: %s", checks)


def main() -> None:
    prepare_directories()
    setup_logging()
    logging.info("Starting isolated corrected rerun in %s", RUN_DIR)
    data = load_corrected_data()
    write_field_audit(data)

    from chronos import Chronos2Pipeline
    logging.info("Loading model %s on %s", MODEL_NAME, DEVICE)
    pipeline = Chronos2Pipeline.from_pretrained(MODEL_NAME, device_map=DEVICE)

    final_wide = run_forecasts(pipeline, data)
    trace = run_backtests(pipeline, data)
    validate_formal_forecast(final_wide)
    plot_distributions(trace)
    verify_outputs(trace)
    logging.info("Corrected rerun completed")


if __name__ == "__main__":
    main()
