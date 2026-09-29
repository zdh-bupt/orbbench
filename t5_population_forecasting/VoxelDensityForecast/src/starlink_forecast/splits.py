from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TimeRange:
    start: pd.Timestamp
    end: pd.Timestamp

    @classmethod
    def from_strings(cls, start: str, end: str) -> "TimeRange":
        left = pd.Timestamp(start)
        right = pd.Timestamp(end)
        left = left.tz_localize("UTC") if left.tz is None else left.tz_convert("UTC")
        right = right.tz_localize("UTC") if right.tz is None else right.tz_convert("UTC")
        if left > right:
            raise ValueError("Split start must not be after split end")
        return cls(left, right)


def valid_window_origins(
    times: np.ndarray,
    history_steps: int,
    horizons: tuple[int, ...],
    split: TimeRange | None = None,
    step: pd.Timedelta = pd.Timedelta(hours=1),
) -> np.ndarray:
    """Return origins whose complete causal history and targets are contiguous."""
    utc = pd.DatetimeIndex(pd.to_datetime(times, utc=True))
    if history_steps < 1 or not horizons or min(horizons) < 1:
        raise ValueError("history_steps and horizons must be positive")
    if tuple(sorted(set(horizons))) != horizons:
        raise ValueError("horizons must be unique and strictly increasing")
    if not utc.is_monotonic_increasing or utc.has_duplicates:
        raise ValueError("times must be unique and strictly increasing")
    max_horizon = max(horizons)
    origins: list[int] = []
    for origin in range(history_steps - 1, len(utc) - max_horizon):
        history_start = origin - history_steps + 1
        expected_history = pd.date_range(
            utc[origin] - step * (history_steps - 1),
            periods=history_steps,
            freq=step,
        )
        if not utc[history_start : origin + 1].equals(expected_history):
            continue
        if any(utc[origin + horizon] != utc[origin] + step * horizon for horizon in horizons):
            continue
        if split is not None:
            if utc[history_start] < split.start:
                continue
            if utc[origin + max_horizon] > split.end:
                continue
        origins.append(origin)
    return np.asarray(origins, dtype=np.int64)


def assert_non_overlapping_ranges(ranges: list[TimeRange], purge: pd.Timedelta) -> None:
    ordered = sorted(ranges, key=lambda value: value.start)
    for previous, current in zip(ordered, ordered[1:]):
        if current.start - previous.end < purge:
            raise ValueError(
                f"Split gap {current.start - previous.end} is smaller than purge interval {purge}"
            )


def chronological_fraction_ranges(
    times: np.ndarray,
    fractions: dict[str, float],
    *,
    purge: pd.Timedelta,
) -> dict[str, TimeRange]:
    """Create deterministic chronological ranges with a purge before each boundary."""

    names = tuple(fractions)
    values = np.asarray([float(fractions[name]) for name in names], dtype=np.float64)
    if len(names) < 2 or np.any(values <= 0) or not np.isclose(values.sum(), 1.0):
        raise ValueError("split fractions must be positive and sum to one")
    utc = pd.DatetimeIndex(pd.to_datetime(times, utc=True)).unique().sort_values()
    if len(utc) < len(names) * 2:
        raise ValueError("too few origin times for chronological splitting")
    boundaries: list[pd.Timestamp] = []
    cumulative = np.cumsum(values)[:-1]
    for fraction in cumulative:
        index = min(max(int(np.floor(fraction * len(utc))), 1), len(utc) - 1)
        boundaries.append(utc[index])
    result: dict[str, TimeRange] = {}
    starts = [utc[0], *boundaries]
    ends = [boundary - purge for boundary in boundaries] + [utc[-1]]
    for name, start, end in zip(names, starts, ends):
        if start > end:
            raise ValueError(
                f"split {name} is empty after applying purge={purge}; provide more data or a smaller smoke purge"
            )
        result[name] = TimeRange(start, end)
    assert_non_overlapping_ranges(list(result.values()), purge)
    return result
