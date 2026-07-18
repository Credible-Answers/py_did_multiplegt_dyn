"""Implementation of the Stata ``avg_time_periods`` option."""

from __future__ import annotations

import math
from typing import Any, Dict, Optional

import numpy as np
import polars as pl


def _present(value: Any) -> bool:
    """Return whether a scalar is usable under Stata-style missing semantics."""
    if value is None:
        return False
    try:
        return not math.isnan(float(value))
    except (TypeError, ValueError):
        return True


def compute_avg_time_periods(
    df: pl.DataFrame,
    l_XX: int,
    *,
    same_switchers: bool = False,
    switchers: str = "",
    continuous: int = 0,
    group_col: str = "group_XX",
    time_col: str = "time_XX",
    treatment_col: str = "treatment_XX",
    outcome_col: str = "outcome_XX",
    first_switch_col: str = "F_g_XX",
    control_horizon_col: str = "T_g_XX",
    baseline_class_col: str = "d_sq_int_XX",
    switch_direction_col: str = "S_g_XX",
    cell_weight_col: str = "N_gt_XX",
) -> Dict[str, Any]:
    """Compute the average duration over which incremental doses accumulate.

    This is a direct Python translation of the Stata ``_compute_Tg`` and
    ``_avg_cumul`` Mata workers. In particular, each switcher gets its own last
    usable horizon, determined by the presence of another not-yet-switching
    group in the same baseline-treatment class with observed outcomes at both
    the reference and horizon dates.
    """
    if isinstance(l_XX, bool) or int(l_XX) < 1:
        raise ValueError("`avg_time_periods` requires at least one effect")

    required = {
        group_col,
        time_col,
        treatment_col,
        outcome_col,
        first_switch_col,
        control_horizon_col,
        baseline_class_col,
        switch_direction_col,
        cell_weight_col,
    }
    baseline_col = "d_sq_XX_orig" if continuous > 0 else "d_sq_XX"
    required.add(baseline_col)
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(
            "Cannot compute `avg_time_periods`; required internal columns are missing: "
            + ", ".join(missing)
        )

    columns = list(required)
    compact = df.select(columns).sort([group_col, time_col])
    groups: Dict[Any, Dict[str, Any]] = {}
    for part in compact.partition_by(group_col, maintain_order=True):
        group_id = part[group_col][0]
        rows = {}
        for row in part.iter_rows(named=True):
            if _present(row[time_col]):
                rows[int(row[time_col])] = row
        first = part.row(0, named=True)
        groups[group_id] = {
            "first_switch": first[first_switch_col],
            "control_horizon": first[control_horizon_col],
            "baseline_class": first[baseline_class_col],
            "baseline": first[baseline_col],
            "direction": first[switch_direction_col],
            "rows": rows,
        }

    if not groups:
        return {
            "avg_cumul": np.nan,
            "N_switch_avg": {f"Effect_{k}": 0.0 for k in range(1, l_XX + 1)},
            "numerator": 0.0,
            "denominator": 0.0,
        }

    max_time = max(
        time_value
        for group_data in groups.values()
        for time_value in group_data["rows"]
    )

    def valid_outcome(group_data: Dict[str, Any], period: int) -> bool:
        row = group_data["rows"].get(period)
        return row is not None and _present(row[outcome_col])

    def has_control(
        switcher_id: Any,
        baseline_class: Any,
        reference_period: int,
        horizon: int,
    ) -> bool:
        for control_id, control in groups.items():
            if control_id == switcher_id or control["baseline_class"] != baseline_class:
                continue
            control_first_switch = control["first_switch"]
            if not _present(control_first_switch) or control_first_switch <= horizon:
                continue
            if valid_outcome(control, reference_period) and valid_outcome(control, horizon):
                return True
        return False

    # First Mata worker: switcher-specific last valid horizon and M_g.
    for group_id, group_data in groups.items():
        first_switch = group_data["first_switch"]
        control_horizon = group_data["control_horizon"]
        if not _present(first_switch):
            group_data["last_horizon"] = None
            group_data["M_g"] = 0
            continue
        first_switch = int(first_switch)
        reference_period = first_switch - 1
        last_horizon = reference_period
        ever_switcher = first_switch < max_time + 1
        if ever_switcher and _present(control_horizon) and valid_outcome(group_data, reference_period):
            for horizon in range(first_switch, int(control_horizon) + 1):
                if not valid_outcome(group_data, horizon):
                    continue
                if has_control(
                    group_id,
                    group_data["baseline_class"],
                    reference_period,
                    horizon,
                ):
                    last_horizon = horizon
        group_data["last_horizon"] = last_horizon
        group_data["M_g"] = min(l_XX, last_horizon - first_switch + 1)

    numerator = 0.0
    denominator = 0.0
    switcher_counts = np.zeros(l_XX, dtype=float)

    # Second Mata worker: dose-weighted average and per-effect counts.
    for group_id, group_data in groups.items():
        first_switch = group_data["first_switch"]
        last_horizon = group_data.get("last_horizon")
        M_g = int(group_data.get("M_g", 0))
        direction = group_data["direction"]
        if not _present(first_switch) or last_horizon is None:
            continue
        first_switch = int(first_switch)
        if first_switch >= max_time + 1 or first_switch > last_horizon:
            continue
        if same_switchers and M_g < l_XX:
            continue
        if switchers == "in" and direction != 1:
            continue
        if switchers == "out" and direction != 0:
            continue

        reference_period = first_switch - 1
        reference_row = group_data["rows"].get(reference_period)
        if reference_row is None or not valid_outcome(group_data, reference_period):
            continue

        for k in range(M_g):
            horizon = first_switch + k
            row = group_data["rows"].get(horizon)
            if row is None or not valid_outcome(group_data, horizon):
                continue
            treatment_value = row[treatment_col]
            baseline_value = group_data["baseline"]
            if not _present(treatment_value) or not _present(baseline_value):
                continue
            if not has_control(
                group_id,
                group_data["baseline_class"],
                reference_period,
                horizon,
            ):
                continue

            increment = abs(float(treatment_value) - float(baseline_value))
            numerator += increment * (M_g - k)
            denominator += increment
            cell_weight = row[cell_weight_col]
            if _present(cell_weight):
                switcher_counts[k] += float(cell_weight)

    avg_cumul = numerator / denominator if denominator != 0 else np.nan
    return {
        "avg_cumul": float(avg_cumul),
        "N_switch_avg": {
            f"Effect_{k}": float(switcher_counts[k - 1])
            for k in range(1, l_XX + 1)
        },
        "numerator": float(numerator),
        "denominator": float(denominator),
    }
