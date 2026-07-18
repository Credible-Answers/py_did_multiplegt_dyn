"""Utilities for the ``reset`` option.

The option follows the Stata implementation: it keeps treatment-balanced
groups, starts a new estimation group after a treatment spell has lasted the
requested number of periods, and retains the original group as the default
clustering level.
"""

from __future__ import annotations

from numbers import Integral
from typing import Optional, Tuple

import polars as pl


def validate_reset(reset: Optional[int]) -> int:
    """Validate and normalize the number of periods used by ``reset``."""
    if reset is None:
        return 0
    if isinstance(reset, bool) or not isinstance(reset, Integral):
        raise TypeError("`reset` must be a non-negative integer")
    if reset < 0:
        raise ValueError("`reset` must be a non-negative integer")
    return int(reset)


def apply_reset(
    df: pl.DataFrame,
    reset: int,
    *,
    group_col: str = "group_XX",
    time_col: str = "time_XX",
    treatment_col: str = "treatment_XX",
    cluster_col: Optional[str] = None,
) -> Tuple[pl.DataFrame, Optional[str]]:
    """Partition original groups into treatment-spell subgroups.

    Parameters
    ----------
    df
        Data after group and time have been converted to dense numeric IDs.
    reset
        A positive number of unchanged periods after which a new subgroup
        starts. Zero leaves the data unchanged.
    cluster_col
        Existing clustering column. If omitted, the original group identifier
        is copied to ``cluster_XX`` and returned as the clustering column.

    Returns
    -------
    tuple
        The transformed data and the effective clustering-column name.
    """
    reset = validate_reset(reset)
    if reset == 0:
        return df, cluster_col

    # Stata keeps only groups tied for the largest number of non-missing
    # treatment observations. This restricts reset() to complete treatment
    # paths even when the outcome panel itself is unbalanced.
    treatment_counts = df.group_by(group_col).agg(
        pl.col(treatment_col).is_not_null().sum().alias("_reset_n_treatment")
    )
    max_count = treatment_counts.select(pl.col("_reset_n_treatment").max()).item()
    complete_groups = treatment_counts.filter(
        pl.col("_reset_n_treatment") == max_count
    ).select(group_col)
    df = df.join(complete_groups, on=group_col, how="inner")
    if df.is_empty():
        raise ValueError("`reset` removed every group because no complete treatment path exists")

    df = df.sort([group_col, time_col])
    row_in_group = pl.int_range(0, pl.len()).over(group_col)
    df = df.with_columns(
        pl.when(row_in_group == 0)
        .then(0)
        .otherwise(
            (pl.col(treatment_col) != pl.col(treatment_col).shift(1).over(group_col))
            .fill_null(False)
            .cast(pl.Int64)
        )
        .alias("_reset_changed")
    )
    df = df.with_columns(
        pl.col("_reset_changed").cum_sum().over(group_col).alias("_reset_spell")
    )
    df = df.with_columns(
        pl.int_range(0, pl.len())
        .over([group_col, "_reset_spell"])
        .alias("_reset_periods_since_change")
    )
    df = df.with_columns(
        (pl.col("_reset_periods_since_change") == reset)
        .cast(pl.Int64)
        .alias("_reset_start")
    )
    df = df.with_columns(
        pl.col("_reset_start").cum_sum().over(group_col).alias("_reset_count")
    )

    if cluster_col is None:
        df = df.with_columns(pl.col(group_col).alias("cluster_XX"))
        cluster_col = "cluster_XX"

    subgroup_map = (
        df.select([group_col, "_reset_count"])
        .unique()
        .sort([group_col, "_reset_count"])
        .with_row_index("_reset_new_group", offset=1)
    )
    df = (
        df.join(subgroup_map, on=[group_col, "_reset_count"], how="left")
        .drop(group_col)
        .rename({"_reset_new_group": group_col})
        .drop([
            "_reset_changed",
            "_reset_spell",
            "_reset_periods_since_change",
            "_reset_start",
            "_reset_count",
        ])
        .sort([group_col, time_col])
    )
    return df, cluster_col
