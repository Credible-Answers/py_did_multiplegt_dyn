"""
normalized_weights option for did_multiplegt_dyn.

This module implements the normalized_weights option which reports the weights
that normalized effects place on current and lagged treatments.
"""

from __future__ import annotations

import warnings
from typing import Optional, Dict, Any, List

import polars as pl
import pandas as pd
import numpy as np


def compute_normalized_weights(
    df: pl.DataFrame,
    l_XX: int,
    group_col: str = "group_XX",
    time_col: str = "time_XX",
    treatment_col: str = "treatment_XX",
    f_g_col: str = "F_g_XX",
    t_g_col: str = "T_g_XX",
    d_sq_col: str = "d_sq_XX",
    n_gt_col: str = "N_gt_XX"
) -> pd.DataFrame:
    """
    Compute the weight matrix for normalized effects.

    For each effect l and lag k, computes the weight that the normalized
    effect places on the treatment change at relative time k.

    Parameters
    ----------
    df : pl.DataFrame
        Data with required columns
    l_XX : int
        Number of effects
    group_col, time_col, treatment_col, f_g_col, t_g_col, d_sq_col, n_gt_col : str
        Column names

    Returns
    -------
    pd.DataFrame
        Weight matrix with rows = lags (k=0, k=1, ..., Total)
        and columns = effects (l=1, l=2, ...)
    """
    # Initialize weight matrix
    # Rows: k=0, k=1, ..., k=(l-1) for each effect, plus Total row
    # Columns: l=1, l=2, ..., l=l_XX
    weight_matrix = np.zeros((l_XX, l_XX))

    # For each effect l
    for l in range(1, l_XX + 1):
        # Get observations at time F_g - 1 + l (effect l period)
        df_effect_l = df.filter(
            (pl.col(time_col) == pl.col(f_g_col) - 1 + l) &
            (l <= (pl.col(t_g_col) - pl.col(f_g_col) + 1))
        )

        if df_effect_l.height == 0:
            continue

        # Compute N_gt for effect l
        n_gt_l = df_effect_l.group_by(group_col).agg([
            pl.col(n_gt_col).first().alias("N_gt_l")
        ])

        # Merge back
        df_effect_l = df_effect_l.join(n_gt_l, on=group_col, how="left")

        # Compute delta_D_l (total treatment difference for effect l)
        delta_D_l = df_effect_l.select([
            (pl.col(treatment_col) - pl.col(d_sq_col)).abs() * pl.col("N_gt_l")
        ]).sum().item()

        if delta_D_l == 0:
            continue

        # N_switchers for effect l
        n_switchers_l = df_effect_l.select(group_col).unique().height

        # For each lag k (0 to l-1)
        for k in range(l):
            # Get treatment difference at time F_g - 1 + l - k
            df_lag_k = df.filter(
                (pl.col(time_col) == pl.col(f_g_col) - 1 + l - k) &
                (pl.col(f_g_col) - 1 + l <= pl.col(t_g_col))
            )

            if df_lag_k.height == 0:
                continue

            # Join N_gt_l
            df_lag_k = df_lag_k.join(n_gt_l, on=group_col, how="left")

            # Compute delta_l_k
            delta_l_k = df_lag_k.select([
                (pl.col(treatment_col) - pl.col(d_sq_col)).abs() * pl.col("N_gt_l")
            ]).sum().item()

            # Weight = (delta_l_k / delta_D_l) / n_switchers_l
            if n_switchers_l > 0:
                weight_matrix[k, l - 1] = (delta_l_k / delta_D_l) / n_switchers_l * n_switchers_l

    # Create DataFrame
    row_names = [f"k={k}" for k in range(l_XX)]
    col_names = [f"l={l}" for l in range(1, l_XX + 1)]

    weight_df = pd.DataFrame(weight_matrix, index=row_names, columns=col_names)

    # Add Total row (sum of weights for each effect)
    totals = weight_df.sum(axis=0)
    weight_df.loc["Total"] = totals

    return weight_df


def format_normalized_weights_output(weight_df: pd.DataFrame) -> str:
    """
    Format normalized weights for console output.

    Parameters
    ----------
    weight_df : pd.DataFrame
        Weight matrix from compute_normalized_weights

    Returns
    -------
    str
        Formatted string for console output
    """
    lines = []

    width = 80
    lines.append("=" * width)
    lines.append("          Weights of Normalized Effects on Current and Lagged Treatments")
    lines.append("=" * width)
    lines.append("")

    # Format the DataFrame as a string with proper alignment
    # Get column widths
    col_width = 10

    # Header row
    header = f"{'':>8}"
    for col in weight_df.columns:
        header += f"{col:>{col_width}}"
    lines.append(header)
    lines.append("-" * width)

    # Data rows
    for idx, row in weight_df.iterrows():
        row_str = f"{idx:>8}"
        for val in row:
            if pd.isna(val) or val == 0:
                row_str += f"{'0.0000':>{col_width}}"
            else:
                row_str += f"{val:>{col_width}.4f}"
        lines.append(row_str)

        # Add separator before Total row
        if idx == weight_df.index[-2]:
            lines.append("-" * width)

    lines.append("=" * width)
    lines.append("")
    lines.append("Note: Columns sum to 1.0 for each effect (normalized).")
    lines.append("      Row k shows weight on treatment change k periods after first switch.")

    return "\n".join(lines)


def print_normalized_weights(
    df: pl.DataFrame,
    l_XX: int,
    group_col: str = "group_XX",
    time_col: str = "time_XX",
    treatment_col: str = "treatment_XX",
    f_g_col: str = "F_g_XX",
    t_g_col: str = "T_g_XX",
    d_sq_col: str = "d_sq_XX",
    n_gt_col: str = "N_gt_XX"
) -> pd.DataFrame:
    """
    Compute and print normalized weights.

    Parameters
    ----------
    df : pl.DataFrame
        Data with required columns
    l_XX : int
        Number of effects
    Other parameters are column names

    Returns
    -------
    pd.DataFrame
        Weight matrix
    """
    weight_df = compute_normalized_weights(
        df=df,
        l_XX=l_XX,
        group_col=group_col,
        time_col=time_col,
        treatment_col=treatment_col,
        f_g_col=f_g_col,
        t_g_col=t_g_col,
        d_sq_col=d_sq_col,
        n_gt_col=n_gt_col
    )

    output_str = format_normalized_weights_output(weight_df)
    print(output_str)

    return weight_df
