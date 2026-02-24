"""
Design option for did_multiplegt_dyn.

This module implements the design option which detects and displays
the different treatment paths that switcher groups follow.
"""

from __future__ import annotations

import warnings
from typing import Tuple, Optional, Union
from pathlib import Path

import polars as pl
import pandas as pd
import numpy as np


def parse_design_args(design: Tuple[float, str]) -> Tuple[float, str]:
    """
    Parse and validate design option arguments.

    Parameters
    ----------
    design : tuple
        Tuple of (percentage, output_path) where:
        - percentage: float between 0 and 1 indicating fraction of paths to show
        - output_path: "console" or path to Excel file

    Returns
    -------
    tuple
        Validated (percentage, output_path)

    Raises
    ------
    ValueError
        If arguments are invalid
    """
    if not isinstance(design, (tuple, list)) or len(design) != 2:
        raise ValueError(
            "design option requires a tuple of (percentage, output_path). "
            "Example: design=(0.9, 'console') or design=(1.0, 'results.xlsx')"
        )

    pct, path = design

    # Validate percentage
    if pct is None or pct == "":
        pct = 1.0
    pct = float(pct)
    if not 0 < pct <= 1:
        raise ValueError(f"design percentage must be between 0 and 1, got {pct}")

    # Validate path
    if not isinstance(path, str) or path == "":
        raise ValueError("design output_path must be 'console' or a file path")

    return pct, path


def detect_treatment_paths(
    df: pl.DataFrame,
    l_XX: int,
    T_max_XX: int,
    weight_col: Optional[str] = None
) -> pl.DataFrame:
    """
    Detect unique treatment paths followed by switcher groups.

    Parameters
    ----------
    df : pl.DataFrame
        Data with columns: group_XX, time_XX, treatment_XX, F_g_XX, weight_XX
    l_XX : int
        Number of effects (periods after switch to consider)
    T_max_XX : int
        Maximum time period in data
    weight_col : str, optional
        Name of weight column, defaults to "weight_XX"

    Returns
    -------
    pl.DataFrame
        DataFrame with treatment paths and group counts
    """
    weight_col = weight_col or "weight_XX"

    # Keep periods from F_g-1 to F_g+l-1 (l periods after first switch)
    df_paths = df.filter(
        (pl.col("time_XX") >= pl.col("F_g_XX") - 1) &
        (pl.col("time_XX") <= pl.col("F_g_XX") + l_XX - 1)
    )

    # Sort and create relative time index within each group
    df_paths = df_paths.sort(["group_XX", "time_XX"])
    df_paths = df_paths.with_columns(
        (pl.col("time_XX").cum_count().over("group_XX")).alias("time_l_XX")
    )

    # Select relevant columns
    cols_to_keep = ["group_XX", "time_l_XX", "treatment_XX", "F_g_XX"]
    if weight_col in df_paths.columns:
        cols_to_keep.append(weight_col)
    df_paths = df_paths.select(cols_to_keep)

    # Aggregate weights by group
    if weight_col in df_paths.columns:
        df_paths = df_paths.with_columns(
            pl.col(weight_col).sum().over("group_XX").alias("g_weight_XX")
        )
    else:
        df_paths = df_paths.with_columns(
            pl.lit(1.0).alias("g_weight_XX")
        )

    # Pivot to wide format (treatment at each relative time)
    df_wide = df_paths.select(
        ["group_XX", "time_l_XX", "treatment_XX", "F_g_XX", "g_weight_XX"]
    ).pivot(
        index=["group_XX", "F_g_XX", "g_weight_XX"],
        on="time_l_XX",
        values="treatment_XX"
    )

    # Rename pivoted columns to treatment_XX1, treatment_XX2, etc.
    rename_map = {}
    for col in df_wide.columns:
        if col not in ["group_XX", "F_g_XX", "g_weight_XX"]:
            rename_map[col] = f"treatment_XX{col}"
    df_wide = df_wide.rename(rename_map)

    # Drop rows with any missing treatment values
    treatment_cols = [c for c in df_wide.columns if c.startswith("treatment_XX")]
    df_wide = df_wide.drop_nulls(subset=treatment_cols)

    # Remove non-switchers (F_g > T_max)
    df_wide = df_wide.filter(pl.col("F_g_XX") <= T_max_XX)

    return df_wide, treatment_cols


def compute_path_statistics(
    df_wide: pl.DataFrame,
    treatment_cols: list
) -> pl.DataFrame:
    """
    Compute statistics for each unique treatment path.

    Parameters
    ----------
    df_wide : pl.DataFrame
        Wide-format data with treatment columns
    treatment_cols : list
        List of treatment column names

    Returns
    -------
    pl.DataFrame
        DataFrame with path statistics (N_groups, pct_groups)
    """
    # Group by treatment path and count
    df_stats = df_wide.group_by(treatment_cols).agg([
        pl.len().alias("N_groups"),
        pl.col("g_weight_XX").sum().alias("weighted_sum")
    ])

    # Calculate percentage of total
    total_weighted = df_stats.select(pl.col("weighted_sum").sum()).item()
    df_stats = df_stats.with_columns(
        (pl.col("weighted_sum") / total_weighted * 100).alias("pct_groups")
    )

    # Sort by frequency (descending)
    df_stats = df_stats.sort("N_groups", descending=True)

    return df_stats


def filter_paths_by_percentage(
    df_stats: pl.DataFrame,
    pct_threshold: float
) -> pl.DataFrame:
    """
    Filter to keep paths accounting for specified percentage of groups.

    Parameters
    ----------
    df_stats : pl.DataFrame
        Path statistics DataFrame
    pct_threshold : float
        Fraction of groups to include (0-1)

    Returns
    -------
    pl.DataFrame
        Filtered DataFrame with paths up to threshold
    """
    # Calculate cumulative percentage
    df_stats = df_stats.with_columns(
        (pl.col("pct_groups") / 100).cum_sum().alias("cum_pct")
    )

    # Keep paths up to threshold (plus one more to exceed threshold)
    df_filtered = df_stats.with_columns(
        (pl.col("cum_pct").shift(1).fill_null(0) < pct_threshold).alias("in_threshold")
    )

    # Include one row past threshold
    include_mask = df_filtered.select(
        pl.col("in_threshold") |
        (pl.col("in_threshold").shift(1).fill_null(True) & ~pl.col("in_threshold"))
    ).to_series()

    df_filtered = df_filtered.filter(include_mask)

    return df_filtered.drop(["cum_pct", "in_threshold", "weighted_sum"])


def format_design_output(
    df_paths: pl.DataFrame,
    treatment_cols: list,
    l_XX: int
) -> pd.DataFrame:
    """
    Format design output for display or export.

    Parameters
    ----------
    df_paths : pl.DataFrame
        Filtered path statistics
    treatment_cols : list
        Treatment column names
    l_XX : int
        Number of effect periods

    Returns
    -------
    pd.DataFrame
        Formatted DataFrame ready for output
    """
    # Reorder columns: N_groups, pct_groups, then treatment columns
    col_order = ["N_groups", "pct_groups"] + sorted(treatment_cols)
    df_out = df_paths.select([c for c in col_order if c in df_paths.columns])

    # Convert to pandas for output
    df_pd = df_out.to_pandas()

    # Rename columns for clarity
    rename_map = {"N_groups": "#Groups", "pct_groups": "%Groups"}
    for i, col in enumerate(sorted(treatment_cols)):
        rename_map[col] = f"l={i}"
    df_pd = df_pd.rename(columns=rename_map)

    # Add row labels
    df_pd.index = [f"TreatPath{i+1}" for i in range(len(df_pd))]

    return df_pd


def export_design(
    df_design: pd.DataFrame,
    output_path: str,
    l_XX: int,
    total_switchers: int,
    pct_shown: float,
    by_level: Optional[str] = None
) -> None:
    """
    Export design results to console or Excel file.

    Parameters
    ----------
    df_design : pd.DataFrame
        Formatted design DataFrame
    output_path : str
        "console" or path to Excel file
    l_XX : int
        Number of effect periods
    total_switchers : int
        Total number of switcher groups
    pct_shown : float
        Percentage of groups shown in output
    by_level : str, optional
        Level of by variable (if by option used)
    """
    if output_path.lower() == "console":
        # Console output
        print("\n" + "=" * 80)
        print(f"  Detection of treatment paths - {l_XX} periods after first switch")
        if by_level:
            print(f"  By: {by_level}")
        print("=" * 80)
        print(df_design.to_string())
        print("=" * 80)
        print(f"Treatment paths detected in switching groups: {total_switchers}")
        print(f"Total % shown: {pct_shown:.2f}%")

        # Interpretation of first row
        if len(df_design) > 0:
            first_row = df_design.iloc[0]
            n_groups = int(first_row["#Groups"])
            treatment_cols = [c for c in df_design.columns if c.startswith("l=")]
            if treatment_cols:
                d_start = first_row[treatment_cols[0]]
                d_vec = [first_row[c] for c in treatment_cols[1:]]
                print(f"\nDesign interpretation (first row):")
                print(f"  {n_groups} groups started with treatment {d_start} "
                      f"and then experienced treatment {d_vec}")
    else:
        # Excel output
        try:
            sheet_name = f"Design{' ' + by_level if by_level else ''}"
            df_design.to_excel(output_path, sheet_name=sheet_name[:31])  # Excel sheet name limit
            print(f"Design exported to {output_path}")
        except ImportError:
            warnings.warn(
                "openpyxl is required for Excel export. "
                "Install with: pip install openpyxl",
                UserWarning
            )
        except Exception as e:
            warnings.warn(f"Failed to export design to Excel: {e}", UserWarning)


def run_design_analysis(
    df: pl.DataFrame,
    design: Tuple[float, str],
    l_XX: int,
    T_max_XX: int,
    weight_col: Optional[str] = None,
    by_level: Optional[str] = None
) -> pd.DataFrame:
    """
    Run the full design analysis.

    Parameters
    ----------
    df : pl.DataFrame
        Input data with group, time, treatment, F_g columns
    design : tuple
        (percentage, output_path) tuple
    l_XX : int
        Number of effect periods
    T_max_XX : int
        Maximum time period
    weight_col : str, optional
        Weight column name
    by_level : str, optional
        Level of by variable

    Returns
    -------
    pd.DataFrame
        Design output DataFrame
    """
    # Parse arguments
    pct_threshold, output_path = parse_design_args(design)

    # Detect treatment paths
    df_wide, treatment_cols = detect_treatment_paths(df, l_XX, T_max_XX, weight_col)

    if len(df_wide) == 0:
        warnings.warn("No valid treatment paths detected", UserWarning)
        return pd.DataFrame()

    # Compute path statistics
    df_stats = compute_path_statistics(df_wide, treatment_cols)
    total_switchers = df_stats.select(pl.col("N_groups").sum()).item()

    # Filter by percentage
    df_filtered = filter_paths_by_percentage(df_stats, pct_threshold)

    # Calculate actual percentage shown
    pct_shown = df_filtered.select(pl.col("pct_groups").sum()).item()

    # Format output
    df_design = format_design_output(df_filtered, treatment_cols, l_XX)

    # Export
    export_design(df_design, output_path, l_XX, total_switchers, pct_shown, by_level)

    return df_design
