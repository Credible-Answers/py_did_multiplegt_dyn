"""
date_first_switch option for did_multiplegt_dyn.

This module implements the date_first_switch option which reports dates of
first treatment changes and the number of groups switching at each date.
"""

from __future__ import annotations

import warnings
from typing import Optional, Union, Tuple, Dict, Any

import polars as pl
import pandas as pd
import numpy as np


def parse_date_first_switch_args(
    date_first_switch: Tuple[bool, str]
) -> Tuple[bool, str]:
    """
    Parse date_first_switch arguments.

    Parameters
    ----------
    date_first_switch : tuple
        (by_baseline_treat, output_path) where by_baseline_treat is bool
        and output_path is "console" or a file path

    Returns
    -------
    tuple
        (by_baseline_treat, output_path)

    Raises
    ------
    ValueError
        If arguments are invalid
    """
    if not isinstance(date_first_switch, (tuple, list)) or len(date_first_switch) != 2:
        raise ValueError(
            "date_first_switch must be a tuple of (by_baseline_treat, output_path). "
            "Example: (False, 'console') or (True, 'switching_dates.xlsx')"
        )

    by_baseline_treat, output_path = date_first_switch

    if not isinstance(by_baseline_treat, bool):
        raise ValueError(
            f"by_baseline_treat must be a boolean, got {type(by_baseline_treat)}"
        )

    if not isinstance(output_path, str):
        raise ValueError(
            f"output_path must be a string ('console' or file path), got {type(output_path)}"
        )

    return by_baseline_treat, output_path


def compute_switching_dates(
    df: pl.DataFrame,
    group_col: str = "group_XX",
    time_col: str = "time_XX",
    f_g_col: str = "F_g_XX",
    t_max_col: str = "T_max_XX",
    d_sq_col: str = "d_sq_XX",
    by_baseline_treat: bool = False
) -> Dict[Any, pd.DataFrame]:
    """
    Compute switching dates and group counts.

    Parameters
    ----------
    df : pl.DataFrame
        Data with group, time, F_g, and d_sq columns
    group_col, time_col, f_g_col, t_max_col, d_sq_col : str
        Column names
    by_baseline_treat : bool
        If True, break down by baseline treatment level

    Returns
    -------
    dict
        Dictionary of DataFrames with switching date statistics.
        Key is None for overall, or baseline treatment value for by_baseline_treat.
    """
    # Filter to switchers only (exclude never-switchers)
    T_max = df[t_max_col].max()
    df_switchers = df.filter(
        (pl.col(f_g_col) != T_max + 1) &
        pl.col(f_g_col).is_not_null()
    )

    # Keep only first observation per group at F_g
    df_first_switch = df_switchers.filter(
        pl.col(time_col) == pl.col(f_g_col)
    ).select([group_col, time_col, f_g_col, d_sq_col]).unique(subset=[group_col])

    results = {}

    if not by_baseline_treat:
        # Simple aggregation by switching date
        counts = df_first_switch.group_by(f_g_col).agg([
            pl.len().alias("n_groups")
        ]).sort(f_g_col)

        total = counts["n_groups"].sum()
        counts = counts.with_columns([
            (pl.col("n_groups") / total * 100).alias("pct_groups")
        ])

        # Convert to pandas for output
        counts_pd = counts.to_pandas()
        counts_pd = counts_pd.rename(columns={
            f_g_col: "Switching Date",
            "n_groups": "#Groups",
            "pct_groups": "%Groups"
        })
        counts_pd = counts_pd.set_index("Switching Date")

        results[None] = counts_pd

    else:
        # Aggregation by switching date AND baseline treatment
        unique_d_sq = df_first_switch[d_sq_col].unique().sort().to_list()

        for d_sq_val in unique_d_sq:
            df_subset = df_first_switch.filter(pl.col(d_sq_col) == d_sq_val)

            counts = df_subset.group_by(f_g_col).agg([
                pl.len().alias("n_groups")
            ]).sort(f_g_col)

            total = counts["n_groups"].sum()
            counts = counts.with_columns([
                (pl.col("n_groups") / total * 100).alias("pct_groups")
            ])

            # Convert to pandas for output
            counts_pd = counts.to_pandas()
            counts_pd = counts_pd.rename(columns={
                f_g_col: "Switching Date",
                "n_groups": "#Groups",
                "pct_groups": "%Groups"
            })
            counts_pd = counts_pd.set_index("Switching Date")

            results[d_sq_val] = counts_pd

    return results


def format_date_first_switch_output(
    results: Dict[Any, pd.DataFrame],
    by_baseline_treat: bool = False
) -> str:
    """
    Format switching dates for console output.

    Parameters
    ----------
    results : dict
        Dictionary of DataFrames from compute_switching_dates
    by_baseline_treat : bool
        Whether results are broken down by baseline treatment

    Returns
    -------
    str
        Formatted string for console output
    """
    lines = []

    width = 60
    lines.append("=" * width)
    lines.append("                    Switching Dates")
    lines.append("=" * width)

    if not by_baseline_treat:
        # Single table
        df = results[None]
        lines.append(f"{'Date':<15} {'#Groups':>12} {'%Groups':>12}")
        lines.append("-" * width)

        for idx, row in df.iterrows():
            date_str = str(int(idx)) if pd.notna(idx) else str(idx)
            lines.append(f"{date_str:<15} {row['#Groups']:>12.0f} {row['%Groups']:>12.2f}")

        lines.append("-" * width)
        lines.append(f"{'Total':<15} {df['#Groups'].sum():>12.0f} {100.0:>12.2f}")

    else:
        # Multiple tables by baseline treatment
        for d_sq_val, df in results.items():
            lines.append(f"\n  Status quo treatment = {d_sq_val}")
            lines.append("-" * width)
            lines.append(f"{'Date':<15} {'#Groups':>12} {'%Groups':>12}")
            lines.append("-" * width)

            for idx, row in df.iterrows():
                date_str = str(int(idx)) if pd.notna(idx) else str(idx)
                lines.append(f"{date_str:<15} {row['#Groups']:>12.0f} {row['%Groups']:>12.2f}")

            lines.append("-" * width)
            lines.append(f"{'Total':<15} {df['#Groups'].sum():>12.0f} {100.0:>12.2f}")

    lines.append("=" * width)

    return "\n".join(lines)


def export_date_first_switch(
    results: Dict[Any, pd.DataFrame],
    output_path: str,
    by_baseline_treat: bool = False
) -> None:
    """
    Export switching dates to console or Excel.

    Parameters
    ----------
    results : dict
        Dictionary of DataFrames from compute_switching_dates
    output_path : str
        "console" for console output, or path to Excel file
    by_baseline_treat : bool
        Whether results are broken down by baseline treatment
    """
    if output_path.lower() == "console":
        output_str = format_date_first_switch_output(results, by_baseline_treat)
        print(output_str)
    else:
        # Export to Excel
        try:
            with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
                if not by_baseline_treat:
                    results[None].to_excel(writer, sheet_name="Switching Dates")
                else:
                    for d_sq_val, df in results.items():
                        sheet_name = f"Status quo = {d_sq_val}"
                        # Truncate sheet name if too long
                        if len(sheet_name) > 31:
                            sheet_name = sheet_name[:31]
                        df.to_excel(writer, sheet_name=sheet_name)

            print(f"Switching dates exported to: {output_path}")

        except ImportError:
            warnings.warn(
                "openpyxl is required for Excel export. "
                "Install with: pip install openpyxl"
            )
            # Fall back to console
            output_str = format_date_first_switch_output(results, by_baseline_treat)
            print(output_str)


def run_date_first_switch(
    df: pl.DataFrame,
    date_first_switch: Tuple[bool, str],
    group_col: str = "group_XX",
    time_col: str = "time_XX",
    f_g_col: str = "F_g_XX",
    t_max_col: str = "T_max_XX",
    d_sq_col: str = "d_sq_XX"
) -> Dict[Any, pd.DataFrame]:
    """
    Run the date_first_switch analysis.

    Parameters
    ----------
    df : pl.DataFrame
        Data with required columns
    date_first_switch : tuple
        (by_baseline_treat, output_path)
    group_col, time_col, f_g_col, t_max_col, d_sq_col : str
        Column names

    Returns
    -------
    dict
        Dictionary of DataFrames with switching date statistics
    """
    by_baseline_treat, output_path = parse_date_first_switch_args(date_first_switch)

    results = compute_switching_dates(
        df=df,
        group_col=group_col,
        time_col=time_col,
        f_g_col=f_g_col,
        t_max_col=t_max_col,
        d_sq_col=d_sq_col,
        by_baseline_treat=by_baseline_treat
    )

    export_date_first_switch(results, output_path, by_baseline_treat)

    return results
