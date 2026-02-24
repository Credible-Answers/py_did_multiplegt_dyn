"""
by_path option for did_multiplegt_dyn.

This module implements the by_path option which stratifies the analysis
by different treatment path sequences, displaying results for each path.
"""

from __future__ import annotations

import warnings
from typing import Optional, Union, List, Dict, Any, Callable

import polars as pl
import pandas as pd
import numpy as np


def validate_by_path_input(
    by_path: Union[str, int],
    by_option: Optional[str] = None
) -> int:
    """
    Validate and parse by_path input.

    Parameters
    ----------
    by_path : str or int
        Either "all" or a positive integer
    by_option : str, optional
        The 'by' option value - cannot be used with by_path

    Returns
    -------
    int
        Number of paths to analyze (or -1 for "all")

    Raises
    ------
    ValueError
        If by_path is invalid or conflicts with by option
    """
    if by_option is not None:
        raise ValueError(
            "The by_path option cannot be combined with the by option. "
            "Please use only one of these options."
        )

    if isinstance(by_path, str):
        if by_path.lower() == "all":
            return -1  # Signal for all paths
        else:
            try:
                return int(by_path)
            except ValueError:
                raise ValueError(
                    f"by_path must be 'all' or a positive integer, got '{by_path}'"
                )
    elif isinstance(by_path, int):
        if by_path <= 0:
            raise ValueError(f"by_path must be positive, got {by_path}")
        return by_path
    else:
        raise ValueError(
            f"by_path must be 'all' or a positive integer, got {type(by_path)}"
        )


def identify_treatment_paths(
    df: pl.DataFrame,
    l_XX: int,
    group_col: str = "group_XX",
    time_col: str = "time_XX",
    treatment_col: str = "treatment_XX",
    f_g_col: str = "F_g_XX"
) -> tuple:
    """
    Identify unique treatment paths for each group.

    Creates treatment sequence for each group from F_g-1 to F_g-1+l
    and assigns path identifiers.

    Parameters
    ----------
    df : pl.DataFrame
        Data with group, time, treatment, and F_g columns
    l_XX : int
        Number of effect periods
    group_col, time_col, treatment_col, f_g_col : str
        Column names

    Returns
    -------
    tuple
        (DataFrame with 'different_paths_XX' column, list of dummy column names)
    """
    # Sort data
    df = df.sort([group_col, time_col])

    # For each group, we need to extract treatment values at relative times:
    # k=0: time = F_g - 1 (baseline)
    # k=1: time = F_g (first switch period)
    # k=2: time = F_g + 1
    # ... up to k=l: time = F_g - 1 + l

    dummy_cols = [f"dummy_treat_{k}" for k in range(l_XX + 1)]

    # Create relative time column
    df = df.with_columns(
        (pl.col(time_col) - pl.col(f_g_col) + 1).alias("rel_time_XX")
    )

    # For each relative time k (0 to l_XX), extract treatment value
    for k in range(l_XX + 1):
        # rel_time = k means time = F_g - 1 + k
        # So k=0 -> rel_time=0, k=1 -> rel_time=1, etc.
        df = df.with_columns(
            pl.when(pl.col("rel_time_XX") == k)
            .then(pl.col(treatment_col))
            .otherwise(None)
            .alias(f"dummy_treat_{k}_temp")
        )
        # Propagate to all observations in the group using first non-null
        df = df.with_columns(
            pl.col(f"dummy_treat_{k}_temp")
            .drop_nulls()
            .first()
            .over(group_col)
            .alias(dummy_cols[k])
        )
        df = df.drop(f"dummy_treat_{k}_temp")

    # Create path string from treatment sequence
    # Convert to int for cleaner display (0, 1 instead of 0.0, 1.0)
    df = df.with_columns(
        pl.concat_str(
            [pl.col(c).cast(pl.Int64).cast(pl.Utf8).fill_null("NA") for c in dummy_cols],
            separator="_"
        ).alias("path_str_XX")
    )

    # Create numeric path identifier - only for switchers
    # (groups where F_g <= T_g, meaning they actually switched)
    switcher_paths = df.filter(
        pl.col(f_g_col) <= pl.col("T_g_XX")
    ).select("path_str_XX").unique()

    path_mapping = switcher_paths.with_row_index("path_id_temp")
    df = df.join(path_mapping, on="path_str_XX", how="left")
    df = df.with_columns(
        (pl.col("path_id_temp") + 1).alias("different_paths_XX")
    )

    # Set path to null for never-switchers
    df = df.with_columns(
        pl.when(pl.col(f_g_col) > pl.col("T_g_XX"))
        .then(None)
        .otherwise(pl.col("different_paths_XX"))
        .alias("different_paths_XX")
    )

    # Clean up
    df = df.drop(["path_str_XX", "path_id_temp", "rel_time_XX"])

    return df, dummy_cols


def rank_paths_by_frequency(
    df: pl.DataFrame,
    first_obs_col: str = "first_obs_by_gp_XX"
) -> pl.DataFrame:
    """
    Rank treatment paths by frequency (most common = 1).

    Only COMPLETE paths (without NA values) are ranked first.
    Incomplete paths are ranked after all complete paths.
    This matches Stata's behavior where by_path only considers complete paths.

    Parameters
    ----------
    df : pl.DataFrame
        Data with 'different_paths_XX' and 'path_str_XX' columns
    first_obs_col : str
        Column indicating first observation per group

    Returns
    -------
    pl.DataFrame
        DataFrame with paths ranked by frequency
    """
    # Count groups per path (using first observation per group to avoid double counting)
    # Also get the path string for deterministic tie-breaking
    path_counts = df.filter(
        pl.col(first_obs_col) & pl.col("different_paths_XX").is_not_null()
    ).group_by("different_paths_XX").agg([
        pl.len().alias("count_path_XX"),
        pl.col("path_str_XX").first().alias("path_str_for_sort")
    ])

    # Mark complete vs incomplete paths (incomplete paths have "NA" in path string)
    path_counts = path_counts.with_columns(
        pl.col("path_str_for_sort").str.contains("NA").alias("is_incomplete")
    )

    # Add count to main dataframe
    df = df.join(
        path_counts.select(["different_paths_XX", "count_path_XX"]),
        on="different_paths_XX",
        how="left"
    )

    # Rank paths: complete paths first (by frequency), then incomplete paths (by frequency)
    # This ensures by_path=4 gets top 4 COMPLETE paths, matching Stata
    rank_df = path_counts.sort(
        ["is_incomplete", "count_path_XX", "path_str_for_sort"],
        descending=[False, True, False]  # complete first, then by count desc, then by string asc
    )
    rank_df = rank_df.with_row_index("rank_paths_XX")
    rank_df = rank_df.with_columns(
        (pl.col("rank_paths_XX") + 1).alias("rank_paths_XX")
    )

    # Create mapping from old path id to new rank
    path_rank_map = rank_df.select(["different_paths_XX", "rank_paths_XX"])

    # Join to get ranks
    df = df.join(path_rank_map, on="different_paths_XX", how="left")

    # Replace path id with rank
    df = df.with_columns(
        pl.col("rank_paths_XX").alias("different_paths_XX")
    )
    df = df.drop("rank_paths_XX")

    return df


def get_control_subset_for_path(
    df: pl.DataFrame,
    path_id: int,
    l_XX: int
) -> pl.DataFrame:
    """
    Get the subset of data for a specific treatment path analysis.

    Parameters
    ----------
    df : pl.DataFrame
        Full dataset with path identifiers
    path_id : int
        The path ID to analyze
    l_XX : int
        Number of effect periods

    Returns
    -------
    pl.DataFrame
        Filtered dataset with path switchers and valid controls
    """
    # Get baseline treatment (d_sq) for this path
    path_data = df.filter(
        pl.col("different_paths_XX") == path_id
    ).select("d_sq_XX").unique().drop_nulls().to_series()

    # Handle case where no data for this path
    if len(path_data) == 0:
        return df.filter(pl.col("different_paths_XX") == path_id)

    path_d_sq = path_data[0]

    # Create control indicator: same baseline, not yet switched
    df = df.with_columns(
        (
            (pl.col("time_XX") < pl.col("F_g_XX")) &
            (pl.col("d_sq_XX") == path_d_sq)
        ).alias("cont_path_alt_XX")
    )

    # Filter to include path switchers OR valid controls
    df_filtered = df.filter(
        (pl.col("different_paths_XX") == path_id) |
        pl.col("cont_path_alt_XX")
    )

    return df_filtered


def get_path_treatment_sequence(
    df: pl.DataFrame,
    path_id: int,
    l_XX: int,
    dummy_cols: List[str]
) -> List[int]:
    """
    Get the treatment sequence for a specific path.

    Parameters
    ----------
    df : pl.DataFrame
        Data with path identifiers and dummy columns
    path_id : int
        Path ID to get sequence for
    l_XX : int
        Number of effect periods
    dummy_cols : list
        List of dummy column names

    Returns
    -------
    list
        Treatment values at each relative time period (as integers)
    """
    path_data = df.filter(
        pl.col("different_paths_XX") == path_id
    ).select(dummy_cols).unique()

    if len(path_data) == 0:
        return []

    # Extract values and convert to integers for cleaner display
    sequence = []
    for col in dummy_cols:
        val = path_data[col][0]
        if val is not None:
            sequence.append(int(val))
        else:
            sequence.append(None)

    return sequence


def count_groups_in_path(
    df: pl.DataFrame,
    path_id: int,
    first_obs_col: str = "first_obs_by_gp_XX"
) -> int:
    """
    Count the number of groups following a specific path.

    Parameters
    ----------
    df : pl.DataFrame
        Data with path identifiers
    path_id : int
        Path ID to count
    first_obs_col : str
        Column indicating first observation per group

    Returns
    -------
    int
        Number of groups in this path
    """
    return df.filter(
        (pl.col("different_paths_XX") == path_id) &
        pl.col(first_obs_col)
    ).height


def format_path_for_display(sequence: List) -> str:
    """
    Format treatment sequence for display (matching Stata format).

    Parameters
    ----------
    sequence : list
        Treatment values at each time period

    Returns
    -------
    str
        Formatted string like "0, 1, 0, 1, 1, 1"
    """
    return ", ".join(str(v) if v is not None else "NA" for v in sequence)
