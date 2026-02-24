"""
Bootstrap option for did_multiplegt_dyn.

This module implements bootstrap standard errors for the DID estimator,
using cluster-aware resampling as in the Stata implementation.
"""

from __future__ import annotations

import warnings
from typing import Tuple, Optional, Dict, Any, List, Callable

import polars as pl
import pandas as pd
import numpy as np


def parse_bootstrap_args(bootstrap: Tuple[int, Optional[int]]) -> Tuple[int, Optional[int]]:
    """
    Parse and validate bootstrap arguments.

    Parameters
    ----------
    bootstrap : tuple
        Tuple of (n_reps, seed) where:
        - n_reps: number of bootstrap replications (minimum 2)
        - seed: random seed (optional, can be None)

    Returns
    -------
    tuple
        Validated (n_reps, seed)

    Raises
    ------
    ValueError
        If arguments are invalid
    """
    if not isinstance(bootstrap, (tuple, list)) or len(bootstrap) != 2:
        raise ValueError(
            "bootstrap option requires a tuple of (n_reps, seed). "
            "Example: bootstrap=(50, 12345) or bootstrap=(100, None)"
        )

    n_reps, seed = bootstrap

    # Validate n_reps
    n_reps = int(n_reps)
    if n_reps < 2:
        raise ValueError(f"bootstrap requires at least 2 replications, got {n_reps}")

    # Validate seed
    if seed is not None:
        seed = int(seed)

    return n_reps, seed


def validate_bootstrap_options(n_reps: int, continuous: int) -> None:
    """
    Validate bootstrap with other options and emit warnings.

    Parameters
    ----------
    n_reps : int
        Number of bootstrap replications
    continuous : int
        Polynomial degree for continuous treatment

    Warns
    -----
    UserWarning
        If bootstrap is used without continuous option
    """
    if continuous == 0:
        warnings.warn(
            "You specified the bootstrap option without the continuous option. "
            "We strongly recommend computing bootstrapped standard errors "
            "only when using the continuous option, as analytical SEs can be "
            "liberal in that case.",
            UserWarning
        )


def resample_clusters(
    df: pl.DataFrame,
    cluster_col: str,
    rng: np.random.Generator
) -> pl.DataFrame:
    """
    Perform cluster-aware resampling.

    Resamples entire clusters (groups) with replacement, preserving
    the within-cluster structure.

    Parameters
    ----------
    df : pl.DataFrame
        Input data
    cluster_col : str
        Column name for clustering (usually "group_XX" or "cluster_XX")
    rng : np.random.Generator
        NumPy random generator

    Returns
    -------
    pl.DataFrame
        Resampled data with same structure as input
    """
    # Get unique cluster IDs
    cluster_ids = df.select(pl.col(cluster_col).unique()).to_series().to_numpy()
    n_clusters = len(cluster_ids)

    # Sample clusters with replacement
    sampled_indices = rng.choice(n_clusters, size=n_clusters, replace=True)
    sampled_clusters = cluster_ids[sampled_indices]

    # Build resampled dataframe by concatenating sampled clusters
    # Create a mapping from original cluster to new cluster id
    dfs = []
    for i, orig_cluster in enumerate(sampled_clusters):
        cluster_data = df.filter(pl.col(cluster_col) == orig_cluster)
        # Assign new cluster id to avoid duplicate cluster ids
        cluster_data = cluster_data.with_columns(
            pl.lit(i).alias("_bootstrap_cluster_id")
        )
        dfs.append(cluster_data)

    if len(dfs) == 0:
        return df.with_columns(pl.lit(0).alias("_bootstrap_cluster_id"))

    resampled = pl.concat(dfs)

    return resampled


def extract_estimates(result: Dict[str, Any], l_XX: int, l_placebo_XX: int) -> Dict[str, float]:
    """
    Extract effect and placebo estimates from result dictionary.

    Parameters
    ----------
    result : dict
        Result from estimation function
    l_XX : int
        Number of effects
    l_placebo_XX : int
        Number of placebos

    Returns
    -------
    dict
        Dictionary of coefficient names to values
    """
    estimates = {}
    dyn = result.get("did_multiplegt_dyn", {})

    # Extract effects
    effects_df = dyn.get("Effects")
    if effects_df is not None and hasattr(effects_df, "shape"):
        for i in range(min(l_XX, len(effects_df))):
            try:
                estimates[f"Effect_{i+1}"] = float(effects_df.iloc[i]["Estimate"])
            except (KeyError, IndexError, TypeError):
                pass

    # Extract placebos
    placebos_df = dyn.get("Placebos")
    if placebos_df is not None and hasattr(placebos_df, "shape") and l_placebo_XX > 0:
        for i in range(min(l_placebo_XX, len(placebos_df))):
            try:
                estimates[f"Placebo_{i+1}"] = float(placebos_df.iloc[i]["Estimate"])
            except (KeyError, IndexError, TypeError):
                pass

    # Extract average effect
    ate_df = dyn.get("ATE")
    if ate_df is not None:
        try:
            if hasattr(ate_df, "iloc"):
                estimates["Av_tot_effect"] = float(ate_df.iloc[0]["Estimate"])
            elif isinstance(ate_df, (int, float)):
                estimates["Av_tot_effect"] = float(ate_df)
        except (KeyError, IndexError, TypeError):
            pass

    return estimates


def compute_bootstrap_statistics(
    bootstrap_estimates: List[Dict[str, float]]
) -> Tuple[Dict[str, float], np.ndarray, List[str]]:
    """
    Compute bootstrap standard errors and variance-covariance matrix.

    Parameters
    ----------
    bootstrap_estimates : list
        List of dictionaries with estimates from each bootstrap iteration

    Returns
    -------
    tuple
        (se_dict, vcov_matrix, coef_names)
        - se_dict: dictionary of coefficient names to bootstrap SEs
        - vcov_matrix: variance-covariance matrix
        - coef_names: list of coefficient names in vcov order
    """
    if not bootstrap_estimates:
        return {}, np.array([[]]), []

    # Get all coefficient names
    all_names = set()
    for est in bootstrap_estimates:
        all_names.update(est.keys())
    coef_names = sorted(all_names)

    # Build matrix of estimates
    n_reps = len(bootstrap_estimates)
    n_coefs = len(coef_names)
    est_matrix = np.full((n_reps, n_coefs), np.nan)

    for i, est in enumerate(bootstrap_estimates):
        for j, name in enumerate(coef_names):
            if name in est:
                est_matrix[i, j] = est[name]

    # Compute standard errors (ignoring NaN)
    se_dict = {}
    for j, name in enumerate(coef_names):
        col = est_matrix[:, j]
        valid = ~np.isnan(col)
        if np.sum(valid) > 1:
            se_dict[name] = float(np.std(col[valid], ddof=1))
        else:
            se_dict[name] = np.nan

    # Compute variance-covariance matrix
    # Use pairwise deletion for missing values
    vcov_matrix = np.full((n_coefs, n_coefs), np.nan)
    for i in range(n_coefs):
        for j in range(n_coefs):
            col_i = est_matrix[:, i]
            col_j = est_matrix[:, j]
            valid = ~np.isnan(col_i) & ~np.isnan(col_j)
            if np.sum(valid) > 1:
                vcov_matrix[i, j] = np.cov(col_i[valid], col_j[valid], ddof=1)[0, 1]

    return se_dict, vcov_matrix, coef_names


def run_bootstrap(
    df: pl.DataFrame,
    estimation_func: Callable,
    estimation_args: Dict[str, Any],
    n_reps: int,
    seed: Optional[int],
    cluster_col: str,
    l_XX: int,
    l_placebo_XX: int,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run bootstrap procedure.

    Parameters
    ----------
    df : pl.DataFrame
        Original data
    estimation_func : callable
        The estimation function to call
    estimation_args : dict
        Arguments for estimation function (excluding df)
    n_reps : int
        Number of bootstrap replications
    seed : int, optional
        Random seed
    cluster_col : str
        Column to use for cluster resampling
    l_XX : int
        Number of effects
    l_placebo_XX : int
        Number of placebos
    verbose : bool
        Whether to print progress

    Returns
    -------
    dict
        Bootstrap results:
        - 'se': dict of coefficient names to bootstrap SEs
        - 'vcov': variance-covariance matrix
        - 'coef_names': list of coefficient names
        - 'n_successful': number of successful iterations
    """
    # Initialize RNG
    rng = np.random.default_rng(seed)

    # Store estimates from each iteration
    bootstrap_estimates = []
    n_failed = 0

    for b in range(n_reps):
        if verbose and (b + 1) % 10 == 0:
            print(f"Bootstrap iteration {b + 1}/{n_reps}")

        try:
            # Resample data
            df_boot = resample_clusters(df, cluster_col, rng)

            # Update cluster column for resampled data
            boot_args = estimation_args.copy()
            boot_args["df"] = df_boot

            # Run estimation (suppress warnings during bootstrap)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                result = estimation_func(**boot_args)

            # Extract estimates
            estimates = extract_estimates(result, l_XX, l_placebo_XX)
            if estimates:
                bootstrap_estimates.append(estimates)
            else:
                n_failed += 1

        except Exception as e:
            n_failed += 1
            if verbose and n_failed <= 5:
                print(f"  Bootstrap iteration {b + 1} failed: {str(e)[:50]}")

    if len(bootstrap_estimates) < 2:
        warnings.warn(
            f"Bootstrap failed: only {len(bootstrap_estimates)} successful iterations",
            UserWarning
        )
        return {
            "se": {},
            "vcov": np.array([[]]),
            "coef_names": [],
            "n_successful": len(bootstrap_estimates)
        }

    if n_failed > 0 and verbose:
        print(f"Bootstrap completed: {n_reps - n_failed}/{n_reps} successful iterations")

    # Compute statistics
    se_dict, vcov_matrix, coef_names = compute_bootstrap_statistics(bootstrap_estimates)

    return {
        "se": se_dict,
        "vcov": vcov_matrix,
        "coef_names": coef_names,
        "n_successful": len(bootstrap_estimates)
    }


def replace_se_with_bootstrap(
    mat_res: np.ndarray,
    bootstrap_se: Dict[str, float],
    l_XX: int,
    l_placebo_XX: int,
    ci_level: float = 0.95
) -> np.ndarray:
    """
    Replace analytical SEs with bootstrap SEs in results matrix.

    Parameters
    ----------
    mat_res : np.ndarray
        Results matrix with columns [Estimate, SE, LB_CI, UB_CI, ...]
    bootstrap_se : dict
        Dictionary of coefficient names to bootstrap SEs
    l_XX : int
        Number of effects
    l_placebo_XX : int
        Number of placebos
    ci_level : float
        Confidence interval level

    Returns
    -------
    np.ndarray
        Updated results matrix with bootstrap SEs and CIs
    """
    from scipy.stats import norm

    z = norm.ppf((1 + ci_level) / 2)

    # Update effects (rows 0 to l_XX-1)
    for i in range(l_XX):
        key = f"Effect_{i+1}"
        if key in bootstrap_se and not np.isnan(bootstrap_se[key]):
            estimate = mat_res[i, 0]
            se = bootstrap_se[key]
            mat_res[i, 1] = se
            mat_res[i, 2] = estimate - z * se  # LB
            mat_res[i, 3] = estimate + z * se  # UB

    # Update placebos (rows l_XX to l_XX + l_placebo_XX - 1)
    for i in range(l_placebo_XX):
        key = f"Placebo_{i+1}"
        row_idx = l_XX + i
        if key in bootstrap_se and not np.isnan(bootstrap_se[key]):
            estimate = mat_res[row_idx, 0]
            se = bootstrap_se[key]
            mat_res[row_idx, 1] = se
            mat_res[row_idx, 2] = estimate - z * se  # LB
            mat_res[row_idx, 3] = estimate + z * se  # UB

    return mat_res
