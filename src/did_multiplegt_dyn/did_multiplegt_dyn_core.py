from __future__ import annotations

import polars as pl
from typing import Any, List, Union, Iterable, Optional
import numpy as np


def _safe_div(num, den):
    """División segura que evita división por cero o nulls."""
    return pl.when(den.is_not_null() & (den != 0)).then(num / den).otherwise(None)

def apply_less_conservative_se(
    df: pl.DataFrame,
    i: int,
    trends_nonparam: Optional[Iterable[str]] = None,
    less_conservative_se: bool = True,
    cluster_col: str | None = None,
) -> pl.DataFrame:
    if not less_conservative_se:
        return df

    if trends_nonparam is None:
        trends_nonparam = []
    else:
        trends_nonparam = list(trends_nonparam)

    dof_s_col = f"dof_s_{i}_XX"
    diff_yN_col = f"diff_y_{i}_N_gt_XX"

    cond_dof = pl.col(dof_s_col) == 1

    def group_cols(path_col: str) -> list[str]:
        return [path_col, *trends_nonparam]

    # --------- Mean's denominator: count_cohort_* (sum of N_gt_XX) ---------
    for path_col, s_tag in [
        ("path_0_XX", "s0"),
        ("path_1_XX", "s1"),
        (f"path_{i}_XX", "s2"),
    ]:
        col = f"count_cohort_{i}_{s_tag}_t_XX"

        val_expr = pl.when(cond_dof).then(pl.col("N_gt_XX")).otherwise(None)

        df = df.with_columns(
            pl.when(cond_dof)
            .then(val_expr.sum().over(group_cols(path_col)))
            .otherwise(None)
            .alias(col)
        )

    # --------- Mean's numerator: total_cohort_* (sum of diff_y_i_N_gt_XX) ---------
    for path_col, s_tag in [
        ("path_0_XX", "s0"),
        ("path_1_XX", "s1"),
        (f"path_{i}_XX", "s2"),
    ]:
        col = f"total_cohort_{i}_{s_tag}_t_XX"

        val_expr = pl.when(cond_dof).then(pl.col(diff_yN_col)).otherwise(None)

        df = df.with_columns(
            pl.when(cond_dof)
            .then(val_expr.sum().over(group_cols(path_col)))
            .otherwise(None)
            .alias(col)
        )

    # --------- Counting number of groups for DOF adjustment ---------
    if cluster_col is None:
        # No clustering: sum of dof_s_i_XX
        val_expr = pl.when(cond_dof).then(pl.col(dof_s_col)).otherwise(None)
        for path_col, s_tag in [
            ("path_0_XX", "s0"),
            ("path_1_XX", "s1"),
            (f"path_{i}_XX", "s2"),
        ]:
            col = f"dof_cohort_{i}_{s_tag}_t_XX"

            df = df.with_columns(
                pl.when(cond_dof)
                .then(val_expr.sum().over(group_cols(path_col)))
                .otherwise(None)
                .alias(col)
            )
    else:
        # Clustering: number of unique clusters among rows with dof_s == 1
        cluster_dof_col = f"cluster_dof_{i}_s_XX"
        df = df.with_columns(
            pl.when(cond_dof)
            .then(pl.col(cluster_col))
            .otherwise(None)
            .alias(cluster_dof_col)
        )

        for path_col, s_tag in [
            ("path_0_XX", "s0"),
            ("path_1_XX", "s1"),
            (f"path_{i}_XX", "s2"),
        ]:
            col = f"dof_cohort_{i}_{s_tag}_t_XX"

            df = df.with_columns(
                pl.when(pl.col(cluster_dof_col).is_not_null())
                .then(pl.col(cluster_dof_col).n_unique().over(group_cols(path_col)))
                .otherwise(None)
                .alias(col)
            )

    # --------- Choose which cohort's DoF to use (s2, then s1, then s0) ---------
    col_s0 = f"dof_cohort_{i}_s0_t_XX"
    col_s1 = f"dof_cohort_{i}_s1_t_XX"
    col_s2 = f"dof_cohort_{i}_s2_t_XX"
    col_st = f"dof_cohort_{i}_s_t_XX"

    df = df.with_columns(
        pl.when(pl.col(col_s2) >= 2)
        .then(pl.col(col_s2))
        .when((pl.col(col_s2) < 2) & (pl.col(col_s1) >= 2))
        .then(pl.col(col_s1))
        .otherwise(pl.col(col_s0))
        .alias(col_st)
    )

    # --------- Mean: pick s2, else s1, else s0 ---------
    col_cnt_s0 = f"count_cohort_{i}_s0_t_XX"
    col_cnt_s1 = f"count_cohort_{i}_s1_t_XX"
    col_cnt_s2 = f"count_cohort_{i}_s2_t_XX"

    col_tot_s0 = f"total_cohort_{i}_s0_t_XX"
    col_tot_s1 = f"total_cohort_{i}_s1_t_XX"
    col_tot_s2 = f"total_cohort_{i}_s2_t_XX"

    col_mean = f"mean_cohort_{i}_s_t_XX"

    df = df.with_columns(
        pl.when(pl.col(col_s2) >= 2)
        .then(pl.col(col_tot_s2) / pl.col(col_cnt_s2))
        .when((pl.col(col_s2) < 2) & (pl.col(col_s1) >= 2))
        .then(pl.col(col_tot_s1) / pl.col(col_cnt_s1))
        .otherwise(pl.col(col_tot_s0) / pl.col(col_cnt_s0))
        .alias(col_mean)
    )

    return df

def compute_E_hat_gt_with_nans_pl(df: pl.DataFrame, i: int, type_sect: str = "effect") -> pl.DataFrame:
    
    if type_sect == "effect":
        E_hat    = f"E_hat_gt_{i}_XX"
        mean_ns  = f"mean_cohort_{i}_ns_t_XX"
        mean_s   = f"mean_cohort_{i}_s_t_XX"
        mean_nss = f"mean_cohort_{i}_ns_s_t_XX"

        dof_ns   = f"dof_cohort_{i}_ns_t_XX"
        dof_s    = f"dof_cohort_{i}_s_t_XX"
        dof_nss  = f"dof_cohort_{i}_ns_s_t_XX"
    else:
        E_hat    = f"E_hat_gt_pl_{i}_XX"
        mean_ns  = f"mean_cohort_pl_{i}_ns_t_XX"
        mean_s   = f"mean_cohort_pl_{i}_s_t_XX"
        mean_nss = f"mean_cohort_pl_{i}_ns_s_t_XX"

        dof_ns   = f"dof_cohort_pl_{i}_ns_t_XX"
        dof_s    = f"dof_cohort_pl_{i}_s_t_XX"
        dof_nss  = f"dof_cohort_pl_{i}_ns_s_t_XX"

    # Start missing
    df = df.with_columns(
        pl.lit(None).cast(pl.Float64).alias(E_hat)
    )

    time = pl.col("time_XX")
    Fg   = pl.col("F_g_XX")

    s    = pl.col(dof_s)
    ns   = pl.col(dof_ns)
    nss  = pl.col(dof_nss)

    # emulate .replace(np.nan, 9999)
    s9999   = s.fill_nan(9999).fill_null(9999)
    ns9999  = ns.fill_nan(9999).fill_null(9999)
    nss9999 = nss.fill_nan(9999).fill_null(9999)

    # Conditions
    cond_A = (time < Fg) | ((Fg - 1 + i) == time)

    cond_B = (time < Fg) & (ns9999 >= 2)

    cond_C = ((Fg - 1 + i) == time) & (s9999 >= 2)

    cond_D = (
        (nss >= 2)
        & (
            (((Fg - 1 + i) == time) & (s9999 == 1))
            | ((time < Fg) & (ns9999 == 1))
        )
    )

    # 1) E_hat = 0 if A
    df = df.with_columns(
        pl.when(cond_A)
          .then(0.0)
          .otherwise(pl.col(E_hat))
          .alias(E_hat)
    )

    # 2) replace with mean_ns if B
    df = df.with_columns(
        pl.when(cond_B)
          .then(pl.col(mean_ns))
          .otherwise(pl.col(E_hat))
          .alias(E_hat)
    )

    # df.loc[df[mean_ns].isna(), E_hat] = np.nan
    df = df.with_columns(
        pl.when(pl.col(mean_ns).is_null() | pl.col(mean_ns).is_nan())
          .then(pl.lit(None))
          .otherwise(pl.col(E_hat))
          .alias(E_hat)
    )

    # 3) replace with mean_s if C
    df = df.with_columns(
        pl.when(cond_C)
          .then(pl.col(mean_s))
          .otherwise(pl.col(E_hat))
          .alias(E_hat)
    )

    # 4) replace with mean_nss if D
    df = df.with_columns(
        pl.when(cond_D)
          .then(pl.col(mean_nss))
          .otherwise(pl.col(E_hat))
          .alias(E_hat)
    )

    # df.loc[df[mean_nss].isna(), E_hat] = np.nan
    df = df.with_columns(
        pl.when(pl.col(mean_nss).is_null() | pl.col(mean_nss).is_nan())
          .then(pl.lit(None))
          .otherwise(pl.col(E_hat))
          .alias(E_hat)
    )

    return df

def compute_DOF_gt_with_nans_pl(df: pl.DataFrame, i: int, type_sect: str = "effect") -> pl.DataFrame:
    # Name mapping
    if type_sect == "effect":
        DOF     = f"DOF_gt_{i}_XX"
        dof_s   = f"dof_cohort_{i}_s_t_XX"
        dof_ns  = f"dof_cohort_{i}_ns_t_XX"
        dof_nss = f"dof_cohort_{i}_ns_s_t_XX"
    else:
        DOF     = f"DOF_gt_pl_{i}_XX"
        dof_s   = f"dof_cohort_pl_{i}_s_t_XX"
        dof_ns  = f"dof_cohort_pl_{i}_ns_t_XX"
        dof_nss = f"dof_cohort_pl_{i}_ns_s_t_XX"

    # Start missing, like Stata: set DOF to NaN
    df = df.with_columns(
        pl.lit(None).cast(pl.Float64).alias(DOF)
    )

    # Convenience columns
    time = pl.col("time_XX")
    Fg   = pl.col("F_g_XX")
    s    = pl.col(dof_s)
    ns   = pl.col(dof_ns)
    nss  = pl.col(dof_nss)

    # Conditions (using Stata-style missing>number via fill_null(9999))
    cond_A = (time < Fg) | ((Fg - 1 + i) == time)

    cond_B = ((Fg - 1 + i) == time) & (s.fill_null(9999) > 1)

    cond_C = (time < Fg) & (ns.fill_null(9999) > 1)

    cond_D_inner = (
        (((Fg - 1 + i) == time) & ((s == 1) & s.is_not_null()))
        | ((time < Fg) & ((ns == 1) & ns.is_not_null()))
    )
    cond_D = (nss.fill_null(9999) >= 2) & cond_D_inner

    # 1) DOF = 1 if A
    df = df.with_columns(
        pl.when(cond_A)
          .then(1.0)
          .otherwise(pl.col(DOF))
          .alias(DOF)
    )

    # 2) replace with sqrt(dof_s/(dof_s-1)) if B
    df = df.with_columns(
        pl.when(cond_B)
          .then((s / (s - 1)).sqrt())
          .otherwise(pl.col(DOF))
          .alias(DOF)
    )
    # df.loc[df[dof_s].isna(), DOF] = np.nan
    df = df.with_columns(
        pl.when(s.is_null())
          .then(pl.lit(None))
          .otherwise(pl.col(DOF))
          .alias(DOF)
    )

    # 3) replace with sqrt(dof_ns/(dof_ns-1)) if C
    df = df.with_columns(
        pl.when(cond_C)
          .then((ns / (ns - 1)).sqrt())
          .otherwise(pl.col(DOF))
          .alias(DOF)
    )
    # (the commented-out pandas line is skipped as in your code)

    # 4) replace with sqrt(dof_nss/(dof_nss-1)) if D
    df = df.with_columns(
        pl.when(cond_D)
          .then((nss / (nss - 1)).sqrt())
          .otherwise(pl.col(DOF))
          .alias(DOF)
    )
    df = df.with_columns(
        pl.when(nss.is_null())
          .then(pl.lit(None))
          .otherwise(pl.col(DOF))
          .alias(DOF)
    )

    return df

def _flatten_vars(extra: Union[str, List[Any], None]) -> List[str]:
    """
    Flattens a nested structure (list, tuple, or set) of variable names into a single list.
    Ensures compatibility with Polars and other data processing steps.
    """

    # --- Handle None input ---
    if extra is None:
        return []

    # --- If a single string is provided, wrap it in a list ---
    if isinstance(extra, str):
        return [extra]

    # --- Flatten nested lists/tuples/sets into a single flat list ---
    flat: List[str] = []
    for x in extra:
        if isinstance(x, (list, tuple, set)):
            flat.extend(list(x))
        else:
            flat.append(x)

    return flat

def compute_dof_cohort_ns_s(
    df: pl.DataFrame,
    i: int,
    cluster_col: str | None = None,
    trends_nonparam: list[str] | None = None
) -> pl.DataFrame:
    """
    Computes the cohort-level degrees of freedom (DOF) for the union of switchers
    and not-yet-switchers ("ns_s") within each (d_sq, time, trends_nonparam) group.

    Behavior:
    - If no cluster is provided → counts the number of rows with dof_ns_s == 1 per group.
    - If a cluster column is provided → counts the number of unique clusters where dof_ns_s == 1.
    """

    # --- Build grouping keys ---
    group_vars = ["d_sq_XX"] + _flatten_vars(trends_nonparam) + ["time_XX"]

    dof_ns_s = f"dof_ns_s_{i}_XX"
    out_col  = f"dof_cohort_{i}_ns_s_t_XX"

    # === Case 1: Without cluster ===
    if cluster_col is None or cluster_col == "":
        # Mask: rows contributing to DOF (dof_ns_s == 1)
        mask_expr = pl.col(dof_ns_s) == 1

        # Count total dof_ns_s == 1 per group
        agg_df = (
            df.filter(mask_expr)
              .group_by(group_vars)
              .agg(pl.count().alias(out_col))
        )

        # Join group-level DOF counts back to df
        df = df.join(agg_df, on=group_vars, how="left")

        # Keep DOF only where condition applies
        df = df.with_columns(
            pl.when(mask_expr).then(pl.col(out_col)).otherwise(None).alias(out_col)
        )

    # === Case 2: With cluster ===
    else:
        clust_dof = f"cluster_dof_{i}_ns_s_XX"

        # Create helper: cluster value only when dof_ns_s == 1
        df = df.with_columns(
            pl.when(pl.col(dof_ns_s) == 1)
              .then(pl.col(cluster_col))
              .otherwise(None)
              .alias(clust_dof)
        )

        mask_expr = pl.col(clust_dof).is_not_null()

        # Count unique clusters per group
        agg_df = (
            df.filter(mask_expr)
              .group_by(group_vars)
              .agg(pl.col(clust_dof).n_unique().alias(out_col))
        )

        # Merge unique cluster counts back to main df
        df = df.join(agg_df, on=group_vars, how="left")

        # Keep values only where the mask applies
        df = df.with_columns(
            pl.when(mask_expr).then(pl.col(out_col)).otherwise(None).alias(out_col)
        )

    return df

def compute_ns_s_means_with_nans(df: pl.DataFrame, i: int, trends_nonparam=None) -> pl.DataFrame:
    """
    Polars version of compute_ns_s_means_with_nans.

    Replicates:
      gen dof_ns_s_i = (dof_s_i==1 | dof_ns_i==1)
      by d_sq trends_nonparam time: gegen count = total(N_gt) if dof_ns_s_i==1
      by d_sq trends_nonparam time: gegen total = total(diff_y_i_N_gt) if dof_ns_s_i==1
      gen mean = total / count

    And: if both dof_s_i and dof_ns_i are NaN, keep the outputs as NaN.
    """

    # --- Column names
    dof_s      = f"dof_s_{i}_XX"
    dof_ns     = f"dof_ns_{i}_XX"
    dof_ns_s   = f"dof_ns_s_{i}_XX"
    count_col  = f"count_cohort_{i}_ns_s_t_XX"
    total_col  = f"total_cohort_{i}_ns_s_t_XX"
    mean_col   = f"mean_cohort_{i}_ns_s_t_XX"
    diff_y_col = f"diff_y_{i}_N_gt_XX"

    # Group keys
    if trends_nonparam is None:
        group_vars = ["d_sq_XX", "time_XX"]
    else:
        group_vars = ["d_sq_XX"] + list(trends_nonparam) + ["time_XX"]

    # nontull = df[dof_s].notnull() | df[dof_ns].notnull()
    nontull_expr = pl.col(dof_s).is_not_null() | pl.col(dof_ns).is_not_null()

    # 1. dof_ns_s = (dof_s==1 or dof_ns==1)
    df = df.with_columns(
        (
            (pl.col(dof_s) == 1) | (pl.col(dof_ns) == 1)
        ).cast(pl.Int64).alias(dof_ns_s)
    )

    mask_ns_s = pl.col(dof_ns_s) == 1

    # 2. Denominator: sum(N_gt_XX) over rows with dof_ns_s==1, *only* for those rows
    # Step 2a: only N_gt_XX for mask rows, null elsewhere
    df = df.with_columns(
        pl.when(mask_ns_s)
          .then(pl.col("N_gt_XX"))
          .otherwise(pl.lit(None))
          .alias("_N_gt_tmp")
    )

    # Step 2b: group sum over group_vars
    df = df.with_columns(
        pl.col("_N_gt_tmp").sum().over(group_vars).alias(count_col)
    ).drop("_N_gt_tmp")

    # Step 2c: set to null for rows not in mask (to mimic df.loc[mask] assignment)
    df = df.with_columns(
        pl.when(mask_ns_s)
          .then(pl.col(count_col))
          .otherwise(pl.lit(None))
          .alias(count_col)
    )

    # Step 2d: also set to null where both dof_s and dof_ns are null (¬nontull)
    df = df.with_columns(
        pl.when(nontull_expr)
          .then(pl.col(count_col))
          .otherwise(pl.lit(None))
          .alias(count_col)
    )

    # 3. Numerator: sum(diff_y_i_N_gt_XX) over rows with dof_ns_s==1, same semantics
    df = df.with_columns(
        pl.when(mask_ns_s)
          .then(pl.col(diff_y_col))
          .otherwise(pl.lit(None))
          .alias("_diff_tmp")
    )

    df = df.with_columns(
        pl.col("_diff_tmp").sum().over(group_vars).alias(total_col)
    ).drop("_diff_tmp")

    df = df.with_columns(
        pl.when(mask_ns_s)
          .then(pl.col(total_col))
          .otherwise(pl.lit(None))
          .alias(total_col)
    )

    df = df.with_columns(
        pl.when(nontull_expr)
          .then(pl.col(total_col))
          .otherwise(pl.lit(None))
          .alias(total_col)
    )

    # 4. Mean
    df = df.with_columns(
        (pl.col(total_col) / pl.col(count_col)).alias(mean_col)
    )

    return df


def did_multiplegt_dyn_core_pl(
    df: pl.DataFrame,
    outcome,
    group,
    time,
    treatment, 
    effects, 
    placebo,
    trends_nonparam,
    controls,
    normalized,
    same_switchers, 
    same_switchers_pl, 
    only_never_switchers,
    globals_dict,
    dict_glob,
    const,
    controls_globals,
    trends_lin,
    less_conservative_se,
    continuous,
    cluster,
    switchers_core=None, 
    **kwargs
):

    # --- Initialize variables (CRAN compliance) ---
    list_names_const = []
    F_g_XX = None
    N_gt_XX = None
    T_d_XX = None
    cum_fillin_XX = None
    d_fg_XX_temp = None
    d_sq_int_XX = None
    dum_fillin_temp_XX = None
    dum_fillin_temp_pl_XX = None
    dummy_XX = None
    fillin_g_XX = None
    fillin_g_pl_XX = None
    group_XX = None
    num_g_paths_0_XX = None
    outcome_XX = None
    path_0_XX = None
    relevant_y_missing_XX = None
    sum_temp_XX = None
    sum_temp_pl_XX = None
    time_XX = None

    dict_vars_gen = {}

    # --- Inherited globals ---
    import numpy as np
    import polars as pl
    L_u_XX = globals_dict.get("L_u_XX", np.nan)
    L_placebo_u_XX = globals_dict.get("L_placebo_u_XX", None)
    L_placebo_a_XX = globals_dict.get("L_placebo_a_XX", None)
    L_a_XX = globals_dict.get("L_a_XX", None)
    t_min_XX = globals_dict.get("t_min_XX", None)
    T_max_XX = globals_dict.get("T_max_XX", None)
    G_XX = globals_dict.get("G_XX", None)

    # --- Define bounds based on switchers_core ---
    if switchers_core == "in":
        l_u_a_XX = np.nanmin(np.array([L_u_XX, effects]))
        if placebo != 0:
            l_placebo_u_a_XX = np.nanmin(np.array([placebo, L_placebo_u_XX]))
        increase_XX = 1
    elif switchers_core == "out":
        l_u_a_XX = np.nanmin(np.array([L_a_XX, effects]))
        if placebo != 0:
            l_placebo_u_a_XX = np.nanmin(np.array([placebo, L_placebo_a_XX]))
        increase_XX = 0

    # --- Initialize base categories ---
    levels_d_sq_XX = df["d_sq_int_XX"].unique().to_list()

    # --- Remove old columns safely ---
    drop_cols = ["num_g_paths_0_XX", "cohort_fullpath_0_XX"]
    df = df.drop([c for c in drop_cols if c in df.columns])

    # if cluster is None:
    #     print("No cluster")
    # print(f"{int(l_u_a_XX + 1)} Number of effects")

    # --- Main loop over i ---
    for i in range(1, int(l_u_a_XX + 1)):

        cols_to_drop = [
            f"distance_to_switch_{i}_XX",
            f"never_change_d_{i}_XX",
            f"N{increase_XX}_t_{i}_XX",
            f"N{increase_XX}_t_{i}_XX_w",
            f"N{increase_XX}_t_{i}_g_XX",
            f"N_gt_control_{i}_XX",
            f"diff_y_{i}_XX",
            f"dummy_U_Gg{i}_XX",
            f"U_Gg{i}_temp_XX",
            f"U_Gg{i}_XX",
            f"count{i}_core_XX",
            f"U_Gg{i}_temp_var_XX",
            f"U_Gg{i}_var_XX",
            f"never_change_d_{i}_wXX",
            f"distance_to_switch_{i}_wXX",
            f"d_fg{i}_XX",
            f"path_{i}_XX",
            f"num_g_paths_{i}_XX",
            f"cohort_fullpath_{i}_XX",
            f"count_cohort_{i}_s_t_XX",
            f"dof_cohort_{i}_s_t_XX",
            f"dof_cohort_{i}_ns_s_t_XX",
            f"dof_cohort_{i}_ns_t_XX",
            f"dof_cohort_{i}_s0_t_XX",
            f"dof_cohort_{i}_s1_t_XX",
            f"dof_cohort_{i}_s2_t_XX",
            f"dof_cohort_{i}_ns_s_t_XX"
            f"count_cohort_{i}_s_t_XX",
            f"count_cohort_{i}_ns_t_XX",
            f"count_cohort_{i}_s0_t_XX",
            f"count_cohort_{i}_s1_t_XX",
            f"count_cohort_{i}_s2_t_XX",
            f"total_cohort_{i}_s_t_XX",
            f"total_cohort_{i}_ns_t_XX",
            f"total_cohort_{i}_s0_t_XX",
            f"total_cohort_{i}_s1_t_XX",
            f"total_cohort_{i}_s2_t_XX",
            f"mean_cohort_{i}_s_t_XX",
            f"mean_cohort_{i}_ns_t_XX",
            f"mean_cohort_{i}_s0_t_XX",
            f"mean_cohort_{i}_s1_t_XX",
            f"mean_cohort_{i}_s2_t_XX",
            f"E_hat_gt_{i}_XX",
            f"DOF_gt_{i}_XX",
        ]

        # Drop columns if they exist (like Stata’s `capture drop`)
        existing = [c for c in cols_to_drop if c in df.columns]
        if existing:
            df = df.drop(existing)

        # 2. sort for the lag calculation
        df = df.sort(["group_XX", "time_XX"])

        # 3. compute long‐difference: outcome_XX - lag(outcome_XX, i)
        diff_y_col = f"diff_y_{i}_XX"
        df = df.with_columns(
            pl.col("outcome_XX").diff(i).over("group_XX").alias(diff_y_col)
        )

        # inside your loop over i:
        if less_conservative_se:
            # 1. temporary flag where time == F_g + i - 1
            df = df.with_columns(
                pl.when(pl.col("time_XX") == pl.col("F_g_XX") + i - 1)
                .then(pl.col("treatment_XX"))
                .otherwise(pl.lit(None))
                .alias("d_fg_XX_temp")
            )

            # 2. group‐level mean of that temp → d_fg{i}_XX
            df = df.with_columns(
                pl.col("d_fg_XX_temp").mean().over("group_XX").alias(f"d_fg{i}_XX")
            )

            # 3. when i == 1, initialize d_fg0_XX & path_0_XX
            if i == 1:
                df = df.with_columns(
                    pl.col("d_sq_XX").alias("d_fg0_XX")
                )
                # categorical cohort id for (d_fg0_XX, F_g_XX)
                df = df.with_columns(
                    pl.concat_str(
                        [
                            pl.col("d_fg0_XX").cast(pl.Utf8),
                            pl.lit("|"),
                            pl.col("F_g_XX").cast(pl.Utf8),
                        ]
                    )
                    .cast(pl.Categorical)      # categories for each distinct (d_fg0_XX, F_g_XX)
                    .to_physical()             # get the integer codes 0,1,2,...
                    .cast(pl.Int64)
                    .add(1)                    # start at 1 instead of 0
                    .alias("path_0_XX")
                )

            # 4. carry forward missing d_fg{i} from the previous period
            if i > 1:
                df = df.with_columns(
                    pl.when(pl.col(f"d_fg{i}_XX").is_null())
                    .then(pl.col(f"d_fg{i-1}_XX"))
                    .otherwise(pl.col(f"d_fg{i}_XX"))
                    .alias(f"d_fg{i}_XX")
                )

            # 5. build the new path_i grouping on (path_{i-1}, d_fg{i})
            df = df.with_columns(
                pl.concat_str(
                    [
                        pl.col(f"path_{i-1}_XX").cast(pl.Utf8),
                        pl.lit("|"),
                        pl.col(f"d_fg{i}_XX").cast(pl.Utf8),
                    ]
                )
                .cast(pl.Categorical)   # categories for each distinct (path_{i-1}_XX, d_fg{i}_XX)
                .to_physical()          # 0,1,2,...
                .cast(pl.Int64)
                .add(1)                 # start from 1
                .alias(f"path_{i}_XX")
            )


            # 6. drop the temp column
            df = df.drop("d_fg_XX_temp")

            # 7. for i == 1, count and flag cohorts on path_0_XX
            if i == 1:
                df = df.with_columns(
                    pl.col("group_XX").n_unique().over("path_0_XX").alias("num_g_paths_0_XX")
                )
                df = df.with_columns(
                    (pl.col("num_g_paths_0_XX") > 1).cast(pl.Int64).alias("cohort_fullpath_0_XX")
                )

            # 8. for every i, count distinct groups per path_i and flag
            df = df.with_columns(
                pl.col("group_XX").n_unique().over(f"path_{i}_XX").alias(f"num_g_paths_{i}_XX")
            )
            df = df.with_columns(
                (pl.col(f"num_g_paths_{i}_XX") > 1)
                .cast(pl.Int64)
                .alias(f"cohort_fullpath_{i}_XX")
            )

        # Column names
        never_change   = f"never_change_d_{i}_XX"
        never_change_w = f"never_change_d_{i}_wXX"
        N_gt_control   = f"N_gt_control_{i}_XX"
        diff_y         = f"diff_y_{i}_XX"

        # 1. Create never_change_d_i_XX
        # start as NaN, then set to 1 where diff_y not null and F_g_XX > time_XX
        df = df.with_columns(
            pl.when(pl.col(diff_y).is_not_null())
            .then((pl.col("F_g_XX") > pl.col("time_XX")).cast(pl.Float64))
            .otherwise(pl.lit(None))
            .alias(never_change)
        )

        # 2. Adjust if only_never_switchers option is on
        if only_never_switchers:
            condition_expr = (
                (pl.col("F_g_XX") > pl.col("time_XX")) &
                (pl.col("F_g_XX") < (T_max_XX + 1)) &
                pl.col(diff_y).is_not_null()
            )
            df = df.with_columns(
                pl.when(condition_expr)
                .then(0.0)
                .otherwise(pl.col(never_change))
                .alias(never_change)
            )

        # 3. Weighted never_change
        df = df.with_columns(
            (pl.col(never_change) * pl.col("N_gt_XX")).alias(never_change_w)
        )

        # 4. Sum within (time, d_sq, trends_nonparam)
        group_vars = ["time_XX", "d_sq_XX"]
        if trends_nonparam is not None:
            group_vars += trends_nonparam

        df = df.with_columns(
            pl.col(never_change_w).sum().over(group_vars).alias(N_gt_control)
        )


        # ================================
        # === SAME SWITCHERS CASE ========
        # ================================
        if same_switchers:
            # Step 1: init
            df = df.sort(["group_XX", "time_XX"])
            df = df.with_columns(pl.lit(0).alias("N_g_control_check_XX"))

            # ==== Step 2: loop over j ====
            for j in range(1, effects + 1):
                # --- long difference: group-wise j-lag diff of outcome_XX ---
                diff_col = f"diff_y_last_XX_{j}"
                df = df.with_columns(
                    (
                        pl.col("outcome_XX")
                        - pl.col("outcome_XX").shift(j).over("group_XX")
                    ).alias(diff_col)
                )

                # --- never change indicator ---
                never_col = f"never_change_d_last_XX_{j}"
                df = df.with_columns(
                    pl.when(
                        pl.col(diff_col).is_not_null() & (pl.col("F_g_XX") > pl.col("time_XX"))
                    )
                    .then(1)
                    .otherwise(0)
                    .alias(never_col)
                )

                # --- adjust if only_never_switchers flag is active ---
                if only_never_switchers:
                    cond_expr = (
                        (pl.col("F_g_XX") > pl.col("time_XX"))
                        & (pl.col("F_g_XX") < (pl.col("T_max_XX") + 1))
                        & pl.col(diff_col).is_not_null()
                    )
                    df = df.with_columns(
                        pl.when(cond_expr)
                        .then(0)
                        .otherwise(pl.col(never_col))
                        .alias(never_col)
                    )

                # --- weighted version ---
                never_w_col = f"never_change_d_last_wXX_{j}"
                df = df.with_columns(
                    (pl.col(never_col) * pl.col("N_gt_XX")).alias(never_w_col)
                )

                # --- group totals (gegen total) ---
                group_cols = ["time_XX", "d_sq_XX"]
                if trends_nonparam:
                    # allow trends_nonparam to be a string or a list
                    if isinstance(trends_nonparam, list):
                        group_cols += trends_nonparam
                    else:
                        group_cols.append(trends_nonparam)

                N_gt_control_col = f"N_gt_control_last_XX_{j}"
                df = df.with_columns(
                    pl.col(never_w_col).sum().over(group_cols).alias(N_gt_control_col)
                )

                # --- group mean at cutoff time ---
                N_g_temp_col = f"N_g_control_last_temp_XX_{j}"
                N_g_mean_col = f"N_g_control_last_m_XX_{j}"

                df = df.with_columns(
                    pl.when(pl.col("time_XX") == (pl.col("F_g_XX") - 1 + j))
                    .then(pl.col(N_gt_control_col))
                    .otherwise(None)
                    .alias(N_g_temp_col)
                )

                df = df.with_columns(
                    pl.col(N_g_temp_col).mean().over("group_XX").alias(N_g_mean_col)
                )

                # --- relevant diff_y at cutoff ---
                diff_y_temp_col = f"diff_y_relev_temp_XX_{j}"
                diff_y_relev_col = f"diff_y_relev_XX_{j}"

                df = df.with_columns(
                    pl.when(pl.col("time_XX") == (pl.col("F_g_XX") - 1 + j))
                    .then(pl.col(diff_col))
                    .otherwise(None)
                    .alias(diff_y_temp_col)
                )

                df = df.with_columns(
                    pl.col(diff_y_temp_col).mean().over("group_XX").alias(diff_y_relev_col)
                )

                # --- update check counter ---
                df = df.with_columns(
                    (
                        pl.col("N_g_control_check_XX")
                        + (
                            (pl.col(N_g_mean_col) > 0)
                            & pl.col(diff_y_relev_col).is_not_null()
                        ).cast(pl.Int64)
                    ).alias("N_g_control_check_XX")
                )


            # --- same_switchers_pl case ---
            if same_switchers_pl:
                df = df.sort(["group_XX", "time_XX"])
                df = df.with_columns(pl.lit(0).alias("N_g_control_check_pl_XX"))

                if trends_nonparam is None:
                    group_cols = ["time_XX", "d_sq_XX"]
                else:
                    group_cols = ["time_XX", "d_sq_XX"] + trends_nonparam

                for j in range(1, int(placebo + 1)):
                    # diff_y_last_XX = outcome_XX - lead(outcome_XX, j)
                    df = df.with_columns(
                        (pl.col("outcome_XX") - pl.col("outcome_XX").shift(-j))
                        .over("group_XX")
                        .alias("diff_y_last_XX")
                    )

                    df = df.with_columns(
                        pl.when(pl.col("diff_y_last_XX").is_not_null())
                        .then((pl.col("F_g_XX") > pl.col("time_XX")).cast(pl.Float64))
                        .otherwise(None)
                        .alias("never_change_d_last_XX")
                    )

                    if only_never_switchers:
                        df = df.with_columns(
                            pl.when(
                                (pl.col("F_g_XX") > pl.col("time_XX"))
                                & (pl.col("F_g_XX") < pl.lit(T_max_XX + 1))
                                & pl.col("diff_y_last_XX").is_not_null()
                            )
                            .then(0.0)
                            .otherwise(pl.col("never_change_d_last_XX"))
                            .alias("never_change_d_last_XX")
                        )

                    # never_change_d_last_wXX = never_change_d_last_XX * N_gt_XX
                    df = df.with_columns(
                        (pl.col("never_change_d_last_XX") * pl.col("N_gt_XX"))
                        .alias("never_change_d_last_wXX")
                    )

                    # bys time_XX d_sq_XX `trends_nonparam': total(never_change_d_last_wXX)
                    df = df.with_columns(
                        pl.col("never_change_d_last_wXX")
                        .sum()
                        .over(group_cols)
                        .alias("N_gt_control_last_XX")
                    )

                    # N_g_control_last_temp_XX = N_gt_control_last_XX if time_XX == F_g_XX - 1 - j
                    mask_time_j = pl.col("time_XX") == (pl.col("F_g_XX") - 1 - j)
                    df = df.with_columns(
                        pl.when(mask_time_j)
                        .then(pl.col("N_gt_control_last_XX"))
                        .otherwise(None)        # pandas np.nan → Polars null
                        .alias("N_g_control_last_temp_XX")
                    )

                    # bys group_XX: mean(N_g_control_last_temp_XX)
                    df = df.with_columns(
                        pl.col("N_g_control_last_temp_XX")
                        .mean()
                        .over("group_XX")
                        .alias("N_g_control_last_m_XX")
                    )

                    # diff_y_relev_temp_XX = diff_y_last_XX if time_XX == F_g_XX - 1 - j
                    df = df.with_columns(
                        pl.when(mask_time_j)
                        .then(pl.col("diff_y_last_XX"))
                        .otherwise(None)
                        .alias("diff_y_relev_temp_XX")
                    )

                    # bys group_XX: mean(diff_y_relev_temp_XX)
                    df = df.with_columns(
                        pl.col("diff_y_relev_temp_XX")
                        .mean()
                        .over("group_XX")
                        .alias("diff_y_relev_XX")
                    )

                    # N_g_control_check_pl_XX += (N_g_control_last_m_XX > 0 & diff_y_relev_XX != .)
                    df = df.with_columns(
                        (
                            pl.col("N_g_control_check_pl_XX")
                            + (
                                (pl.col("N_g_control_last_m_XX") > 0)
                                & pl.col("diff_y_relev_XX").is_not_null()
                            ).cast(pl.Int64)
                        ).alias("N_g_control_check_pl_XX")
                    )


                # relevant_y_missing_XX
                df = df.with_columns(
                    (
                        pl.col("outcome_XX").is_null()
                        & (pl.col("time_XX") >= pl.col("F_g_XX") - 1 - placebo)
                        & (pl.col("time_XX") <= pl.col("F_g_XX") - 1 + effects)
                    ).alias("relevant_y_missing_XX")
                )

                # controls option
                if controls:
                    df = df.with_columns(
                        pl.when(
                            (pl.col("fd_X_all_non_missing_XX") == 0)
                            & (pl.col("time_XX") >= pl.col("F_g_XX") - 1 - placebo)
                            & (pl.col("time_XX") <= pl.col("F_g_XX") - 1 + effects)
                        )
                        .then(True)
                        .otherwise(pl.col("relevant_y_missing_XX"))
                        .alias("relevant_y_missing_XX")
                    )

                df = df.with_columns(
                    (pl.col("N_g_control_check_pl_XX") == placebo).alias("fillin_g_pl_XX")
                )

                still_col = f"still_switcher_{i}_XX"
                df = df.with_columns(
                    (
                        (pl.col("F_g_XX") - 1 + effects <= pl.col("T_g_XX"))
                        & (pl.col("N_g_control_check_XX") == effects)
                    ).alias(still_col)
                )
                # distance_to_switch_`i'_XX =
                # (still_switcher_i & time_XX == F_g_XX - 1 + i
                #  & i <= L_g_XX & S_g_XX == increase_XX
                #  & N_gt_control_i_XX > 0 & N_gt_control_i_XX != .) if diff_y_i_XX != .
                dist_col = f"distance_to_switch_{i}_XX"
                n_gt_ctrl_col = f"N_gt_control_{i}_XX"
                diff_y_i_col = f"diff_y_{i}_XX"

                df = df.with_columns(
                    pl.when(
                        (pl.col(f"still_switcher_{i}_XX") == 1)
                        & (pl.col("time_XX") == (pl.col("F_g_XX") - 1 + i))
                        & (i <= pl.col("L_g_XX"))
                        & (pl.col("S_g_XX") == pl.lit(increase_XX))
                        & (pl.col(n_gt_ctrl_col) > 0)
                        & (pl.col(n_gt_ctrl_col).is_not_null())
                        & (pl.col(diff_y_i_col).is_not_null())
                    )
                    .then(1.0)
                    .otherwise(0.0)
                    .alias(dist_col)
                )
            else:
                # ---- relevant_y_missing ----
                df = df.with_columns(
                    ((pl.col("outcome_XX").is_null()) &
                     (pl.col("time_XX") >= (pl.col("F_g_XX") - 1)) &
                     (pl.col("time_XX") <= (pl.col("F_g_XX") - 1 + effects))
                    ).alias("relevant_y_missing_XX")
                )

                if controls:
                    df = df.with_columns(
                        pl.when(
                            (pl.col("fd_X_all_non_missing_XX") == 0) &
                            (pl.col("time_XX") >= pl.col("F_g_XX")) &
                            (pl.col("time_XX") <= (pl.col("F_g_XX") - 1 + effects))
                        )
                        .then(1)
                        .otherwise(pl.col("relevant_y_missing_XX"))
                        .alias("relevant_y_missing_XX")
                    )

                # ---- still_switcher ----
                still_col = f"still_switcher_{i}_XX"
                df = df.with_columns(
                    (
                        ((pl.col("F_g_XX") - 1 + effects <= pl.col("T_g_XX")) &
                         (pl.col("N_g_control_check_XX") == effects))
                        .cast(pl.Int64)
                    ).alias(still_col)
                )

                # ---- distance_to_switch ----
                dist_col = f"distance_to_switch_{i}_XX"
                df = df.with_columns(pl.lit(None).alias(dist_col))

                df = df.with_columns(
                    pl.when(
                        (pl.col(f"diff_y_{i}_XX").is_not_null()) &
                        (pl.col(still_col) == 1) &
                        (pl.col("time_XX") == (pl.col("F_g_XX") - 1 + i)) &
                        (i <= pl.col("L_g_XX")) &
                        (pl.col("S_g_XX") == pl.lit(increase_XX)) &
                        (pl.col(f"N_gt_control_{i}_XX") > 0) &
                        (pl.col(f"N_gt_control_{i}_XX").is_not_null())
                    )
                    .then(1.0)
                    .otherwise(0.0)
                    .alias(dist_col)
                )

                df = df.with_columns(
                    pl.when(pl.col(f"diff_y_{i}_XX").is_null())
                    .then(None)
                    .otherwise(pl.col(dist_col))
                    .alias(dist_col)
                )

        else:
            col_dist = f"distance_to_switch_{i}_XX"
            col_diff = f"diff_y_{i}_XX"
            col_ctrl = f"N_gt_control_{i}_XX"

            # Build the "product of booleans" condition
            cond_expr = (
                (pl.col("time_XX") == (pl.col("F_g_XX") - 1 + i)) &
                (pl.col("L_g_XX") >= i) &
                (pl.col("S_g_XX") == increase_XX) &
                (pl.col(col_ctrl) > 0) &
                pl.col(col_ctrl).is_not_null()
            )

            df = df.with_columns(
                pl.when(pl.col(col_diff).is_null())
                .then(pl.lit(None))                     # NaN where diff_y is NaN
                .otherwise(cond_expr.cast(pl.Float64))  # 0.0 / 1.0 elsewhere
                .alias(col_dist)
            )


        # Necesitamos checkear como se estan contando los switchers
        # Ensure the "distance_to_switch" column is numeric
        df = df.with_columns(pl.col(f"distance_to_switch_{i}_XX").cast(pl.Float64))

        # Create weighted distance variable
        df = df.with_columns(
            (pl.col(f"distance_to_switch_{i}_XX") * pl.col("N_gt_XX"))
            .alias(f"distance_to_switch_{i}_wXX")
        )

        # Sum over time to get counts per period
        df = df.with_columns([
            pl.sum(f"distance_to_switch_{i}_wXX").over("time_XX").alias(f"N{increase_XX}_t_{i}_XX"),
            pl.sum(f"distance_to_switch_{i}_XX").over("time_XX").alias(f"N_dw{increase_XX}_t_{i}_XX")
        ])

        dict_vars_gen[f"N{increase_XX}_{i}_XX"] = 0
        dict_vars_gen[f"N{increase_XX}_dw_{i}_XX"] = 0
        n_placebo = dict_vars_gen[f"N{increase_XX}_{i}_XX"]
        n_dw_placebo = dict_vars_gen[f"N{increase_XX}_dw_{i}_XX"]

        # ——— Loop over time and add up the period‐means ———
        for t in range(int(t_min_XX), int(T_max_XX + 1)):
            col_p  = f"N{increase_XX}_t_{i}_XX"
            col_dp = f"N_dw{increase_XX}_t_{i}_XX"

            mask = df.filter(pl.col("time_XX") == t)
            n_placebo    += mask[col_p].mean()
            n_dw_placebo += mask[col_dp].mean()

        dict_vars_gen[f"N{increase_XX}_{i}_XX"] = n_placebo
        dict_vars_gen[f"N{increase_XX}_dw_{i}_XX"] = n_dw_placebo

        # Count groups ℓ periods away from switch by (time, baseline treatment, trends)
        group_cols = ["time_XX", "d_sq_XX"] + ([] if trends_nonparam is None else trends_nonparam)

        df = df.with_columns(
            pl.sum(f"distance_to_switch_{i}_wXX").over(group_cols).alias(f"N{increase_XX}_t_{i}_g_XX")
        )

        # Safe division as Polars expression
        def safe_div_expr(num_expr: pl.Expr, den_expr: pl.Expr) -> pl.Expr:
            return (
                pl.when((den_expr != 0) & den_expr.is_not_null())
                .then(num_expr / den_expr)
                .otherwise(None)
            )

        if controls:
            # Initialize intermediate variable
            part2_col = f"part2_switch{increase_XX}_{i}_XX"
            df = df.with_columns(pl.lit(0.0).alias(part2_col))

            # Compute T_d_XX: last period by baseline treatment
            df = df.with_columns(
                (pl.col("F_g_XX").max().over("d_sq_int_XX") - 1).alias("T_d_XX")
            )

            count_controls = 0
            for var in controls:
                count_controls += 1

                # Sort by panel identifiers (for lags to be meaningful)
                df = df.sort(["group_XX", "time_XX"])

                # Compute lags within groups: L{i}_{var}
                lag_col = f"L{i}_{var}"
                df = df.with_columns(
                    pl.col(var).shift(i).over("group_XX").alias(lag_col)
                )

                # Compute the placebo difference: diff_X{count_controls}_{i}_XX
                diff_col = f"diff_X{count_controls}_{i}_XX"
                df = df.with_columns(
                    (pl.col(var) - pl.col(lag_col)).alias(diff_col)
                )

                # Weighted long difference: diff_X{count_controls}_{i}_N_XX
                diff_n_col = f"diff_X{count_controls}_{i}_N_XX"
                df = df.with_columns(
                    (pl.col("N_gt_XX") * pl.col(diff_col)).alias(diff_n_col)
                )

                # Constant column G_XX
                df = df.with_columns(pl.lit(G_XX).alias("G_XX"))

                # Loop over d-levels
                for l in levels_d_sq_XX:
                    l = int(l)

                    # Column names
                    m_g_col   = f"m{increase_XX}_g_{count_controls}_{l}_{i}_XX"
                    m_sum_col = f"m{increase_XX}_{l}_{count_controls}_{i}_XX"
                    M_col     = f"M{increase_XX}_{l}_{count_controls}_{i}_XX"

                    # Conditions
                    cond1 = (
                        (pl.lit(i) <= (pl.col("T_g_XX") - 2))
                        & (pl.col("d_sq_int_XX") == l)
                    )
                    cond2 = (
                        (pl.col("time_XX") >= (i + 1))
                        & (pl.col("time_XX") <= pl.col("T_g_XX"))
                    )

                    # Inner pieces
                    num_inner_expr = (
                        pl.col(f"distance_to_switch_{i}_XX")
                        - safe_div_expr(
                            pl.col(f"N{increase_XX}_t_{i}_g_XX"),
                            pl.col(f"N_gt_control_{i}_XX"),
                        )
                        * pl.col(f"never_change_d_{i}_XX")
                    )

                    # N{increase}_{i}_XX from dict_vars_gen (scalar)
                    N_i_col = f"N{increase_XX}_{i}_XX"
                    df = df.with_columns(pl.lit(dict_vars_gen[N_i_col]).alias(N_i_col))

                    frac_expr = safe_div_expr(pl.col("G_XX"), pl.col(N_i_col))

                    # m`increase'_g_count_l_i_XX
                    df = df.with_columns(
                        (
                            cond1.cast(pl.Float64)
                            * frac_expr
                            * (
                                num_inner_expr
                                * cond2.cast(pl.Float64)
                                * pl.col(diff_n_col)
                            )
                        ).alias(m_g_col)
                    )

                    # Sum across t within group g, keep only first row per group
                    tmp_idx = "__idx_group_tmp"
                    df = df.with_columns(
                        pl.col(m_g_col).sum().over("group_XX").alias(m_sum_col),
                        pl.arange(0, pl.count()).over("group_XX").alias(tmp_idx),
                    )
                    df = df.with_columns(
                        pl.when(pl.col(tmp_idx) == 0)
                        .then(pl.col(m_sum_col))
                        .otherwise(None)
                        .alias(m_sum_col)
                    ).drop(tmp_idx)

                    # Total m across groups (scalar)
                    total_m = float(
                        df.select(pl.col(m_sum_col).sum()).to_series()[0]
                    )

                    # M^{+/-}_{d,l} : total of m over all obs, scaled by 1/G_XX
                    df = df.with_columns(
                        safe_div_expr(pl.lit(total_m), pl.col("G_XX")).alias(M_col)
                    )

                    # ----- Number of groups within each not-yet-switched cohort (E_hat_denom...) -----
                    # dummy_XX = (F_g_XX > time_XX) & (d_sq_int_XX == l) if diff_y_XX not missing
                    df = df.with_columns(
                        pl.when(
                            (pl.col("diff_y_XX").is_not_null())
                            & (pl.col("F_g_XX") > pl.col("time_XX"))
                            & (pl.col("d_sq_int_XX") == l)
                        )
                        .then(1)
                        .otherwise(0)
                        .alias("dummy_XX")
                    )

                    E_hat_col = f"E_hat_denom_{count_controls}_{l}_XX"

                    if cluster is None:
                        # no clustering: sum dummy by time, only for d_sq_int==l
                        df = df.with_columns(
                            pl.when(pl.col("d_sq_int_XX") == l)
                            .then(pl.col("dummy_XX").sum().over("time_XX"))
                            .otherwise(None)
                            .alias(E_hat_col)
                        )
                    else:
                        # clustering: nunique over cluster among dummy==1
                        df = df.with_columns(
                            pl.when(pl.col("dummy_XX") == 1)
                            .then(pl.col(cluster))
                            .otherwise(None)
                            .alias("cluster_temp_XX")
                        )

                        df = df.with_columns(
                            pl.when(pl.col("cluster_temp_XX").is_not_null())
                            .then(
                                pl.col("cluster_temp_XX")
                                .n_unique()
                                .over("time_XX")
                            )
                            .otherwise(None)
                            .alias(E_hat_col)
                        )

                    # Indicator for at least two groups in cohort
                    Ey_hat_gt_col = f"E_y_hat_gt_{l}_XX"
                    Ey_hat_gt_int_col = f"E_y_hat_gt_int_{l}_XX"

                    df = df.with_columns(
                        (
                            pl.col(Ey_hat_gt_int_col)
                            * (pl.col(E_hat_col) >= 2).cast(pl.Float64)
                        ).alias(Ey_hat_gt_col)
                    )

                    # ----- Summation from t=2 to F_g-1 in U^{+,var,X}_{g,l} / U^{-,var,X}_{g,l} -----
                    in_sum_temp_col     = f"in_sum_temp_{count_controls}_{l}_XX"
                    in_sum_temp_adj_col = f"in_sum_temp_adj_{count_controls}_{l}_XX"
                    in_sum_col          = f"in_sum_{count_controls}_{l}_XX"

                    # N_c_l_temp_XX and N_c_l_XX
                    N_c_temp_col = f"N_c_{l}_temp_XX"
                    N_c_col      = f"N_c_{l}_XX"

                    # N_c_l_temp_XX = N_gt_XX * 1{ d_sq_int==l, 2<=time<=T_d, time<F_g, diff_y not missing }
                    cond_Nc = (
                        (pl.col("d_sq_int_XX") == l)
                        & (pl.col("time_XX") >= 2)
                        & (pl.col("time_XX") <= pl.col("T_d_XX"))
                        & (pl.col("time_XX") < pl.col("F_g_XX"))
                        & (pl.col("diff_y_XX").is_not_null())
                    )

                    df = df.with_columns(
                        pl.when(cond_Nc)
                        .then(pl.col("N_gt_XX"))
                        .otherwise(0.0)
                        .alias(N_c_temp_col)
                    )

                    # Total across all obs (scalar replicated)
                    total_Nc = float(
                        df.select(pl.col(N_c_temp_col).sum()).to_series()[0]
                    )

                    df = df.with_columns(
                        pl.lit(total_Nc).alias(N_c_col)
                    )

                    # Adjust demeaning when E_hat_denom == 1
                    # 1) Start with: 0 where Ey_hat_gt_col not null, else null
                    df = df.with_columns(
                        pl.when(pl.col(Ey_hat_gt_col).is_not_null())
                        .then(0.0)
                        .otherwise(None)
                        .alias(in_sum_temp_adj_col)
                    )

                    # 2) Overwrite where Ey_hat_gt_col not null AND E_hat_col > 1
                    df = df.with_columns(
                        pl.when(
                            (pl.col(Ey_hat_gt_col).is_not_null()) & (pl.col(E_hat_col) > 1)
                        )
                        .then(
                            (pl.col(E_hat_col) / (pl.col(E_hat_col) - 1.0)).sqrt() - 1.0
                            # or equivalently:
                            # ( (pl.col(E_hat_col) / (pl.col(E_hat_col) - 1.0)) ** 0.5 ) - 1.0
                        )
                        .otherwise(pl.col(in_sum_temp_adj_col))
                        .alias(in_sum_temp_adj_col)
                    )


                    # Build in-summand:
                    # (prod_X{count}_Ngt_XX * (1 + 1{E_hat>=2} * adj)
                    #  * (diff_y_XX - E_y_hat_gt_l_XX) * 1{2<=time<=F_g-1}) / N_c_l_XX
                    prod_col = f"prod_X{count_controls}_Ngt_XX"

                    summand_expr = (
                        pl.col(prod_col)
                        * (
                            1.0
                            + (pl.col(E_hat_col) >= 2).cast(pl.Float64)
                            * pl.col(in_sum_temp_adj_col)
                        )
                        * (pl.col("diff_y_XX") - pl.col(Ey_hat_gt_col))
                        * (
                            (pl.col("time_XX") >= 2)
                            & (pl.col("time_XX") <= (pl.col("F_g_XX") - 1))
                        ).cast(pl.Float64)
                    )

                    df = df.with_columns(
                        safe_div_expr(summand_expr, pl.col(N_c_col)).alias(in_sum_temp_col)
                    )

                    # Sum within group g across t
                    df = df.with_columns(
                        pl.col(in_sum_temp_col).sum().over("group_XX").alias(in_sum_col)
                    )

                # -----------------------------------------------
                # Second loop over l: demeaning diff_y_{i}_XX by controls
                # -----------------------------------------------
                for l in levels_d_sq_XX:
                    l = int(l)
                    if dict_glob[f"useful_res_{l}_XX"] > 1:
                        coef = dict_glob[f"coefs_sq_{l}_XX"][count_controls - 1]
                        diff_y_col = f"diff_y_{i}_XX"

                        df = df.with_columns(
                            pl.when(pl.col("d_sq_int_XX") == l)
                            .then(
                                pl.col(diff_y_col)
                                - coef * pl.col(f"diff_X{count_controls}_{i}_XX")
                            )
                            .otherwise(pl.col(diff_y_col))
                            .alias(diff_y_col)
                        )

                        # Drop if exists and then generate new column = 0
                        col_name = f"in_brackets_{l}_{count_controls}_XX"
                        if col_name in df.columns:
                            df = df.drop(col_name)
                        df = df.with_columns(pl.lit(0.0).alias(col_name))

        import polars as pl

        # 1. Weighted long difference of outcome
        df = df.with_columns(
            (pl.col(f"diff_y_{i}_XX") * pl.col("N_gt_XX")).alias(f"diff_y_{i}_N_gt_XX")
        )

        # 2. DOF indicator: 1 if N_gt_XX != 0 and diff_y not missing
        df = df.with_columns(
            (
                (pl.col("N_gt_XX") != 0) &
                pl.col(f"diff_y_{i}_XX").is_not_null()
            ).cast(pl.Int64).alias(f"dof_y_{i}_N_gt_XX")
        )

        # Drop old columns if they exist
        cols_to_drop = [f"dof_ns_{i}_XX", f"dof_s_{i}_XX"]
        existing = [c for c in cols_to_drop if c in df.columns]
        if existing:
            df = df.drop(existing)

        # Generate dof_ns_i_XX
        df = df.with_columns(
            (
                (pl.col("N_gt_XX") != 0) &
                pl.col(f"diff_y_{i}_XX").is_not_null() &
                (pl.col(f"never_change_d_{i}_XX") == 1) &
                (pl.col(f"N{increase_XX}_t_{i}_XX") > 0) &
                pl.col(f"N{increase_XX}_t_{i}_XX").is_not_null()
            ).cast(pl.Int64).alias(f"dof_ns_{i}_XX")
        )

        # Generate dof_s_i_XX
        df = df.with_columns(
                (
                    (pl.col("N_gt_XX") != 0) &
                    (pl.col(f"distance_to_switch_{i}_XX") == 1)
                )
                .cast(pl.Int64)
                .fill_null(0)                    # Replaces null → 0
                .alias(f"dof_s_{i}_XX")
            )

        # 3. Define mask for "never switchers" with valid diff_y and controls
        mask_expr = pl.col(f"dof_ns_{i}_XX") == 1

        if trends_nonparam is None:
            group_cols = ["d_sq_XX", "time_XX"]
        else:
            group_cols = ["d_sq_XX"] + trends_nonparam + ["time_XX"]

        # 4. Denominator of the mean: count of control groups weighted by N_gt_XX
        df = df.with_columns(
            pl.when(mask_expr)
            .then(pl.col("N_gt_XX"))
            .otherwise(0)
            .alias("_mask_den")
        )

        df = df.with_columns(
            pl.col("_mask_den").sum().over(group_cols).alias(f"count_cohort_{i}_ns_t_XX")
        )

        # Set to NaN where mask is False
        df = df.with_columns(
            pl.when(mask_expr)
            .then(pl.col(f"count_cohort_{i}_ns_t_XX"))
            .otherwise(pl.lit(None))
            .alias(f"count_cohort_{i}_ns_t_XX")
        )

        df = df.drop("_mask_den")

        # 5. Numerator of the mean: sum of weighted diff_y over the same mask
        mask_expr = pl.col(f"dof_ns_{i}_XX") == 1

        import polars as pl

        dof_ns_col   = f"dof_ns_{i}_XX"
        diff_yN_col  = f"diff_y_{i}_N_gt_XX"
        total_col    = f"total_cohort_{i}_ns_t_XX"

        df = df.with_columns(
            pl.when(pl.col(dof_ns_col) == 1)
                # dof_ns == 1: keep diff_yN; but null stays null
                .then(pl.col(diff_yN_col))
                # dof_ns == 0: set to 0 regardless of diff_yN
                .otherwise(0)
                .alias("_val_num")
        )

        df = df.with_columns(
            pl.col("_val_num").sum().over(group_cols).alias(total_col)
        )

        # set to NaN where mask is False (dof_ns != 1)
        df = df.with_columns(
            pl.when(pl.col(dof_ns_col) == 1)
            .then(pl.col(total_col))
            .otherwise(pl.lit(None))
            .alias(total_col)
        )

        # optional: drop temp column
        df = df.drop("_val_num")


        # 6. Mean for never-switcher cohort
        df = df.with_columns(
            (pl.col(f"total_cohort_{i}_ns_t_XX") / pl.col(f"count_cohort_{i}_ns_t_XX"))
            .alias(f"mean_cohort_{i}_ns_t_XX")
        )

        # 7. DOF for cohort adjustment: count of valid dof observations
        if trends_nonparam is None:
            group_cols = ["d_sq_XX", "time_XX"]
        else:
            group_cols = ["d_sq_XX", "time_XX"] + trends_nonparam

        dof_ns_col = f"dof_ns_{i}_XX"

        if cluster is None or cluster == "":  # case: no cluster
            mask_expr_ns = pl.col(dof_ns_col) == 1
            df = df.with_columns(
                pl.when(mask_expr_ns)
                .then(pl.col(dof_ns_col).sum().over(group_cols))
                .otherwise(pl.lit(None))
                .alias(f"dof_cohort_{i}_ns_t_XX")
            )

        else:  # case: cluster is provided
            
            import polars as pl
            import numpy as np

            # Assuming `df` is a Polars DataFrame
            cluster_col = cluster
            group_vars = ["d_sq_XX"] + (trends_nonparam or []) + ["time_XX"]

            clust_dof = f"cluster_dof_{i}_ns_XX"
            dof_ns    = f"dof_ns_{i}_XX"
            out_col   = f"dof_cohort_{i}_ns_t_XX"

            group_vars = ["d_sq_XX"] + (trends_nonparam or []) + ["time_XX"]

            df = df.with_columns(
                    pl.when(pl.col(dof_ns) == 1)
                    .then(pl.col(cluster_col))
                    .otherwise(None)
                    .alias(clust_dof)
                )

            counts = (
                df.filter(pl.col(clust_dof).is_not_null())
                .group_by(group_vars)
                .agg(pl.col(clust_dof).n_unique().alias(out_col))
            )

            df = df.join(counts, on=group_vars, how="left")


        # --- If not less_conservative_se: compute mean_cohort_s and dof_cohort_s ---
        if not less_conservative_se:
            if trends_nonparam is None:
                group_vars = ['d_sq_XX', 'F_g_XX', 'd_fg_XX', f'distance_to_switch_{i}_XX']
            else:
                group_vars = ['d_sq_XX', 'F_g_XX', 'd_fg_XX', f'distance_to_switch_{i}_XX'] + list(trends_nonparam)

            # Column names
            dof_s      = f"dof_s_{i}_XX"
            count_col  = f"count_cohort_{i}_s_t_XX"
            total_col  = f"total_cohort_{i}_s_t_XX"
            mean_col   = f"mean_cohort_{i}_s_t_XX"
            dof_cohort = f"dof_cohort_{i}_s_t_XX"   # (not used yet in this snippet)
            diff_y     = f"diff_y_{i}_N_gt_XX"

            # --- mask: dof_s == 1 ---
            mask_expr = pl.col(dof_s) == 1

            # --- Denominator: sum of N_gt_XX over rows with mask, by group_vars ---
            den_df = (
                df
                .filter(mask_expr)
                .group_by(group_vars)
                .agg(
                    pl.col("N_gt_XX").sum().alias(count_col)
                )
            )

            # Merge denominator back to df (left join on group_vars)
            df = df.join(den_df, on=group_vars, how="left")

            # --- Numerator: sum of diff_y over rows with mask, by group_vars ---
            num_df = (
                df
                .filter(mask_expr)
                .group_by(group_vars)
                .agg(
                    pl.col(diff_y).sum().alias(total_col)
                )
            )

            # Merge numerator back to df
            df = df.join(num_df, on=group_vars, how="left")

            # --- Mean (numerator / denominator) ---
            df = df.with_columns(
                (pl.col(total_col) / pl.col(count_col)).alias(mean_col)
            )

        # === DOF adjustment for s cohort ===
        if not less_conservative_se:
            if not cluster:
                df_dof_s = (
                    df.filter(pl.col(dof_s) == 1)
                      .group_by(group_vars)
                      .agg(pl.col(dof_s).sum().alias(dof_cohort))
                )
                if df_dof_s.height > 0:
                    df = df.join(df_dof_s, on=group_vars, how="left")
                else:
                    df = df.with_columns(pl.lit(None).alias(dof_cohort))
            else:
                cluster_dof = f"cluster_dof_{i}_s_XX"
                df = df.with_columns(
                    pl.when(pl.col(dof_s) == 1)
                    .then(pl.col(cluster))
                    .otherwise(None)
                    .alias(cluster_dof)
                )

                df_dof_s = (
                    df.filter(pl.col(cluster_dof).is_not_null())
                      .group_by(group_vars)
                      .agg(pl.col(cluster_dof).n_unique().alias(dof_cohort))
                )

                if df_dof_s.height > 0:
                    df = df.join(df_dof_s, on=group_vars, how="left")
                else:
                    df = df.with_columns(pl.lit(None).alias(dof_cohort))

        else:
            # Apply the less conservative standard error adjustment
            df = apply_less_conservative_se(
                df=df,
                i=int(i),
                trends_nonparam=trends_nonparam,
                less_conservative_se=less_conservative_se,
                cluster_col=cluster
            )

            # Align fallback DOF selection with fullpath logic
            dof_s2 = df[f'dof_cohort_{i}_s2_t_XX']
            dof_s1 = df[f'dof_cohort_{i}_s1_t_XX']
            dof_s0 = df[f'dof_cohort_{i}_s0_t_XX']

            df = df.with_columns(
                pl.when(pl.col(f'cohort_fullpath_{i}_XX') == 1).then(dof_s2)
                .when(pl.col('cohort_fullpath_1_XX') == 1).then(dof_s1)
                .otherwise(dof_s0)
                .alias(f'dof_cohort_{i}_s_t_XX')
            )
        # Generating Variables for Standard Error Calculation
        df = compute_ns_s_means_with_nans(df, i=i, trends_nonparam=trends_nonparam)
        df = compute_dof_cohort_ns_s(df, i=i, cluster_col=cluster, trends_nonparam=trends_nonparam)
        df = compute_E_hat_gt_with_nans_pl(df, i)
        df = compute_DOF_gt_with_nans_pl(df, i)
        
        # --- 8. Generate U_Gg_i ---
        N_val = dict_vars_gen[f"N{increase_XX}_{i}_XX"]
        if N_val != 0:

            # Dummy for not-yet-treated groups
            df = df.with_columns(
                ((i <= (pl.col("T_g_XX") - 1)).cast(pl.Int32)).alias(f"dummy_U_Gg{i}_XX")
            )


            # ==== Define column names ====
            col_temp     = f"U_Gg{i}_temp_XX"
            col_final    = f"U_Gg{i}_XX"
            col_dummy    = f"dummy_U_Gg{i}_XX"
            col_Ni       = f"N{increase_XX}_{i}_XX"
            col_dist     = f"distance_to_switch_{i}_XX"
            col_Nit_g    = f"N{increase_XX}_t_{i}_g_XX"
            col_Nctrl    = f"N_gt_control_{i}_XX"
            col_never    = f"never_change_d_{i}_XX"
            col_diff     = f"diff_y_{i}_XX"
            col_count    = f"count{i}_core_XX"
            col_temp_var = f"U_Gg{i}_temp_var_XX"
            col_DOF      = f"DOF_gt_{i}_XX"
            col_Ehat     = f"E_hat_gt_{i}_XX"

            # If N_val and G_XX are scalars; if they are columns, drop pl.lit()
            df = df.with_columns([
                pl.lit(N_val).alias(col_Ni),
                pl.lit(G_XX).alias("G_XX"),
            ])

            # ==== Step 1: Generate U_Gg{i}_temp_XX ====
            # Safe division: when N_gt_control is 0, the ratio term is 0 (matches Stata behavior)
            safe_ratio = pl.when(pl.col(col_Nctrl) != 0).then(
                (pl.col(col_Nit_g) / pl.col(col_Nctrl)) * pl.col(col_never)
            ).otherwise(0.0)

            df = df.with_columns(
                (
                    pl.col(col_dummy)
                    * (pl.col("G_XX") / pl.col(col_Ni))
                    * (
                        (pl.col("time_XX") >= (i + 1))
                        & (pl.col("time_XX") <= pl.col("T_g_XX"))
                    ).cast(pl.Int64)         # indicator as 0/1
                    * pl.col("N_gt_XX")
                    * (pl.col(col_dist) - safe_ratio)
                ).alias(col_temp)
            )

            # ==== Step 2: Multiply by diff_y ====
            df = df.with_columns(
                (pl.col(col_temp) * pl.col(col_diff)).alias(col_temp)
            )

            # ==== Step 3: Group total (gegen … total) ====
            df = df.with_columns(
                pl.col(col_temp).sum().over("group_XX").alias(col_final)
            )


            # ==== Step 4: Multiply by first_obs_by_gp_XX ====
            df = df.with_columns(
                (pl.col(col_final) * pl.col("first_obs_by_gp_XX")).alias(col_final)
            )

            # ==== Step 5: Count core ====
            condition = (
                (pl.col(col_temp).is_not_null() & (pl.col(col_temp) != 0))
                |
                (
                    (pl.col(col_temp) == 0) & (pl.col(col_diff) == 0) &
                    (
                        (pl.col(col_dist) != 0)
                        |
                        ((pl.col(col_Nit_g) != 0) & (pl.col(col_never) != 0))
                    )
                )
            )

            df = df.with_columns(
                pl.when(condition)
                .then(pl.col("N_gt_XX"))
                .otherwise(0)
                .alias(col_count)
            )

            # ==== Step 6 & 7: Final computation into temp_var ====
            # Use same safe_ratio from Step 1 (already defined above)
            df = df.with_columns(
                (
                    pl.col(col_dummy)
                    * (pl.col("G_XX") / pl.col(col_Ni))
                    * (pl.col(col_dist) - safe_ratio)
                    * (
                        (pl.col("time_XX") >= (i + 1))
                        & (pl.col("time_XX") <= pl.col("T_g_XX"))
                    ).cast(pl.Int64)   # indicator 0/1
                    * pl.col("N_gt_XX")
                    * pl.col(col_DOF)
                    * (pl.col(col_diff) - pl.col(col_Ehat))
                ).alias(col_temp_var)
            )

        # --- 6. Adjustment for controls, if any ---
        if controls:
            part2 = f"part2_switch{increase_XX}_{i}_XX"
            df = df.with_columns(pl.lit(0.0).alias(part2))

            for l in levels_d_sq_XX:
                l = int(l)
                if dict_glob[f"useful_res_{l}_XX"] > 1:

                    col_combined_temp = f"combined{increase_XX}_temp_{l}_{i}_XX"
                    df = df.drop([c for c in [col_combined_temp] if c in df.columns])
                    df = df.with_columns(pl.lit(0.0).alias(col_combined_temp))

                    for j in range(1, count_controls + 1):
                        col_in_brackets_j = f"in_brackets_{l}_{j}_XX"
                        if col_in_brackets_j not in df.columns:
                            df = df.with_columns(pl.lit(0.0).alias(col_in_brackets_j))

                        for k in range(1, count_controls + 1):
                            col_in_brackets_temp = f"in_brackets_{l}_{j}_{k}_temp_XX"
                            df = df.drop([c for c in [col_in_brackets_temp] if c in df.columns])

                            coef_jk = dict_glob[f"inv_Denom_{l}_XX"][j-1, k-1]
                            mask = (pl.col("d_sq_int_XX") == l) & (pl.col("F_g_XX") >= 3)

                            df = df.with_columns(
                                pl.when(mask)
                                .then(coef_jk * pl.col(f"in_sum_{k}_{l}_XX"))
                                .otherwise(0.0)
                                .alias(col_in_brackets_temp)
                            )

                            df = df.with_columns(
                                (pl.col(col_in_brackets_j) + pl.col(col_in_brackets_temp)).alias(col_in_brackets_j)
                            )

                        coef_theta = float(np.array(dict_glob[f"coefs_sq_{l}_XX"]).reshape(-1, 1)[j-1, 0])
                        df = df.with_columns(
                            (pl.col(col_in_brackets_j) - coef_theta).alias(col_in_brackets_j)
                        )

                        col_combined_j = f"combined{increase_XX}_temp_{l}_{j}_{i}_XX"
                        df = df.drop([c for c in [col_combined_j] if c in df.columns])
                        df = df.with_columns(
                            (pl.col(f"M{increase_XX}_{l}_{j}_{i}_XX") * pl.col(col_in_brackets_j)).alias(col_combined_j)
                        )

                        df = df.with_columns(
                            (pl.col(col_combined_temp) + pl.col(col_combined_j)).alias(col_combined_temp)
                        )

                    col_part2 = f"part2_switch{increase_XX}_{i}_XX"
                    if col_part2 not in df.columns:
                        df = df.with_columns(pl.lit(0.0).alias(col_part2))
                    df = df.with_columns(
                        (pl.col(col_part2) + pl.col(col_combined_temp)).alias(col_part2)
                    )

        # --- 7. Sum U_Gg_var over time by group ---
        df = df.with_columns(
            pl.col(f"U_Gg{i}_temp_var_XX").sum().over("group_XX").alias(f"U_Gg{i}_var_XX")
        )

        # --- 8. Adjustment for controls, if any ---
        if controls:
            if increase_XX == 1:
                df = df.with_columns(
                    (pl.col(f"U_Gg{i}_var_XX") - pl.col(f"part2_switch1_{i}_XX")).alias(f"U_Gg{i}_var_XX")
                )
            elif increase_XX == 0:
                df = df.with_columns(
                    (pl.col(f"U_Gg{i}_var_XX") - pl.col(f"part2_switch0_{i}_XX")).alias(f"U_Gg{i}_var_XX")
                )

        # --- inside your loop over i ---
        if normalized:
            mask = (
                (pl.col("time_XX") >= pl.col("F_g_XX"))
                & (pl.col("time_XX") <= pl.col("F_g_XX") - 1 + i)
                & (pl.col("S_g_XX") == increase_XX)
            )

            if continuous == 0:
                df = df.with_columns(
                    pl.when(mask)
                    .then(pl.col("treatment_XX") - pl.col("d_sq_XX"))
                    .otherwise(None)
                    .alias("sum_temp_XX")
                )
            elif continuous > 0:
                df = df.with_columns(
                    pl.when(mask)
                    .then(pl.col("treatment_XX_orig") - pl.col("d_sq_XX_orig"))
                    .otherwise(None)
                    .alias("sum_temp_XX")
                )

            # 2. sum up by group
            df = df.with_columns(
                pl.col("sum_temp_XX").sum().over("group_XX").alias(f"sum_treat_until_{i}_XX")
            )

            # 3. drop helper
            df = df.drop("sum_temp_XX")

            # 4. compute delta_D_i_cum_temp_XX
            N_val = dict_vars_gen[f"N{increase_XX}_{i}_XX"]
            ratio = pl.col("N_gt_XX") / N_val
            switch_mask = pl.col(f"distance_to_switch_{i}_XX") == 1

            df = df.with_columns(
                pl.when(switch_mask)
                .then(
                    ratio
                    * (
                        pl.col("S_g_XX") * pl.col(f"sum_treat_until_{i}_XX")
                        + (1 - pl.col("S_g_XX")) * (-pl.col(f"sum_treat_until_{i}_XX"))
                    )
                )
                .otherwise(None)
                .alias(f"delta_D_{i}_cum_temp_XX")
            )

            # 5. store scalar
            dict_vars_gen[f"delta_norm_{i}_XX"] = (
                df.select(pl.col(f"delta_D_{i}_cum_temp_XX").sum()).item()
            )

    # 1. Compute Ntrendslin (same Python logic, no Polars needed here)
    Ntrendslin = 1
    Ntrendslin = min(
        int(dict_vars_gen[f"N{increase_XX}_{i}_XX"])
        for i in range(1, int(l_u_a_XX + 1))
    )

    # 2. If linear trends requested and there's at least one valid path
    if trends_lin and Ntrendslin != 0:

        # print(f"This is the l_u_a_XX {l_u_a_XX} value")
        # print("accion extra")

        lu = int(l_u_a_XX)
        col_TL     = f"U_Gg{lu}_TL"
        col_var_TL = f"U_Gg{lu}_var_TL"
        col_XX     = f"U_Gg{lu}_XX"
        col_var_XX = f"U_Gg{lu}_var_XX"

        # Drop TL columns if they exist
        for c in [col_TL, col_var_TL]:
            if c in df.columns:
                df = df.drop(c)

        # Initialize TL columns to 0
        df = df.with_columns([
            pl.lit(0.0).alias(col_TL),
            pl.lit(0.0).alias(col_var_TL),
        ])

        # List all columns starting with 'U_Gg' (for debugging, like in your pandas code)
        cols_U_Gg = [col for col in df.columns if col.startswith("U_Gg")]
        # print(cols_U_Gg)

        # Accumulate over i = 1..l_u_a_XX
        for i in range(1, lu + 1):
            # print(col_TL)
            df = df.with_columns([
                (pl.col(col_TL) + pl.col(f"U_Gg{i}_XX")).alias(col_TL),
                (pl.col(col_var_TL) + pl.col(f"U_Gg{i}_var_XX")).alias(col_var_TL),
            ])

        # Copy totals into the final columns
        df = df.with_columns([
            pl.col(col_TL).alias(col_XX),
            pl.col(col_var_TL).alias(col_var_XX),
        ])



        


    if placebo != 0:

        if l_placebo_u_a_XX >= 1:

            for i in range(1, int(l_placebo_u_a_XX + 1)):
                i = int(i)

                # 1) Drop old columns
                cols_to_drop = [
                    f"diff_y_pl_{i}_XX", f"U_Gg_pl_{i}_temp_XX", f"U_Gg_placebo_{i}_XX",
                    f"U_Gg_pl_{i}_temp_var_XX", f"U_Gg_pl_{i}_var_XX",
                    f"mean_diff_y_pl_{i}_nd_sq_t_XX", f"mean_diff_y_pl_{i}_d_sq_t_XX",
                    f"count_diff_y_pl_{i}_nd_sq_t_XX", f"count_diff_y_pl_{i}_d_sq_t_XX",
                    f"dist_to_switch_pl_{i}_XX", f"never_change_d_pl_{i}_XX",
                    f"N{increase_XX}_t_placebo_{i}_XX", f"N{increase_XX}_t_placebo_{i}_g_XX",
                    f"N_gt_control_placebo_{i}_XX", f"dummy_U_Gg_pl_{i}_XX",
                    f"never_change_d_pl_{i}_wXX", f"dist_to_switch_pl_{i}_wXX",
                    f"dof_cohort_pl_{i}_ns_t_XX", f"count_cohort_pl_{i}_ns_t_XX",
                    f"total_cohort_pl_{i}_ns_t_XX", f"mean_cohort_pl_{i}_ns_t_XX",
                    f"dof_cohort_pl_{i}_s_t_XX", f"dof_cohort_pl_{i}_s0_t_XX",
                    f"dof_cohort_pl_{i}_s1_t_XX", f"dof_cohort_pl_{i}_s2_t_XX",
                    f"count_cohort_pl_{i}_s_t_XX", f"count_cohort_pl_{i}_s0_t_XX",
                    f"count_cohort_pl_{i}_s1_t_XX", f"count_cohort_pl_{i}_s2_t_XX",
                    f"total_cohort_pl_{i}_s_t_XX", f"total_cohort_pl_{i}_s0_t_XX",
                    f"total_cohort_pl_{i}_s1_t_XX", f"total_cohort_pl_{i}_s2_t_XX",
                    f"mean_cohort_pl_{i}_s_t_XX", f"mean_cohort_pl_{i}_s0_t_XX",
                    f"mean_cohort_pl_{i}_s1_t_XX", f"mean_cohort_pl_{i}_s2_t_XX",
                ]
                df = df.drop([c for c in cols_to_drop if c in df.columns])

                # 2) Long difference (placebo i)
                df = df.with_columns(
                    (
                        pl.col("outcome_XX").shift(2 * i).over("group_XX")
                        - pl.col("outcome_XX").shift(i).over("group_XX")
                    ).alias(f"diff_y_pl_{i}_XX")
                )

                # 3) Identify controls for placebo
                df = df.with_columns(
                    (
                        pl.col(f"never_change_d_{i}_XX")
                        * pl.col(f"diff_y_pl_{i}_XX").is_not_null().cast(pl.Int32)
                    ).alias(f"never_change_d_pl_{i}_XX")
                ).with_columns(
                    pl.when(pl.col(f"never_change_d_{i}_XX").is_null())
                    .then(None)
                    .otherwise(pl.col(f"never_change_d_pl_{i}_XX"))
                    .alias(f"never_change_d_pl_{i}_XX")
                )

                # 4) Weighted control counts
                df = df.with_columns(
                    (
                        pl.col(f"never_change_d_pl_{i}_XX") * pl.col("N_gt_XX")
                    ).alias(f"never_change_d_pl_{i}_wXX")
                )

                cols_gr_sel = (
                    ["time_XX", "d_sq_XX"]
                    if trends_nonparam is None
                    else ["time_XX", "d_sq_XX"] + trends_nonparam
                )

                df = df.with_columns(
                    pl.col(f"never_change_d_pl_{i}_wXX")
                    .sum()
                    .over(cols_gr_sel)
                    .alias(f"N_gt_control_placebo_{i}_XX")
                )

                # 5) Distance-to-switch placebo
                col_dist = f"distance_to_switch_{i}_XX"
                col_diff = f"diff_y_pl_{i}_XX"
                col_N = f"N_gt_control_placebo_{i}_XX"
                col_res = f"dist_to_switch_pl_{i}_XX"

                df = df.with_columns(
                    (
                        pl.col(col_dist)
                        * (
                            (pl.col(col_diff).is_not_null())
                            & (pl.col(col_N).is_not_null())
                            & (pl.col(col_N) > 0)
                        ).cast(pl.Int32)
                    ).alias(col_res)
                ).with_columns(
                    pl.when(pl.col(col_dist).is_null() | pl.col(col_diff).is_null())
                    .then(None)
                    .otherwise(pl.col(col_res))
                    .alias(col_res)
                )

                if same_switchers_pl:
                    df = df.with_columns(
                        (
                            pl.col(f"dist_to_switch_pl_{i}_XX")
                            * pl.col("fillin_g_pl_XX")
                        ).alias(f"dist_to_switch_pl_{i}_XX")
                    )

                df = df.with_columns(
                    (
                        pl.col(f"dist_to_switch_pl_{i}_XX") * pl.col("N_gt_XX")
                    ).alias(f"dist_to_switch_pl_{i}_wXX")
                )

                # 6) Counts by time
                df = df.with_columns(
                    [
                        pl.col(f"dist_to_switch_pl_{i}_wXX")
                        .sum()
                        .over("time_XX")
                        .alias(f"N{increase_XX}_t_placebo_{i}_XX"),
                        pl.col(f"dist_to_switch_pl_{i}_XX")
                        .sum()
                        .over("time_XX")
                        .alias(f"N{increase_XX}_t_placebo_{i}_dwXX"),
                    ]
                )

                # 7) Scalar N_placebo_i
                n_placebo = 0.0
                n_dw_placebo = 0.0

                for t in range(int(t_min_XX), int(T_max_XX + 1)):
                    df_t = df.filter(pl.col("time_XX") == t)
                    n_placebo += (
                        df_t.select(pl.col(f"N{increase_XX}_t_placebo_{i}_XX").mean()).item()
                        or 0.0
                    )
                    n_dw_placebo += (
                        df_t.select(pl.col(f"N{increase_XX}_t_placebo_{i}_dwXX").mean()).item()
                        or 0.0
                    )

                dict_vars_gen[f"N{increase_XX}_placebo_{i}_XX"] = n_placebo
                dict_vars_gen[f"N{increase_XX}_dw_placebo_{i}_XX"] = n_dw_placebo

                # 8) Group-by cohort
                df = df.with_columns(
                    pl.col(f"dist_to_switch_pl_{i}_wXX")
                    .sum()
                    .over(cols_gr_sel)
                    .alias(f"N{increase_XX}_t_placebo_{i}_g_XX")
                )

                # --- Controls section ---
                if controls:
                    df = df.with_columns(
                        pl.lit(0.0).alias(f"part2_pl_switch{increase_XX}_{i}_XX")
                    )

                    for j, var in enumerate(controls, start=1):
                        j = int(j)

                        # diff_X
                        df = df.with_columns(
                            (
                                pl.col(var).shift(2 * i).over("group_XX")
                                - pl.col(var).shift(i).over("group_XX")
                            ).alias(f"diff_X_{j}_placebo_{i}_XX")
                        )

                        df = df.with_columns(
                            (
                                pl.col("N_gt_XX") * pl.col(f"diff_X_{j}_placebo_{i}_XX")
                            ).alias(f"diff_X{j}_pl_{i}_N_XX")
                        )

                        for l in levels_d_sq_XX:
                            l = int(l)
                            mask_l = pl.col("d_sq_int_XX") == l
                            denom = dict_vars_gen.get(f"N{increase_XX}_placebo_{i}_XX", 0.0)

                            if denom == 0 or denom is None:
                                scale_val = 0.0
                            else:
                                scale_val = float(G_XX) / float(denom)

                            df = df.with_columns(
                                (
                                    ((pl.col("T_g_XX") - 2) >= i).cast(pl.Int32)
                                    * mask_l.cast(pl.Int32)
                                    * pl.lit(scale_val)
                                    * (
                                        pl.col(f"dist_to_switch_pl_{i}_XX")
                                        - _safe_div(
                                            pl.col(f"N{increase_XX}_t_placebo_{i}_g_XX"),
                                            pl.col(f"N_gt_control_placebo_{i}_XX"),
                                        )
                                        * pl.col(f"never_change_d_pl_{i}_XX")
                                    )
                                    * (
                                        (pl.col("time_XX") >= (i + 1))
                                        & (pl.col("time_XX") <= pl.col("T_g_XX"))
                                    ).cast(pl.Int32)
                                    * pl.col(f"diff_X{j}_pl_{i}_N_XX")
                                ).alias(f"m{increase_XX}_pl_g_{l}_{j}_{i}_XX")
                            )

                            df = df.with_columns(
                                pl.col(f"m{increase_XX}_pl_g_{l}_{j}_{i}_XX")
                                .sum()
                                .over("group_XX")
                                .alias(f"m_pl{increase_XX}_{l}_{j}_{i}_XX")
                            )

                            df = df.with_columns(
                                pl.when(pl.col("first_obs_by_gp_XX") == 1)
                                .then(pl.col(f"m_pl{increase_XX}_{l}_{j}_{i}_XX"))
                                .otherwise(None)
                                .alias(f"m_pl{increase_XX}_{l}_{j}_{i}_XX")
                            )

                            df = df.with_columns(
                                _safe_div(
                                    pl.col(f"m_pl{increase_XX}_{l}_{j}_{i}_XX").sum(),
                                    pl.lit(G_XX),
                                ).alias(f"M_pl{increase_XX}_{l}_{j}_{i}_XX")
                            )
                        # --- Placebo controls & DOF / cohort computations (robust Polars version) ---
                        for l in levels_d_sq_XX:
                            l = int(l)
                            if dict_glob.get(f"useful_res_{l}_XX", 0) > 1:
                                # English comment: retrieve coefficient vector for this d_sq level
                                coefs = np.array(dict_glob[f"coefs_sq_{l}_XX"]).reshape(-1, 1)

                                # English comment: create an adjusted diff_y_pl column that subtracts coef * diff_X when d_sq_int matches
                                # We create a temp column to avoid potential overwrite/read-order issues
                                adj_col = f"diff_y_pl_{i}_XX_adj_{l}_temp"
                                df = df.with_columns(
                                    pl.when(pl.col("d_sq_int_XX") == l)
                                    .then(pl.col(f"diff_y_pl_{i}_XX") - pl.lit(float(coefs[j - 1, 0])) * pl.col(f"diff_X_{j}_placebo_{i}_XX"))
                                    .otherwise(pl.col(f"diff_y_pl_{i}_XX"))
                                    .alias(adj_col)
                                )

                                # Replace original column with adjusted temp (drop temp after)
                                df = df.with_columns(
                                    pl.col(adj_col).alias(f"diff_y_pl_{i}_XX")
                                ).drop(adj_col)

                                # English comment: init in_brackets placeholder for this l and j (used later by controls logic)
                                in_brackets_col = f"in_brackets_pl_{l}_{j}_XX"
                                if in_brackets_col not in df.columns:
                                    df = df.with_columns(pl.lit(0).alias(in_brackets_col))


                # --- 1. weighted placebo differences + dof_ns_pl, dof_s_pl ---

                diff_y_pl_col     = f"diff_y_pl_{i}_XX"
                diff_y_pl_N_col   = f"diff_y_pl_{i}_N_gt_XX"
                dof_ns_pl         = f"dof_ns_pl_{i}_XX"
                dof_s_pl          = f"dof_s_pl_{i}_XX"
                never_change_col  = f"never_change_d_pl_{i}_XX"
                N_t_placebo_col   = f"N{increase_XX}_t_placebo_{i}_XX"
                dist_to_switch_pl = f"dist_to_switch_pl_{i}_XX"

                # diff_y_pl * N_gt_XX
                df = df.with_columns(
                    (pl.col(diff_y_pl_col) * pl.col("N_gt_XX")).alias(diff_y_pl_N_col)
                )

                # dof_ns_pl: indicator (0/1)
                df = df.with_columns(
                    (
                        (pl.col("N_gt_XX") != 0)
                        & pl.col(diff_y_pl_col).is_not_null()
                        & (pl.col(never_change_col) == 1)
                        & (pl.col(N_t_placebo_col) > 0)
                        & pl.col(N_t_placebo_col).is_not_null()
                    ).cast(pl.Int64).alias(dof_ns_pl)
                )

                # dof_s_pl: indicator (0/1)
                df = df.with_columns(
                        (
                            (pl.col("N_gt_XX") != 0) &
                            (pl.col(dist_to_switch_pl) == 1)
                        )
                        .cast(pl.Int64)          # 1 / 0 / null
                        .fill_null(0)            # <-- null becomes 0
                        .alias(dof_s_pl)
                    )

                # --- 2. Groups for ns-cohort averages ---

                if trends_nonparam is None:
                    group_cols = ["d_sq_XX", "time_XX"]
                else:
                    group_cols = ["d_sq_XX", "time_XX"] + trends_nonparam

                mask_ns_expr = pl.col(dof_ns_pl) == 1

                # count_cohort_pl_{i}_ns_t_XX: sum of N_gt_XX over group_cols, only for mask_ns
                count_ns_col = f"count_cohort_pl_{i}_ns_t_XX"
                df = df.with_columns(
                    pl.when(mask_ns_expr)
                    .then(pl.col("N_gt_XX"))
                    .otherwise(0)
                    .alias("_N_ns_tmp")
                )
                df = df.with_columns(
                    pl.when(mask_ns_expr)
                    .then(pl.col("_N_ns_tmp").sum().over(group_cols))
                    .otherwise(pl.lit(None))
                    .alias(count_ns_col)
                ).drop("_N_ns_tmp")

                # total_cohort_pl_{i}_ns_t_XX: sum of diff_y_pl_N, only for mask_ns
                total_ns_col = f"total_cohort_pl_{i}_ns_t_XX"
                df = df.with_columns(
                    pl.when(mask_ns_expr)
                    .then(pl.col(diff_y_pl_N_col))
                    .otherwise(0)
                    .alias("_diffN_ns_tmp")
                )
                df = df.with_columns(
                    pl.when(mask_ns_expr)
                    .then(pl.col("_diffN_ns_tmp").sum().over(group_cols))
                    .otherwise(pl.lit(None))
                    .alias(total_ns_col)
                ).drop("_diffN_ns_tmp")

                # mean_cohort_pl_{i}_ns_t_XX
                mean_ns_col = f"mean_cohort_pl_{i}_ns_t_XX"
                df = df.with_columns(
                    (pl.col(total_ns_col) / pl.col(count_ns_col)).alias(mean_ns_col)
                )

                # --- 3. DOF for ns-cohort (dof_cohort_pl_{i}_ns_t_XX) ---

                col_dof_ns      = dof_ns_pl
                col_dof_coh_ns  = f"dof_cohort_pl_{i}_ns_t_XX"

                if trends_nonparam is None:
                    group_keys = ["d_sq_XX", "time_XX"]
                else:
                    group_keys = ["d_sq_XX"] + trends_nonparam + ["time_XX"]

                if cluster is None or cluster == "":
                    # sum of dof_ns_pl within group_keys, but only where dof_ns_pl == 1
                    df = df.with_columns(
                        pl.when(pl.col(col_dof_ns) == 1)
                        .then(pl.col(col_dof_ns).sum().over(group_keys))
                        .otherwise(pl.lit(None))
                        .cast(pl.Float64)
                        .alias(col_dof_coh_ns)
                    )
                else:
                    # Case 2: with cluster – unique clusters among rows with dof_ns_pl == 1
                    clust_dof = f"cluster_dof_pl_{i}_ns_XX"
                    df = df.with_columns(
                        pl.when(pl.col(col_dof_ns) == 1)
                        .then(pl.col(cluster))
                        .otherwise(pl.lit(None))
                        .alias(clust_dof)
                    )

                    agg = (
                        df.filter(pl.col(clust_dof).is_not_null())
                        .group_by(group_keys)
                        .agg(pl.col(clust_dof).n_unique().alias(col_dof_coh_ns))
                    )

                    df = df.join(agg, on=group_keys, how="left")

                    # rows where clust_dof is null -> dof_coh_ns = NaN
                    df = df.with_columns(
                        pl.when(pl.col(clust_dof).is_null())
                        .then(pl.lit(None))
                        .otherwise(pl.col(col_dof_coh_ns))
                        .alias(col_dof_coh_ns)
                    )

                # --- 4. Switchers cohort (C_k) demeaning ---

                mask_sw_expr = pl.col(dof_s_pl) == 1

                if trends_nonparam is None:
                    group_cols_sw = ["d_sq_XX", "F_g_XX", "d_fg_XX"]
                else:
                    group_cols_sw = ["d_sq_XX", "F_g_XX", "d_fg_XX"] + trends_nonparam

                # denominator: sum N_gt_XX within group_cols_sw for mask_sw
                count_sw_col = f"count_cohort_pl_{i}_s_t_XX"
                df = df.with_columns(
                    pl.when(mask_sw_expr)
                    .then(pl.col("N_gt_XX"))
                    .otherwise(0)
                    .alias("_N_s_tmp")
                )
                df = df.with_columns(
                    pl.when(mask_sw_expr)
                    .then(pl.col("_N_s_tmp").sum().over(group_cols_sw))
                    .otherwise(pl.lit(None))
                    .alias(count_sw_col)
                ).drop("_N_s_tmp")

                # numerator: sum diff_y_pl_N within group_cols_sw for mask_sw
                total_sw_col = f"total_cohort_pl_{i}_s_t_XX"
                df = df.with_columns(
                    pl.when(mask_sw_expr)
                    .then(pl.col(diff_y_pl_N_col))
                    .otherwise(0)
                    .alias("_diffN_s_tmp")
                )
                df = df.with_columns(
                    pl.when(mask_sw_expr)
                    .then(pl.col("_diffN_s_tmp").sum().over(group_cols_sw))
                    .otherwise(pl.lit(None))
                    .alias(total_sw_col)
                ).drop("_diffN_s_tmp")

                # mean
                mean_s_col = f"mean_cohort_pl_{i}_s_t_XX"
                df = df.with_columns(
                    (pl.col(total_sw_col) / pl.col(count_sw_col)).alias(mean_s_col)
                )

                # --- 5. DOF for switchers (dof_cohort_pl_{i}_s_t_XX) ---

                if trends_nonparam is None:
                    group_keys_s = ["d_sq_XX", "F_g_XX", "d_fg_XX"]
                else:
                    group_keys_s = ["d_sq_XX", "F_g_XX", "d_fg_XX"] + trends_nonparam

                col_dof_cohs = f"dof_cohort_pl_{i}_s_t_XX"
                col_dof_s    = dof_s_pl

                if cluster is None or cluster == "":
                    df = df.with_columns(
                        pl.when(pl.col(col_dof_s) == 1)
                        .then(pl.col(col_dof_s).sum().over(group_keys_s))
                        .otherwise(pl.lit(None))
                        .alias(col_dof_cohs)
                    )
                else:
                    clust_dof_s = f"cluster_dof_pl_{i}_s_XX"
                    df = df.with_columns(
                        pl.when(pl.col(col_dof_s) == 1)
                        .then(pl.col(cluster))
                        .otherwise(pl.lit(None))
                        .alias(clust_dof_s)
                    )

                    agg_s = (
                        df.filter(pl.col(clust_dof_s).is_not_null())
                        .group_by(group_keys_s)
                        .agg(pl.col(clust_dof_s).n_unique().alias(col_dof_cohs))
                    )

                    df = df.join(agg_s, on=group_keys_s, how="left")
                    df = df.with_columns(
                        pl.when(pl.col(clust_dof_s).is_null())
                        .then(pl.lit(None))
                        .otherwise(pl.col(col_dof_cohs))
                        .alias(col_dof_cohs)
                    )

                # --- 6. Union of switchers and not-yet switchers (ns_s) ---

                if trends_nonparam is None:
                    group_keys_any = ["d_sq_XX", "time_XX"]
                else:
                    group_keys_any = ["d_sq_XX", "time_XX"] + trends_nonparam

                # print(group_keys_any)

                col_dof_s    = dof_s_pl
                col_dof_ns   = dof_ns_pl
                col_dof_any  = f"dof_ns_s_pl_{i}_XX"

                col_N      = "N_gt_XX"
                col_diffN  = diff_y_pl_N_col

                col_count  = f"count_cohort_pl_{i}_ns_s_t_XX"
                col_total  = f"total_cohort_pl_{i}_ns_s_t_XX"
                col_mean   = f"mean_cohort_pl_{i}_ns_s_t_XX"

                # mask_dof = (dof_s == 1) OR (dof_ns == 1)
                mask_dof_expr = ((pl.col(col_dof_s) == 1) | (pl.col(col_dof_ns) == 1))

                # numeric 1.0/0.0
                df = df.with_columns(
                    mask_dof_expr.cast(pl.Float64).alias(col_dof_any)
                )

                # set NaN where either dof_s or dof_ns is NaN
                df = df.with_columns(
                    pl.when(pl.col(col_dof_s).is_null() | pl.col(col_dof_ns).is_null())
                    .then(pl.lit(None))
                    .otherwise(pl.col(col_dof_any))
                    .alias(col_dof_any)
                )

                # group sums for union cohort (only among mask_dof rows)
                df = df.with_columns([
                    pl.when(mask_dof_expr)
                    .then(pl.col(col_N))
                    .otherwise(0)
                    .alias("_N_any_tmp"),
                    pl.when(mask_dof_expr)
                    .then(pl.col(col_diffN))
                    .otherwise(0)
                    .alias("_diffN_any_tmp"),
                ])

                df = df.with_columns([
                    pl.when(mask_dof_expr)
                    .then(pl.col("_N_any_tmp").sum().over(group_keys_any))
                    .otherwise(pl.lit(None))
                    .alias(col_count),
                    pl.when(mask_dof_expr)
                    .then(pl.col("_diffN_any_tmp").sum().over(group_keys_any))
                    .otherwise(pl.lit(None))
                    .alias(col_total),
                ]).drop(["_N_any_tmp", "_diffN_any_tmp"])

                df = df.with_columns(
                    (pl.col(col_total) / pl.col(col_count)).alias(col_mean)
                )

                col_dof_coh_any = f"dof_cohort_pl_{i}_ns_s_t_XX"
                # print(f"este es el {cluster}")

                if cluster is None or cluster == "":
                    df = df.with_columns(
                        pl.when(pl.col(col_dof_any) == 1)
                        .then(pl.col(col_dof_any).sum().over(group_keys_any))
                        .otherwise(pl.lit(None))
                        .alias(col_dof_coh_any)
                    )
                else:
                    # print("estoy corriendo el correcto code")

                    # dof_ns_s_pl_{i}_XX is the same logical indicator as col_dof_any
                    col_dof_union = col_dof_any
                    clust_dof_any = f"cluster_dof_pl_{i}_ns_s_XX"

                    df = df.with_columns(
                        pl.when(pl.col(col_dof_union) == 1)
                        .then(pl.col(cluster))
                        .otherwise(pl.lit(None))
                        .alias(clust_dof_any)
                    )

                    # Drop previous col_dof_coh_any if exists
                    if col_dof_coh_any in df.columns:
                        df = df.drop(col_dof_coh_any)

                    agg_any = (
                        df.filter(pl.col(clust_dof_any).is_not_null())
                        .group_by(group_keys_any)
                        .agg(pl.col(clust_dof_any).n_unique().alias(col_dof_coh_any))
                    )

                    df = df.join(agg_any, on=group_keys_any, how="left")

                    df = df.with_columns(
                        pl.when(pl.col(clust_dof_any).is_null())
                        .then(pl.lit(None))
                        .otherwise(pl.col(col_dof_coh_any))
                        .alias(col_dof_coh_any)
                    )



                # ---- Compute E_hat_gt and DOF_gt ----
                df = compute_E_hat_gt_with_nans_pl(df, i=i, type_sect="placebo")
                
                df = compute_DOF_gt_with_nans_pl(df, i=i, type_sect="placebo")

                # ---- dummy_U_Gg_pl ----
                df = df.with_columns([
                    (pl.lit(i) <= (pl.col("T_g_XX") - 1)).cast(pl.Int64).alias(f"dummy_U_Gg_pl_{i}_XX")
                ])

                N_placebo = dict_vars_gen[f"N{increase_XX}_placebo_{i}_XX"]

                # dummy_U_Gg_pl_i_XX = 1{ i <= T_g_XX - 1 }
                dummy_col = f"dummy_U_Gg_pl_{i}_XX"
                df = df.with_columns(
                    (pl.lit(i) <= (pl.col("T_g_XX") - 1)).cast(pl.Int64).alias(dummy_col)
                )

                N_placebo = dict_vars_gen[f"N{increase_XX}_placebo_{i}_XX"]
                if N_placebo != 0:
                    temp_col      = f"U_Gg_pl_{i}_temp_XX"
                    placebo_col   = f"U_Gg_placebo_{i}_XX"
                    count_col     = f"count{i}_pl_core_XX"
                    temp_var_col  = f"U_Gg_pl_{i}_temp_var_XX"

                    dist_col      = f"dist_to_switch_pl_{i}_XX"
                    Nt_col        = f"N{increase_XX}_t_placebo_{i}_g_XX"
                    Ngt_ctrl_col  = f"N_gt_control_placebo_{i}_XX"
                    never_col     = f"never_change_d_pl_{i}_XX"
                    diff_y_col    = f"diff_y_pl_{i}_XX"
                    dof_col       = f"DOF_gt_pl_{i}_XX"
                    E_hat_col     = f"E_hat_gt_pl_{i}_XX"

                    # Common pieces: (dist_to_switch - (N_t / N_gt_control) * never_change)
                    # Safe division: when N_gt_control is 0, the ratio term is 0 (matches Stata behavior)
                    safe_ratio_pl = pl.when(pl.col(Ngt_ctrl_col) != 0).then(
                        (pl.col(Nt_col) / pl.col(Ngt_ctrl_col)) * pl.col(never_col)
                    ).otherwise(0.0)

                    dist_expr = pl.col(dist_col) - safe_ratio_pl

                    # Time window 1{ time in [i+1, T_g_XX] }
                    time_window_expr = (
                        (pl.col("time_XX") >= (i + 1)) & (pl.col("time_XX") <= pl.col("T_g_XX"))
                    ).cast(pl.Float64)

                    # -----------------------------------------------
                    # U_Gg_pl_{i}_temp_XX
                    # -----------------------------------------------
                    df = df.with_columns(
                        (
                            pl.col(dummy_col).cast(pl.Float64)
                            * (G_XX / N_placebo)
                            * pl.col("N_gt_XX")
                            * dist_expr
                            * pl.col(diff_y_col)
                            * time_window_expr
                        ).alias(temp_col)
                    )

                    # -----------------------------------------------
                    # U_Gg_placebo_{i}_XX = sum_g temp * first_obs_by_gp_XX
                    # -----------------------------------------------
                    df = df.with_columns(
                        (
                            pl.col(temp_col).sum().over("group_XX")
                            * pl.col("first_obs_by_gp_XX")
                        ).alias(placebo_col)
                    )

                    # -----------------------------------------------
                    # count{i}_pl_core_XX
                    # -----------------------------------------------
                    # cond:
                    # (~isna(temp) & temp != 0)
                    # OR (temp == 0 & diff_y == 0 & (dist != 0 OR (N_t != 0 & never != 0)))
                    cond_expr = (
                        (
                            pl.col(temp_col).is_not_null() & (pl.col(temp_col) != 0)
                        )
                        | (
                            (pl.col(temp_col) == 0)
                            & (pl.col(diff_y_col) == 0)
                            & (
                                (pl.col(dist_col) != 0)
                                | (
                                    (pl.col(Nt_col) != 0)
                                    & (pl.col(never_col) != 0)
                                )
                            )
                        )
                    )

                    df = df.with_columns(
                        pl.when(cond_expr)
                        .then(pl.col("N_gt_XX"))
                        .otherwise(0.0)
                        .cast(pl.Float64)
                        .alias(count_col)
                    )

                    # The pandas code also had:
                    # sel_cols = [temp_col, diff_y_col, dist_col, never_col]
                    # idx = df[sel_cols].isna().sum(axis=1) == 4
                    # and then commented-out modification using idx.
                    # Since that line is commented, we do not need to replicate it.

                    # -----------------------------------------------
                    # U_Gg_pl_{i}_temp_var_XX
                    # -----------------------------------------------
                    df = df.with_columns(
                        (
                            pl.col(dummy_col).cast(pl.Float64)
                            * (G_XX / N_placebo)
                            * dist_expr
                            * time_window_expr
                            * pl.col("N_gt_XX")
                            * pl.col(dof_col)
                            * (pl.col(diff_y_col) - pl.col(E_hat_col))
                        ).alias(temp_var_col)
                    )


                    import polars as pl
                    import numpy as np

                    if controls is not None:
                        part2_col = f"part2_pl_switch{increase_XX}_{i}_XX"

                        # make sure accumulator exists
                        if part2_col not in df.columns:
                            df = df.with_columns(pl.lit(0.0).alias(part2_col))

                        for l in levels_d_sq_XX:
                            l = int(l)

                            # initialize the combined term for this l (per row)
                            combined_col = f"combined_pl{increase_XX}_temp_{l}_{i}_XX"
                            df = df.with_columns(pl.lit(0.0).alias(combined_col))

                            inv_Denom = np.array(dict_glob[f"inv_Denom_{l}_XX"])
                            coefsq = np.array(dict_glob[f"coefs_sq_{l}_XX"]).reshape(-1)

                            # mask: 1{ d_sq_int_XX == l & F_g_XX >= 3 }
                            mask_expr = (
                                (pl.col("d_sq_int_XX") == l) & (pl.col("F_g_XX") >= 3)
                            ).cast(pl.Float64)

                            for j in range(1, count_controls + 1):
                                j = int(j)
                                bracket_col = f"in_brackets_pl_{l}_{j}_XX"

                                # ----- build Σ_k inv_Denom[j-1,k-1] * in_sum_{k,l} * mask -----
                                expr = pl.lit(0.0)
                                for k in range(1, count_controls + 1):
                                    k = int(k)
                                    coeff = float(inv_Denom[j - 1, k - 1])
                                    in_sum_col = f"in_sum_{k}_{l}_XX"
                                    expr = expr + pl.lit(coeff) * pl.col(in_sum_col) * mask_expr

                                # Now set the bracket column *from scratch*:
                                # in_brackets_pl_{l,j} = Σ_k(...) - coefsq[j-1]
                                c_j = float(coefsq[j - 1])
                                df = df.with_columns(
                                    (expr - pl.lit(c_j)).alias(bracket_col)
                                )

                                # combined_pl_temp += M_pl * in_brackets
                                M_col = f"M_pl{increase_XX}_{l}_{j}_{i}_XX"
                                df = df.with_columns(
                                    (pl.col(combined_col) + pl.col(M_col) * pl.col(bracket_col)).alias(
                                        combined_col
                                    )
                                )

                            # after finishing j-loop for this l:
                            # part2_pl_switch += combined_pl_temp only for rows with d_sq_int_XX == l
                            df = df.with_columns(
                                (
                                    pl.col(part2_col)
                                    + pl.when(pl.col("d_sq_int_XX") == l)
                                        .then(pl.col(combined_col))
                                        .otherwise(0.0)
                                ).alias(part2_col)
                            )



                    temp_var_col = f"U_Gg_pl_{i}_temp_var_XX"
                    var_col      = f"U_Gg_pl_{i}_var_XX"

                    df = df.with_columns([
                        # ensure numeric (float)
                        pl.col(temp_var_col).cast(pl.Float64).alias(temp_var_col),

                        # sum variance term by group_XX
                        pl.col(temp_var_col).sum().over("group_XX").alias(var_col),
                    ])


                    # ---- final adjustment ----
                    if controls is not None:
                        if increase_XX == 1:
                            df = df.with_columns([
                                (pl.col(f"U_Gg_pl_{i}_var_XX") - pl.col(f"part2_pl_switch1_{i}_XX"))
                                .alias(f"U_Gg_pl_{i}_var_XX")
                            ])
                        elif increase_XX == 0:
                            df = df.with_columns([
                                (pl.col(f"U_Gg_pl_{i}_var_XX") - pl.col(f"part2_pl_switch0_{i}_XX"))
                                .alias(f"U_Gg_pl_{i}_var_XX")
                            ])
                # ---- normalized placebo ----
                if normalized:
                    cond = (
                        (pl.col("time_XX") >= pl.col("F_g_XX"))
                        & (pl.col("time_XX") <= (pl.col("F_g_XX") - 1 + i))
                        & (pl.col("S_g_XX") == increase_XX)
                    )
                    if continuous == 0:
                        df = df.with_columns([
                            pl.when(cond)
                            .then(pl.col("treatment_XX") - pl.col("d_sq_XX"))
                            .otherwise(None)
                            .alias("sum_temp_pl_XX")
                        ])
                    elif continuous > 0:
                        df = df.with_columns([
                            pl.when(cond)
                            .then(pl.col("treatment_XX_orig") - pl.col("d_sq_XX_orig"))
                            .otherwise(None)
                            .alias("sum_temp_pl_XX")
                        ])

                    df = df.with_columns([
                        pl.col("sum_temp_pl_XX").sum().over("group_XX").alias(f"sum_treat_until_{i}_pl_XX")
                    ]).drop("sum_temp_pl_XX")

                    df = df.with_columns([
                        pl.when(pl.col(f"dist_to_switch_pl_{i}_XX") == 1)
                        .then(
                            (pl.col("N_gt_XX") / N_placebo)
                            * (
                                pl.col("S_g_XX") * pl.col(f"sum_treat_until_{i}_pl_XX")
                                + (1 - pl.col("S_g_XX")) * (-pl.col(f"sum_treat_until_{i}_pl_XX"))
                            )
                        )
                        .otherwise(None)
                        .alias(f"delta_D_pl_{i}_cum_temp_XX")
                    ])

                    dict_vars_gen[f"delta_norm_pl_{i}_XX"] = float(
                        df[f"delta_D_pl_{i}_cum_temp_XX"].sum()
                    )
    

        # ---- Ntrendslin_pl (same Python logic) ----
        Ntrendslin_pl = 1
        if trends_lin:
            Ntrendslin_pl = min(
                int(dict_vars_gen[f"N{increase_XX}_placebo_{i}_XX"])
                for i in range(1, int(l_placebo_u_a_XX + 1))
            )

    

        if trends_lin and Ntrendslin_pl != 0:
            # print(f"Entramos a subsection placebo. This values {l_placebo_u_a_XX}")
            lp = int(l_placebo_u_a_XX)

            # Column names
            col_TL        = f"U_Gg_pl_{lp}_TL"
            col_var_TL    = f"U_Gg_pl_{lp}_var_TL"
            col_placebo   = f"U_Gg_placebo_{lp}_XX"
            col_pl_var_XX = f"U_Gg_pl_{lp}_var_XX"

            # Drop old TL columns if they exist
            for c in [col_TL, col_var_TL]:
                if c in df.columns:
                    df = df.drop(c)

            # Initialize TL columns to 0
            df = df.with_columns([
                pl.lit(0.0).alias(col_TL),
                pl.lit(0.0).alias(col_var_TL),
            ])

            # Accumulate over i = 1..lp
            for i in range(1, int(lp + 1)):
                df = df.with_columns([
                    (pl.col(col_TL) + pl.col(f"U_Gg_placebo_{i}_XX")).alias(col_TL),
                    (pl.col(col_var_TL) + pl.col(f"U_Gg_pl_{i}_var_XX")).alias(col_var_TL),
                ])

            # Copy back into the final placebo columns
            df = df.with_columns([
                pl.col(col_TL).alias(col_placebo),
                pl.col(col_var_TL).alias(col_pl_var_XX),
            ])

            
    if not trends_lin:
        # 1) Compute sum_N{increase_XX}_l_XX
        # print(f"this increase {increase_XX}")
        total_key = f"sum_N{increase_XX}_l_XX"
        dict_vars_gen[total_key] = sum(
            dict_vars_gen[f"N{increase_XX}_{i}_XX"]
            for i in range(1, int(l_u_a_XX) + 1)
        )

        # 2) Initialize needed DataFrame columns
        init_cols = ["U_Gg_XX", "U_Gg_num_XX", "U_Gg_den_XX", "U_Gg_num_var_XX", "U_Gg_var_XX"]
        df = df.with_columns([pl.lit(0).alias(col) for col in init_cols])

        for i in range(1, int(l_u_a_XX) + 1):
            # Column names
            N_increase      = dict_vars_gen[f"N{increase_XX}_{i}_XX"]
            sum_N_increase  = dict_vars_gen[f"sum_N{increase_XX}_l_XX"]
            delta_temp      = f"delta_D_{i}_temp_XX"
            delta           = f"delta_D_{i}_XX"
            delta_g         = f"delta_D_g_{i}_XX"
            dist_to_switch  = f"distance_to_switch_{i}_XX"

            # Only run if N_increase != 0
            if N_increase != 0:

                # 1. Compute weight
                w_i = N_increase / sum_N_increase
                dict_vars_gen[f"w_{i}_XX"] = w_i

                # 2. Compute delta_D_temp
                if continuous == 0:
                    df = df.with_columns(
                        pl.when(pl.col(dist_to_switch) == 1)
                        .then(
                            (pl.col("N_gt_XX") / pl.lit(N_increase)) *
                            (
                                (pl.col("treatment_XX") - pl.col("d_sq_XX")) * pl.col("S_g_XX") +
                                (1 - pl.col("S_g_XX")) * (pl.col("d_sq_XX") - pl.col("treatment_XX"))
                            )
                        )
                        .otherwise(pl.lit(None))
                        .alias(delta_temp)
                    )

                elif continuous > 0:
                    den_col = f"N{increase_XX}_{i}_XX"

                    df = df.with_columns(
                        pl.when(pl.col(dist_to_switch) == 1)
                        .then(
                            (pl.col("N_gt_XX") / pl.col(den_col)) *
                            (
                                (pl.col("treatment_XX_orig") - pl.col("d_sq_XX_orig")) * pl.col("S_g_XX") +
                                (1 - pl.col("S_g_XX")) * (pl.col("d_sq_XX_orig") - pl.col("treatment_XX_orig"))
                            )
                        )
                        .otherwise(pl.lit(None))
                        .alias(delta_temp)
                    )

                # Replace missing with 0
                df = df.with_columns(
                    pl.col(delta_temp).fill_null(0).alias(delta_temp)
                )

                # 3. Aggregate delta_D (sum over all obs → scalar replicated in column)
                total_delta = df.select(pl.col(delta_temp).sum()).item()
                df = df.with_columns(
                    pl.lit(total_delta).alias(delta)
                )

                # 4. Compute delta_D_g
                df = df.with_columns(
                    (pl.col(delta_temp) * (pl.lit(N_increase) / pl.col("N_gt_XX"))).alias(delta_g)
                )

                # 5. Drop temp
                df = df.drop(delta_temp)

                # 6. Update U_Gg_* numerators and denominators (cumulative over i)
                w_i = dict_vars_gen[f"w_{i}_XX"]

                df = df.with_columns([
                    # U_Gg_num_XX += w_i * U_Gg{i}_XX
                    (pl.col("U_Gg_num_XX") + w_i * pl.col(f"U_Gg{i}_XX")).over("group_XX").alias("U_Gg_num_XX"),

                    # U_Gg_num_var_XX += w_i * U_Gg{i}_var_XX
                    (pl.col("U_Gg_num_var_XX") + w_i * pl.col(f"U_Gg{i}_var_XX")).over("group_XX").alias("U_Gg_num_var_XX"),

                    # U_Gg_den_XX += w_i * delta
                    (pl.col("U_Gg_den_XX") + w_i * pl.col(delta)).over("group_XX").alias("U_Gg_den_XX")
                ])

        df = df.with_columns(
        U_Gg_XX      = pl.col("U_Gg_num_XX") / pl.col("U_Gg_den_XX"),
        U_Gg_var_XX  = pl.col("U_Gg_num_var_XX") / pl.col("U_Gg_den_XX"),
    )

    # =========================================================
    # ===  Propagate constants and normalized deltas        ===
    # =========================================================
    for e in list(const.keys()):
        if e in locals():
            const[e] = locals()[e]
        elif e in dict_vars_gen:
            const[e] = dict_vars_gen[e]
        else:
            const[e] = 0

    sum_key = f"sum_N{increase_XX}_l_XX"
    if sum_key in locals():
        const[sum_key] = locals()[sum_key]
    elif sum_key in dict_vars_gen:
        const[sum_key] = dict_vars_gen[sum_key]

    # --- Add normalized deltas ---
    if normalized:
        for i in range(1, int(l_u_a_XX + 1)):
            key = f"delta_norm_{i}_XX"
            if key in locals():
                const[key] = locals()[key]
            elif key in dict_vars_gen:
                const[key] = dict_vars_gen[key]

    # --- Add placebo deltas ---
    if placebo != 0 and l_placebo_u_a_XX >= 1 and normalized:
        for i in range(1, int(l_placebo_u_a_XX + 1)):
            key = f"delta_norm_pl_{i}_XX"
            if key in locals():
                const[key] = locals()[key]
            elif key in dict_vars_gen:
                const[key] = dict_vars_gen[key]

    # --- Final output ---
    data = {'df': df, 'const': dict_vars_gen}
    return data









