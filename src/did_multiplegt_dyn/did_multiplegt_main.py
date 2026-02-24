from __future__ import annotations

import polars as pl
import numpy as np
import math
import warnings
from scipy.stats import norm, chi2, t as student_t
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.stats.outliers_influence import summary_table
from statsmodels.stats.diagnostic import linear_harvey_collier
from statsmodels.stats.contrast import ContrastResults
from statsmodels.stats.sandwich_covariance import cov_hc0, cov_hc1
import pandas as pd
from .did_multiplegt_dyn_core import did_multiplegt_dyn_core_pl
from ._utils import *
from ._date_first_switch import run_date_first_switch
from ._normalized_weights import print_normalized_weights
# Equivalent of R's MASS::ginv (generalized inverse)

def did_multiplegt_main(
        df: pl.DataFrame,
        outcome,
        group,
        time,
        treatment,
        cluster=None,
        effects=1,
        placebo=1,
        normalized=False,
        effects_equal=False,
        controls=None,
        trends_nonparam=None,
        trends_lin=False,
        continuous=0,
        weight=None,
        predict_het=None,
        same_switchers=False,
        same_switchers_pl=False,
        switchers="",
        only_never_switchers=False,
        ci_level=95,
        save_results=None,
        less_conservative_se=False,
        more_granular_demeaning=False,
        dont_drop_larger_lower=False,
        drop_if_d_miss_before_first_switch=False,
        bootstrap=None,
        by_path=None,
        design=None,
        by=None,
        normalized_weights=False,
        predict_het_hc2bm=False,
        date_first_switch=None
):

    import polars as pl
    import numpy as np
    import warnings

    # === Initialize all variables used later ===
    dict_glob = {}
    gr_id = None
    weight_XX = None
    F_g_XX = None
    F_g_trunc_XX = None
    N_gt_XX = None
    T_g_XX = None
    U_Gg_var_global_XX = None
    Yg_Fg_min1_XX = None
    Yg_Fg_min2_XX = None
    avg_diff_temp_XX = None
    avg_post_switch_treat_XX = None
    avg_post_switch_treat_XX_temp = None
    clust_U_Gg_var_global_XX = None
    cluster_XX = None
    cluster_var_g_XX = None
    controls_time_XX = None
    count_time_post_switch_XX = None
    count_time_post_switch_XX_temp = None
    counter = None
    counter_temp = None
    d_F_g_XX = None
    d_F_g_temp_XX = None
    d_fg_XX = None
    d_sq_XX = None
    d_sq_int_XX = None
    d_sq_temp_XX = None
    diff_y_XX = None
    ever_change_d_XX = None
    fd_X_all_non_missing_XX = None
    first_obs_by_clust_XX = None
    first_obs_by_gp_XX = None
    group_XX = None
    last_obs_D_bef_switch_XX = None
    last_obs_D_bef_switch_t_XX = None
    max_time_d_nonmiss_XX = None
    mean_D = None
    mean_Y = None
    min_time_d_miss_aft_ynm_XX = None
    min_time_d_nonmiss_XX = None
    min_time_y_nonmiss_XX = None
    never_change_d_XX = None
    sd_het = None
    sum_weights_control_XX = None
    temp_F_g_XX = None
    time_XX = None
    time_d_miss_XX = None
    time_d_nonmiss_XX = None
    time_y_nonmiss_XX = None
    treatment_XX_v1 = None
    var_F_g_XX = None

    # Subset columns but do NOT rename originals
    original_names = [outcome, group, time, treatment]

    # Continuous option: checking polynomial order
    if continuous > 0:
        degree_pol = continuous

    # Bootstrap validation warnings
    if bootstrap is not None and continuous == 0:
        warnings.warn(
            "You specified the bootstrap option without the continuous option. "
            "We strongly recommend computing bootstrapped standard errors "
            "only when using the continuous option, as analytical SEs can be "
            "liberal in that case.",
            UserWarning
        )
    if bootstrap is None and continuous > 0:
        warnings.warn(
            "You specified the continuous option without the bootstrap option. "
            "We recommend computing bootstrapped standard errors when using "
            "the continuous option as analytical SEs can be liberal in that case.",
            UserWarning
        )

    # =========================================================================
    # BY OPTION: Run estimation separately for each unique value of by variable
    # =========================================================================
    if by is not None:
        # Get unique values of the by variable
        by_values = df.select(by).unique().sort(by).to_series().to_list()

        # Store results for each subgroup
        all_by_results = []

        for by_val in by_values:
            # Subset data for this value
            df_subset = df.filter(pl.col(by) == by_val)

            # Recursively call without the by option
            subset_result = did_multiplegt_main(
                df=df_subset,
                outcome=outcome,
                group=group,
                time=time,
                treatment=treatment,
                cluster=cluster,
                effects=effects,
                placebo=placebo,
                normalized=normalized,
                effects_equal=effects_equal,
                controls=controls,
                trends_nonparam=trends_nonparam,
                trends_lin=trends_lin,
                continuous=continuous,
                weight=weight,
                predict_het=predict_het,
                same_switchers=same_switchers,
                same_switchers_pl=same_switchers_pl,
                switchers=switchers,
                only_never_switchers=only_never_switchers,
                ci_level=ci_level,
                save_results=None,  # Don't save individual results
                less_conservative_se=less_conservative_se,
                dont_drop_larger_lower=dont_drop_larger_lower,
                drop_if_d_miss_before_first_switch=drop_if_d_miss_before_first_switch,
                bootstrap=bootstrap,
                by_path=None,  # Don't nest by_path in by
                design=None,  # Don't show design for each subset
                by=None  # No nested by
            )

            # Add by_results metadata to indicate which subgroup this is
            if 'did_multiplegt_dyn' in subset_result:
                subset_result['did_multiplegt_dyn']['by_results'] = {
                    'by_var': by,
                    'by_value': by_val
                }

            all_by_results.append({
                'by_var': by,
                'by_value': by_val,
                'result': subset_result
            })

        # Return combined results
        # Use the first subgroup's result as base but include all by_results
        combined_result = all_by_results[0]['result'] if all_by_results else {'did_multiplegt_dyn': {}}
        combined_result['did_multiplegt_dyn']['all_by_results'] = all_by_results

        return combined_result

    # =========================================================================
    # BY_PATH OPTION: Run estimation separately for each treatment path
    # =========================================================================
    if by_path is not None:
        from ._by_path import (
            validate_by_path_input,
            rank_paths_by_frequency,
            get_control_subset_for_path,
            format_path_for_display
        )

        # Validate input
        n_paths_requested = validate_by_path_input(by_path)

        # We need to prepare the data first to identify paths
        # Subset columns but do NOT rename originals yet
        temp_original_names = [outcome, group, time, treatment]
        if trends_nonparam:
            temp_original_names += trends_nonparam
        if weight:
            temp_original_names.append(weight)
        if controls:
            temp_original_names += controls
        if (cluster) and (cluster != group):
            temp_original_names.append(cluster)
        temp_original_names = list(dict.fromkeys(temp_original_names))
        df_temp = df.select(temp_original_names)

        # Standardize names for path identification
        df_temp = df_temp.rename({
            outcome: "outcome_XX",
            group: "group_XX",
            time: "time_XX",
            treatment: "treatment_XX",
        })

        # Sort data
        df_temp = df_temp.sort(["group_XX", "time_XX"])

        # Create T_g (last time per group)
        df_temp = df_temp.with_columns([
            pl.col("time_XX").max().over("group_XX").alias("T_g_XX")
        ])

        # Detect treatment changes to find F_g
        df_temp = df_temp.with_columns([
            pl.col("treatment_XX").shift(1).over("group_XX").alias("treatment_lag")
        ])
        df_temp = df_temp.with_columns([
            ((pl.col("treatment_XX") != pl.col("treatment_lag")) &
             pl.col("treatment_lag").is_not_null()).alias("treatment_changed")
        ])
        df_temp = df_temp.with_columns([
            pl.when(pl.col("treatment_changed"))
            .then(pl.col("time_XX"))
            .otherwise(None)
            .alias("switch_time")
        ])
        df_temp = df_temp.with_columns([
            pl.col("switch_time").min().over("group_XX").alias("F_g_XX")
        ])

        # Fill F_g with T_g+1 for never-switchers
        df_temp = df_temp.with_columns([
            pl.when(pl.col("F_g_XX").is_null())
            .then(pl.col("T_g_XX") + 1)
            .otherwise(pl.col("F_g_XX"))
            .alias("F_g_XX")
        ])

        # Create first_obs_by_gp_XX
        df_temp = df_temp.with_columns([
            (pl.col("time_XX").cum_count().over("group_XX") == 1).alias("first_obs_by_gp_XX")
        ])

        # Create d_sq_XX (baseline treatment before switch)
        df_temp = df_temp.with_columns([
            pl.when(pl.col("time_XX") == pl.col("F_g_XX") - 1)
            .then(pl.col("treatment_XX"))
            .otherwise(None)
            .alias("d_sq_temp")
        ])
        df_temp = df_temp.with_columns([
            pl.col("d_sq_temp").max().over("group_XX").alias("d_sq_XX")
        ])

        # Create dummy columns for treatment at each relative time period
        # k=0: baseline (F_g - 1), k=1: F_g, k=2: F_g+1, ..., k=l: F_g-1+l
        dummy_cols = [f"dummy_treat_{k}" for k in range(effects + 1)]

        for k in range(effects + 1):
            target_time_col = f"target_time_{k}"
            # Target time for this k: F_g - 1 + k
            df_temp = df_temp.with_columns([
                (pl.col("F_g_XX") - 1 + k).alias(target_time_col)
            ])
            # Extract treatment at target time
            df_temp = df_temp.with_columns([
                pl.when(pl.col("time_XX") == pl.col(target_time_col))
                .then(pl.col("treatment_XX"))
                .otherwise(None)
                .alias(f"treat_at_{k}_temp")
            ])
            # Propagate to all observations in the group
            df_temp = df_temp.with_columns([
                pl.col(f"treat_at_{k}_temp").drop_nulls().first().over("group_XX").alias(dummy_cols[k])
            ])
            df_temp = df_temp.drop([target_time_col, f"treat_at_{k}_temp"])

        # Create path string from treatment sequence (integers for cleaner display)
        # NA values indicate periods where data is not available
        df_temp = df_temp.with_columns([
            pl.concat_str(
                [pl.col(c).cast(pl.Int64).cast(pl.Utf8).fill_null("NA") for c in dummy_cols],
                separator=", "
            ).alias("path_str_XX")
        ])

        # Only switchers have valid paths (F_g <= T_g)
        # Include paths even if some periods have NA (incomplete paths)
        df_temp = df_temp.with_columns([
            pl.when(pl.col("F_g_XX") <= pl.col("T_g_XX"))
            .then(pl.col("path_str_XX"))
            .otherwise(None)
            .alias("path_str_XX")
        ])

        # Calculate how many complete effect periods each group has
        # This is T_g - F_g + 1 (number of periods from F_g to T_g inclusive)
        df_temp = df_temp.with_columns([
            (pl.col("T_g_XX") - pl.col("F_g_XX") + 1).alias("max_effects_available_XX")
        ])

        # Create numeric path identifier
        switcher_paths = df_temp.filter(
            pl.col("path_str_XX").is_not_null()
        ).select("path_str_XX").unique()

        if switcher_paths.height == 0:
            warnings.warn("No treatment paths found in data", UserWarning)
            # Fall through to regular estimation
        else:
            path_mapping = switcher_paths.with_row_index("path_id_temp")
            df_temp = df_temp.join(path_mapping, on="path_str_XX", how="left")
            df_temp = df_temp.with_columns([
                (pl.col("path_id_temp") + 1).alias("different_paths_XX")
            ])
            df_temp = df_temp.drop("path_id_temp")

            # Rank paths by frequency
            df_temp = rank_paths_by_frequency(df_temp)

            # Count total paths
            n_total_paths = df_temp.select(
                pl.col("different_paths_XX").drop_nulls().n_unique()
            ).item()

            # Determine number of paths to analyze
            if n_paths_requested == -1:  # "all"
                n_paths = int(n_total_paths)
            else:
                n_paths = min(n_paths_requested, int(n_total_paths))
                if n_paths_requested > n_total_paths:
                    warnings.warn(
                        f"Requested {n_paths_requested} paths but only {n_total_paths} exist. "
                        f"Analyzing all {n_total_paths} paths.",
                        UserWarning
                    )

            # Store results for each path
            all_path_results = []

            for k in range(1, n_paths + 1):
                # Get path info - extract treatment sequence from path_str
                path_row = df_temp.filter(
                    pl.col("different_paths_XX") == k
                ).select(["path_str_XX", "max_effects_available_XX"]).unique()

                if path_row.height == 0:
                    continue

                path_str = path_row["path_str_XX"][0]
                # Parse the path string back to a list
                treatment_sequence = [int(x) if x != "NA" else None for x in path_str.split(", ")]

                # Count groups in this path
                n_groups = df_temp.filter(
                    (pl.col("different_paths_XX") == k) & pl.col("first_obs_by_gp_XX")
                ).height

                # Determine maximum effects that can be estimated for this path
                # Based on the minimum data span among groups in this path
                path_max_effects = df_temp.filter(
                    (pl.col("different_paths_XX") == k) & pl.col("first_obs_by_gp_XX")
                ).select("max_effects_available_XX").min().item()

                # Adjust effects for this path
                path_effects = min(effects, int(path_max_effects)) if path_max_effects else effects

                # Calculate max placebos: need data before F_g-1
                # For placebos, we need observations at F_g-1-p for p=1,2,...
                # Max placebos = max over groups of (F_g - 1 - first_time_in_group)
                # Use max() because Stata estimates placebos for groups that have data,
                # even if some groups don't (those just won't contribute to the estimation)
                path_groups_data = df_temp.filter(
                    pl.col("different_paths_XX") == k
                ).group_by("group_XX").agg([
                    pl.col("F_g_XX").first().alias("F_g"),
                    pl.col("time_XX").min().alias("min_time")
                ])
                if path_groups_data.height > 0:
                    path_groups_data = path_groups_data.with_columns([
                        (pl.col("F_g") - 1 - pl.col("min_time")).alias("placebos_available")
                    ])
                    path_max_placebos = int(path_groups_data["placebos_available"].max())
                    path_max_placebos = max(0, path_max_placebos)  # Ensure non-negative
                else:
                    path_max_placebos = 0
                path_placebos = min(placebo, path_max_placebos)

                # Show warnings if effects or placebos were reduced
                if path_effects < effects:
                    print(f"Path ({', '.join(str(x) if x is not None else 'NA' for x in treatment_sequence)}): "
                          f"The number of effects which can be estimated is at most {path_effects}. "
                          f"The command will therefore try to estimate {path_effects} effect(s).")
                if path_placebos < placebo:
                    print(f"Path ({', '.join(str(x) if x is not None else 'NA' for x in treatment_sequence)}): "
                          f"The number of placebos which can be estimated is at most {path_placebos}. "
                          f"The command will therefore try to estimate {path_placebos} placebo(s).")

                # Get filtered data for this path
                df_path = get_control_subset_for_path(df_temp, k, path_effects)

                # Map back to original column names for estimation
                df_path = df_path.rename({
                    "outcome_XX": outcome,
                    "group_XX": group,
                    "time_XX": time,
                    "treatment_XX": treatment,
                })

                # Run estimation for this path with adjusted effects/placebos
                try:
                    path_result = did_multiplegt_main(
                        df=df_path,
                        outcome=outcome,
                        group=group,
                        time=time,
                        treatment=treatment,
                        cluster=cluster,
                        effects=path_effects,
                        placebo=path_placebos,
                        normalized=normalized,
                        effects_equal=effects_equal,
                        controls=controls,
                        trends_nonparam=trends_nonparam,
                        trends_lin=trends_lin,
                        continuous=continuous,
                        weight=weight,
                        predict_het=predict_het,
                        same_switchers=True,  # Required for by_path
                        same_switchers_pl=same_switchers_pl,
                        switchers=switchers,
                        only_never_switchers=only_never_switchers,
                        ci_level=ci_level,
                        save_results=None,
                        less_conservative_se=less_conservative_se,
                        dont_drop_larger_lower=dont_drop_larger_lower,
                        drop_if_d_miss_before_first_switch=drop_if_d_miss_before_first_switch,
                        bootstrap=bootstrap,
                        by_path=None,  # No nested by_path
                        design=None,
                        by=None
                    )

                    all_path_results.append({
                        'path_id': k,
                        'treatment_sequence': treatment_sequence,
                        'n_groups': n_groups,
                        'result': path_result
                    })
                except Exception as e:
                    warnings.warn(f"Estimation failed for path {k}: {str(e)}", UserWarning)
                    all_path_results.append({
                        'path_id': k,
                        'treatment_sequence': treatment_sequence,
                        'n_groups': n_groups,
                        'result': {'did_multiplegt_dyn': {'Effects': None, 'ATE': None, 'Placebos': None}},
                        'error': str(e)
                    })

            # Return combined results
            combined_result = all_path_results[0]['result'] if all_path_results else {'did_multiplegt_dyn': {}}
            combined_result['did_multiplegt_dyn']['all_path_results'] = all_path_results

            return combined_result

    if trends_nonparam:
        original_names += trends_nonparam
    if weight:
        original_names.append(weight)
    if controls:
        original_names += controls
    if (cluster) and (cluster != group):
        original_names.append(cluster)
    if predict_het:
        # Handle both formats: ['var', 'all'] or [['var1', 'var2'], 'all']
        het_vars = predict_het[0]
        if isinstance(het_vars, str):
            original_names.append(het_vars)
        else:
            original_names += list(het_vars)
    if by:
        original_names.append(by)

    # Polars doesn't need .copy(), and it doesn't like duplicate column names,
    # so we deduplicate while preserving order:
    original_names = list(dict.fromkeys(original_names))

    # Subset columns
    df = df.select(original_names)

    # Standardize names
    df = df.rename({
        outcome:   "outcome",
        group:     "group",
        time:      "time",
        treatment: "treatment",
    })

    # 2. Data preparation steps
    # -------------------------


    # Patch the cluster variable
    if cluster is not None:
        if cluster == group:
            cluster = None
        else:
            # df['cluster_XX'] = df[cluster].copy()
            df = df.with_columns(
                pl.col(cluster).alias("cluster_XX")
            )
            cluster = "cluster_XX"


    # Standardize names
    group     = "group"
    outcome   = "outcome"
    time      = "time"
    treatment = "treatment"

    # In pandas: df.rename(columns={...})
    # In polars:
    rename_map = {
        outcome:   "outcome",
        group:     "group",
        time:      "time",
        treatment: "treatment",
    }

    # mimic pandas behavior: if cluster is None, nothing happens;
    # if cluster is a column name, rename it to 'cluster_XX';
    # if cluster == 'cluster_XX', this is just identity
    if cluster is not None:
        rename_map[cluster] = "cluster_XX"

    df = df.rename(rename_map)


    # we drop observations which are missing group, time, or controls
    group     = "group"
    outcome   = "outcome"
    time      = "time"
    treatment = "treatment"


    # Drop rows with missing group or time
    # df = df.dropna(subset=[group, time]).copy()
    df = df.drop_nulls(subset=[group, time])

    # Drop rows with missing controls
    if controls:
        for var in controls:
            # df = df.dropna(subset=[var]).copy()
            df = df.drop_nulls(subset=[var])

    # Drop rows with missing cluster
    if cluster is not None:
        # df = df.dropna(subset=['cluster_XX']).copy()
        df = df.drop_nulls(subset=["cluster_XX"])


    # Drop groups with always-missing treatment or outcome
    # pandas:
    # df['mean_D'] = df.groupby(group)[treatment].transform(lambda x: x.mean(skipna=True))
    # df['mean_Y'] = df.groupby(group)[outcome].transform(lambda x: x.mean(skipna=True))

    df = df.with_columns([
        pl.col(treatment).mean().over(group).alias("mean_D"),
        pl.col(outcome).mean().over(group).alias("mean_Y"),
    ])

    # df = df[df['mean_D'].notna() & df['mean_Y'].notna()]
    df = df.filter(
        pl.col("mean_D").is_not_null() & pl.col("mean_Y").is_not_null()
    )

    # df = df.drop(columns=['mean_D', 'mean_Y'])
    df = df.drop(["mean_D", "mean_Y"])

   

    # -------------------------------------------------
    # Predict_het option for heterogeneous treatment effects
    # -------------------------------------------------
    predict_het_good = []
    if predict_het is not None:
        if not (isinstance(predict_het, list) and len(predict_het) == 2):
            raise ValueError(
                "Syntax error in predict_het option: list with 2 elements required. "
                "Set the second element to -1 to include all the effects."
            )
        if normalized:
            warnings.warn(
                "The options normalized and predict_het cannot be specified together; "
                "predict_het will be ignored."
            )
        else:
            pred_het, het_effects = predict_het
            # Handle both formats: 'var' or ['var1', 'var2']
            if isinstance(pred_het, str):
                pred_het = [pred_het]
            for v in pred_het:
                # df['sd_het'] = df.groupby(group)[v].transform(lambda x: x.std(skipna=True))
                df = df.with_columns(
                    pl.col(v).std().over(group).alias("sd_het")
                )

                m = df["sd_het"].mean()
                if m is None:
                    m = np.nan

                if float(m) == 0:
                    predict_het_good.append(v)
                else:
                    warnings.warn(
                        f"The variable {v} specified in predict_het is time-varying; it will be ignored."
                    )

                # df = df.drop(columns=['sd_het'])
                df = df.drop("sd_het")


    # -------------------------------------------------
    # — Collapse and weight —
    # -------------------------------------------------

    # 1. Create the weight variable
    if weight is None:
        # df['weight_XX'] = 1
        df = df.with_columns(
            pl.lit(1).alias("weight_XX")
        )
    else:
        # df['weight_XX'] = df[weight]
        df = df.with_columns(
            pl.col(weight).alias("weight_XX")
        )

    # df['weight_XX'] = df['weight_XX'].fillna(0)
    df = df.with_columns(
        pl.col("weight_XX").fill_null(0).alias("weight_XX")
    )

    # 2. Check whether data are already at the (group, time) level
    # df['counter_temp'] = 1
    df = df.with_columns(
        pl.lit(1).alias("counter_temp")
    )

    # df['counter'] = df.groupby([group, time])['counter_temp'].transform('sum')
    df = df.with_columns(
        pl.col("counter_temp").sum().over([group, time]).alias("counter")
    )

    # aggregated_data = df['counter'].max() == 1
    aggregated_data = df["counter"].max() == 1

    # df = df.drop(columns=['counter', 'counter_temp'])
    df = df.drop(["counter", "counter_temp"])


    # -------------------------------------------------
    # Re-aggregate if not already at (group, time) level
    # -------------------------------------------------
    if not aggregated_data:
        # zero out weight where treatment is missing
        # df.loc[df['treatment'].isna(), 'weight_XX'] = 0
        df = df.with_columns(
            pl.when(pl.col("treatment").is_null())
            .then(0)
            .otherwise(pl.col("weight_XX"))
            .alias("weight_XX")
        )

        # if no clustering variable specified, create a dummy
        if cluster is None:
            # df['cluster_XX'] = 1
            df = df.with_columns(
                pl.lit(1).alias("cluster_XX")
            )

        # build list of columns to re‐aggregate with weighted means
        to_wmean = ['treatment', 'outcome']
        to_wmean += trends_nonparam or []
        if weight is not None:
            to_wmean.append(weight)
        to_wmean += controls or []
        to_wmean += predict_het_good or []
        to_wmean.append('cluster_XX')

        group_keys = ["group", "time"]

        # df1: weighted means (using weight_XX)
        agg_exprs = []
        for col in to_wmean:
            agg_exprs.append(
                pl.when(pl.col("weight_XX").sum() > 0)
                .then(
                    (pl.col(col) * pl.col("weight_XX")).sum()
                    / pl.col("weight_XX").sum()
                )
                .otherwise(pl.lit(None))
                .alias(col)
            )

        df1 = df.groupby(group_keys).agg(agg_exprs)

        # df2: sum of weights
        # df2 = grp['weight_XX'].sum().reset_index(name='weight_XX')
        df2 = df.groupby(group_keys).agg(
            pl.col("weight_XX").sum().alias("weight_XX")
        )

        # merge back
        # df = pd.merge(df1, df2, on=['group', 'time'] )
        df = df1.join(df2, on=group_keys, how="inner")

        # drop dummy cluster if we added it
        if cluster is None:
            # df = df.drop(columns='cluster_XX')
            df = df.drop("cluster_XX")

    # — Generate factorized versions of Y, G, T and D —
    df = df.with_columns([
        pl.col(outcome).alias("outcome_XX"),
    ])

    # sort by time (originally: df = df.sort_values(time))
    df = df.sort(by=time)
    # Create factor mapping for group
    group_mapping = (
        df[group]
        .unique()
        .sort()
        .to_frame()
        .with_row_index("group_XX", offset=1)  # starts at 1
        .select([group, "group_XX"])
    )

    # Create factor mapping for time
    time_mapping = (
        df[time]
        .unique()
        .sort()
        .to_frame()
        .with_row_index("time_XX", offset=1)
        .select([time, "time_XX"])
    )

    # Join mappings and add columns
    df = (
        df
        .join(group_mapping, on=group, how="left")
        .join(time_mapping, on=time, how="left")
        .with_columns([
            pl.col(outcome).alias("outcome_XX"),
            pl.col(treatment).alias("treatment_XX")
        ])
    )

    import polars as pl
    import numpy as np

    # — Ensure numeric (float) —
    df = df.with_columns([
        pl.col("group_XX").cast(pl.Float64),
        pl.col("time_XX").cast(pl.Float64)
    ])

    # — Variables for imbalanced panels —
    # First/last time where D (treatment) is not missing
    df = df.with_columns(
        pl.when(pl.col("treatment_XX").is_not_null())
        .then(pl.col("time_XX"))
        .otherwise(pl.lit(None))
        .alias("time_d_nonmiss_XX")
    )

    # First time where Y (outcome) is not missing
    df = df.with_columns(
        pl.when(pl.col("outcome_XX").is_not_null())
        .then(pl.col("time_XX"))
        .otherwise(pl.lit(None))
        .alias("time_y_nonmiss_XX")
    )

    # Group by group_XX and compute min/max per group
    df = df.with_columns([
        pl.col("time_d_nonmiss_XX").min().over("group_XX").alias("min_time_d_nonmiss_XX"),
        pl.col("time_d_nonmiss_XX").max().over("group_XX").alias("max_time_d_nonmiss_XX"),
        pl.col("time_y_nonmiss_XX").min().over("group_XX").alias("min_time_y_nonmiss_XX"),
    ])

    # — First time D is missing *after* Y is seen —
    df = df.with_columns(
        pl.when(
            pl.col("treatment_XX").is_null() &
            (pl.col("time_XX") >= pl.col("min_time_y_nonmiss_XX"))
        )
        .then(pl.col("time_XX"))
        .otherwise(pl.lit(None))
        .alias("time_d_miss_XX")
    )

    # Min of time_d_miss_XX per group
    df = df.with_columns(
        pl.col("time_d_miss_XX").min().over("group_XX").alias("min_time_d_miss_aft_ynm_XX")
    )

    # Drop intermediate columns
    df = df.drop([
        "time_d_nonmiss_XX", "time_y_nonmiss_XX", "time_d_miss_XX"
    ])

    # — Baseline treatment D_{g,1}: treatment at earliest time where D is observed —
    df = df.with_columns(
        pl.when(pl.col("time_XX") == pl.col("min_time_d_nonmiss_XX"))
        .then(pl.col("treatment_XX"))
        .otherwise(pl.lit(None))
        .alias("d_sq_temp_XX")
    )

    # Mean of d_sq_temp_XX per group (should be single value per group)
    df = df.with_columns(
        pl.col("d_sq_temp_XX").mean().over("group_XX").alias("d_sq_XX")
    )

    df = df.drop("d_sq_temp_XX")

    # — Enforce “Design Restriction 2”: no strict increase AND decrease from baseline —
    df = df.with_columns(
        (pl.col("treatment_XX") - pl.col("d_sq_XX")).alias("diff_from_sq_XX")
    )

    # Sort by group_XX and time_XX (like Stata)
    df = df.sort(["group_XX", "time_XX"])

    # Compute ever_strict_increase_XX: cumulative flag if treatment > baseline and not null
    df = df.with_columns(
        pl.when((pl.col("diff_from_sq_XX") > 0) & pl.col("treatment_XX").is_not_null())
        .then(1)
        .otherwise(0)
        .alias("ever_strict_increase_XX")
    ).with_columns(
        pl.col("ever_strict_increase_XX")
        .cum_sum()
        .clip(upper_bound=1)
        .over("group_XX")
        .alias("ever_strict_increase_XX")
    )

    # Compute ever_strict_decrease_XX
    df = df.with_columns(
        pl.when((pl.col("diff_from_sq_XX") < 0) & pl.col("treatment_XX").is_not_null())
        .then(1)
        .otherwise(0)
        .alias("ever_strict_decrease_XX")
    ).with_columns(
        pl.col("ever_strict_decrease_XX")
        .cum_sum()
        .clip(upper_bound=1)
        .over("group_XX")
        .alias("ever_strict_decrease_XX")
    )


    # Drop rows where both increase and decrease flags are 1
    if not dont_drop_larger_lower:  # assuming this variable exists
        df = df.filter(
            ~((pl.col("ever_strict_increase_XX") == 1) & (pl.col("ever_strict_decrease_XX") == 1))
        )

    # Drop helper columns
    df = df.drop(["ever_strict_increase_XX", "ever_strict_decrease_XX"])


    import polars as pl
    import numpy as np

    # — 1. Ever changed treatment —
    df = df.with_columns(
        pl.when(
            (pl.col("diff_from_sq_XX").abs() > 0) & pl.col("treatment_XX").is_not_null()
        )
        .then(True)
        .otherwise(False)
        .alias("ever_change_d_XX")
    )

    # Sort by group and time
    df = df.sort(["group_XX", "time_XX"])

    # Carry forward: cummax within group (cast to int for compatibility)
    df = df.with_columns(
        pl.col("ever_change_d_XX")
        .cast(pl.Int8)
        .cum_max()
        .over("group_XX")
        .cast(pl.Boolean)
        .alias("ever_change_d_XX")
    )

    # — 2. First treatment-change date (F_g) —
    # Temp: mark first True in ever_change_d_XX per group
    df = df.with_columns(
        pl.when(
            pl.col("ever_change_d_XX") &
            (~pl.col("ever_change_d_XX").shift(1).over("group_XX").fill_null(False))
        )
        .then(pl.col("time_XX"))
        .otherwise(0)
        .alias("temp_F_g_XX")
    )

    # Max of temp_F_g_XX per group → F_g_XX
    df = df.with_columns(
        pl.col("temp_F_g_XX").max().over("group_XX").alias("F_g_XX")
    )

    # Drop temp column
    # df = df.drop("temp_F_g_XX")

    # — 3. Continuous option: polynomials of D_{g,1} —
    if continuous > 0:
        for p in range(1, degree_pol + 1):
            p = int(p)
            df = df.with_columns(
                (pl.col("d_sq_XX") ** p).alias(f"d_sq_{p}_XX")
            )
        df = df.with_columns([
            pl.col("d_sq_XX").alias("d_sq_XX_orig"),
            pl.lit(0).alias("d_sq_XX")  # set d_sq_XX to 0 after creating polys
        ])

    # — 4. Integer levels of d_sq_XX (factorize) —
    # Group by d_sq_XX, assign dense rank starting at 1
    # — 4. Integer levels of d_sq_XX (factorize) —
    mapping_df = (
        df.select("d_sq_XX")
        .filter(pl.col("d_sq_XX").is_not_null())
        .unique()
        .sort("d_sq_XX")
        .with_row_index("d_sq_int_XX", offset=1)
        .select(["d_sq_XX", "d_sq_int_XX"])
    )

    df = df.join(mapping_df, on="d_sq_XX", how="left")

    # — 5. Drop baseline treatments with no variation in F_g —
    group_cols = ["d_sq_XX"] + (trends_nonparam or [])
    df = df.with_columns(
        pl.col("F_g_XX").std().over(group_cols).round(3).alias("var_F_g_XX")
    )

    # Keep only groups with variation > 0
    df = df.filter(pl.col("var_F_g_XX") > 0).drop("var_F_g_XX")

    if df.is_empty():
        raise ValueError(
            "No treatment effect can be estimated. Design Restriction 1 is not satisfied."
        )

    G_XX = df["group_XX"].n_unique()

    # — 6. Restrict to cells with at least one “never-changer” —
    df = df.with_columns(
        (1 - pl.col("ever_change_d_XX").cast(pl.Int32)).alias("never_change_d_XX")
    )

    ctrl_group = ["time_XX", "d_sq_XX"] + (trends_nonparam or [])
    df = df.with_columns(
        pl.col("never_change_d_XX").max().over(ctrl_group).alias("controls_time_XX")
    )

    df = df.filter(pl.col("controls_time_XX") > 0)

    # — 7. Adjust F_g for never-changers: set to T_max + 1 —
    t_min_XX = df["time_XX"].min()
    T_max_XX = df["time_XX"].max()

    df = df.with_columns(
        pl.when(pl.col("F_g_XX") == 0)
        .then(T_max_XX + 1)
        .otherwise(pl.col("F_g_XX"))
        .alias("F_g_XX")
    )

    # — 8. Missing-treatment: conservative drop (set Y to NaN) —
    if drop_if_d_miss_before_first_switch:
        mask = (
            (pl.col("min_time_d_miss_aft_ynm_XX") < pl.col("F_g_XX")) &
            (pl.col("time_XX") >= pl.col("min_time_d_miss_aft_ynm_XX"))
        )
        df = df.with_columns(
            pl.when(mask)
            .then(pl.lit(None))
            .otherwise(pl.col("outcome_XX"))
            .alias("outcome_XX")
        )

    # — 9. Last observed D before switch —
    df = df.with_columns(
        pl.when(
            (pl.col("time_XX") < pl.col("F_g_XX")) & pl.col("treatment_XX").is_not_null()
        )
        .then(pl.col("time_XX"))
        .otherwise(pl.lit(None))
        .alias("last_obs_D_bef_switch_t_XX")
    )

    df = df.with_columns(
        pl.col("last_obs_D_bef_switch_t_XX")
        .max()
        .over("group_XX")
        .alias("last_obs_D_bef_switch_XX")
    )

    import polars as pl

    # — Drop outcomes before first non-missing D —
    df = df.with_columns(
        pl.when(pl.col("time_XX") < pl.col("min_time_d_nonmiss_XX"))
        .then(pl.lit(None))
        .otherwise(pl.col("outcome_XX"))
        .alias("outcome_XX")
    )

    # — Fill missing D before switch with baseline (d_sq_XX) —
    mask = (
        (pl.col("F_g_XX") < T_max_XX + 1) &
        pl.col("treatment_XX").is_null() &
        (pl.col("time_XX") < pl.col("last_obs_D_bef_switch_XX")) &
        (pl.col("time_XX") > pl.col("min_time_d_nonmiss_XX"))
    )

    df = df.with_columns(
        pl.when(mask)
        .then(pl.col("d_sq_XX"))
        .otherwise(pl.col("treatment_XX"))
        .alias("treatment_XX")
    )

    # — Drop outcomes in ambiguous window and truncate controls —
    ambiguous_mask = (
        (pl.col("F_g_XX") < T_max_XX + 1) &
        (pl.col("time_XX") > pl.col("last_obs_D_bef_switch_XX")) &
        (pl.col("last_obs_D_bef_switch_XX") < (pl.col("F_g_XX") - 1))
    )

    df = df.with_columns(
        pl.when(ambiguous_mask)
        .then(pl.lit(None))
        .otherwise(pl.col("outcome_XX"))
        .alias("outcome_XX")
    )

    # Truncate controls and set F_g to T_max + 1
    trunc_mask = (
        (pl.col("F_g_XX") < T_max_XX + 1) &
        (pl.col("last_obs_D_bef_switch_XX") < (pl.col("F_g_XX") - 1))
    )

    df = df.with_columns(
        pl.lit(np.nan).alias("trunc_control_XX")
    )

    df = df.with_columns(
        pl.when(trunc_mask)
        .then(pl.col("last_obs_D_bef_switch_XX") + 1)
        .otherwise(pl.col("trunc_control_XX"))
        .alias("trunc_control_XX")
    )


    df = df.with_columns([
        pl.when(trunc_mask)
        .then(T_max_XX + 1)
        .otherwise(pl.col("F_g_XX"))
        .alias("F_g_XX")
    ])

    # — Carry forward post-switch D (d_F_g_XX) —
    df = df.with_columns(
        pl.when(pl.col("time_XX") == pl.col("F_g_XX"))
        .then(pl.col("treatment_XX"))
        .otherwise(pl.lit(None))
        .alias("d_F_g_temp_XX")
    )

    df = df.with_columns(
        pl.col("d_F_g_temp_XX").mean().over("group_XX").alias("d_F_g_XX")
    )

    # Fill post-switch missing D if last_obs_D_bef_switch_XX == F_g_XX - 1
    post_switch_mask = (
        (pl.col("F_g_XX") < T_max_XX + 1) &
        pl.col("treatment_XX").is_null() &
        (pl.col("time_XX") > pl.col("F_g_XX")) &
        (pl.col("last_obs_D_bef_switch_XX") == pl.col("F_g_XX") - 1)
    )

    df = df.with_columns(
        pl.when(post_switch_mask)
        .then(pl.col("d_F_g_XX"))
        .otherwise(pl.col("treatment_XX"))
        .alias("treatment_XX")
    )

    df = df.drop("d_F_g_temp_XX")

    # — For never-changers: fill mid-panel D, drop post-LD_g outcomes —
    never_changer_mask_fill = (
        (pl.col("F_g_XX") == T_max_XX + 1) &
        pl.col("treatment_XX").is_null() &
        (pl.col("time_XX") > pl.col("min_time_d_nonmiss_XX")) &
        (pl.col("time_XX") < pl.col("max_time_d_nonmiss_XX"))
    )

    df = df.with_columns(
        pl.when(never_changer_mask_fill)
        .then(pl.col("d_sq_XX"))
        .otherwise(pl.col("treatment_XX"))
        .alias("treatment_XX")
    )

    # Drop outcomes after max_time_d_nonmiss_XX for never-changers
    never_changer_drop_y = (
        (pl.col("F_g_XX") == T_max_XX + 1) &
        (pl.col("time_XX") > pl.col("max_time_d_nonmiss_XX"))
    )

    df = df.with_columns(
        pl.when(never_changer_drop_y)
        .then(pl.lit(None))
        .otherwise(pl.col("outcome_XX"))
        .alias("outcome_XX")
    )

    # Set trunc_control_XX for never-changers
    df = df.with_columns(
        pl.when(pl.col("F_g_XX") == T_max_XX + 1)
        .then(pl.col("max_time_d_nonmiss_XX") + 1)
        .otherwise(pl.col("trunc_control_XX"))
        .alias("trunc_control_XX")
    )

    # — 10. Save outcome levels if predict_het —
    if predict_het is not None and len(predict_het_good) > 0:
        df = df.with_columns(pl.col("outcome_XX").alias("outcome_non_diff_XX"))

    # — 1. trends_lin adjustments (first-differencing) —
    import polars as pl

    if trends_lin:
        # 1) Drop units with F_g_XX == 2
        df = df.filter(pl.col("F_g_XX") != 2)

        # 2) Ensure sorted for group-wise lag
        df = df.sort(["group_XX", "time_XX"])

        # 3) First-differences for outcome and each control, within group_XX
        vars_to_diff = ["outcome_XX"] + (controls or [])

        df = df.with_columns([
            (
                pl.col(v) - pl.col(v).shift(1).over("group_XX")
            ).alias(v)
            for v in vars_to_diff
        ])

        # 4) Drop period 1 after differencing
        df = df.filter(pl.col("time_XX") != 1)

        # 5) t_min_XX = min time_XX in the (differenced) data
        t_min_XX = df.select(pl.col("time_XX").min()).to_series()[0]

    import polars as pl
    import numpy as np

    # 2. Balance the panel by filling missing (group_XX, time_XX) cells
    # Drop any stray column
    if "joint_trends_XX" in df.columns:
        df = df.drop("joint_trends_XX")

    # Get full Cartesian product of groups × times
    groups = df.select("group_XX").unique()
    times  = df.select("time_XX").unique()

    full_index = groups.join(times, how="cross")  # columns: group_XX, time_XX

    # Left-join original df onto the full index to fill missing cells
    df = (
        full_index
        .join(df, on=["group_XX", "time_XX"], how="left")
    )

    # 3. Recompute numeric types
    df = df.with_columns([
        pl.col("group_XX").cast(pl.Int64),
        pl.col("time_XX").cast(pl.Int64),
    ])

    # 4. Collapse baseline d_sq_XX by group
    df = df.with_columns([
        pl.col("d_sq_XX").mean().over("group_XX").alias("d_sq_XX"),
        pl.col("d_sq_int_XX").mean().over("group_XX").alias("d_sq_int_XX"),
    ])

    # 2. F_g_XX := mean(F_g_XX) by group_XX
    df = df.with_columns(
        pl.col("F_g_XX").mean().over("group_XX").alias("F_g_XX")
    )

    # 5. Define N_gt_XX
    df = df.with_columns(
        pl.when(pl.col("outcome_XX").is_null() | pl.col("treatment_XX").is_null())
        .then(0)
        .otherwise(pl.col("weight_XX"))
        .alias("N_gt_XX")
    )

    # 6. Compute F_g_trunc_XX
    F = pl.col("F_g_XX")
    Tc = pl.col("trunc_control_XX")

    F_g_trunc_expr = pl.when(F < Tc).then(F).otherwise(Tc)
    F_g_trunc_expr = pl.when(Tc.is_null()).then(F).otherwise(F_g_trunc_expr)
    F_g_trunc_expr = pl.when(F.is_null()).then(Tc).otherwise(F_g_trunc_expr)

    df = df.with_columns(
        F_g_trunc_expr.alias("F_g_trunc_XX")
    )

    # 7. Compute T_g_XX by (d_sq_XX + trends_nonparam) groups
    group_cols = ["d_sq_XX"] + (trends_nonparam or [])

    df = df.with_columns(
        (pl.col("F_g_trunc_XX").max().over(group_cols) - 1).alias("T_g_XX")
    )

    # 1. Compute average post-switch treatment by group
    df = df.with_columns(
        pl.when(
            (pl.col("time_XX") >= pl.col("F_g_XX")) &
            (pl.col("time_XX") <= pl.col("T_g_XX"))
        )
        .then(pl.col("treatment_XX"))
        .otherwise(pl.lit(None))
        .alias("treatment_XX_v1")
    )

    df = df.with_columns(
        pl.col("treatment_XX_v1").sum().over("group_XX").alias("avg_post_switch_treat_XX_temp")
    ).drop("treatment_XX_v1")

    # 2. Count post-switch periods by group
    mask_expr = (
        (pl.col("time_XX") >= pl.col("F_g_XX")) &
        (pl.col("time_XX") <= pl.col("T_g_XX"))
    )

    df = df.with_columns(
        pl.when(mask_expr & pl.col("treatment_XX").is_not_null())
        .then(1)
        .otherwise(pl.lit(None))
        .alias("count_time_post_switch_XX_temp")
    )

    df = df.with_columns(
        pl.col("count_time_post_switch_XX_temp").sum().over("group_XX").alias("count_time_post_switch_XX")
    )

    # 3. Finalize avg_post_switch_treat_XX
    df = df.with_columns(
        (pl.col("avg_post_switch_treat_XX_temp") / pl.col("count_time_post_switch_XX"))
        .alias("avg_post_switch_treat_XX_temp")
    )

    df = df.with_columns(
        pl.col("avg_post_switch_treat_XX_temp").mean().over("group_XX").alias("avg_post_switch_treat_XX")
    ).drop("avg_post_switch_treat_XX_temp")

    import polars as pl
    import numpy as np

    # 4. Define S_g_XX
    if continuous == 0:
        # mask = (avg_post == d_sq) & (F_g_XX != T_g_XX + 1)
        mask_expr = (
            (pl.col("avg_post_switch_treat_XX") == pl.col("d_sq_XX"))
            & (pl.col("F_g_XX") != (pl.col("T_g_XX") + 1))
        )
        # df = df.loc[~mask]
        df = df.filter(~mask_expr)

        # df['S_g_XX'] = (avg_post > d_sq).astype(int)
        df = df.with_columns(
            (pl.col("avg_post_switch_treat_XX") > pl.col("d_sq_XX"))
            .cast(pl.Int64)
            .alias("S_g_XX")
        )

        # mask = avg_post is null → S_g_XX = NaN
        df = df.with_columns(
            pl.when(pl.col("avg_post_switch_treat_XX").is_null())
            .then(pl.lit(None))
            .otherwise(pl.col("S_g_XX"))
            .alias("S_g_XX")
        )

        # mask = avg_post is null & d_sq not null & S_g_XX is null → S_g_XX = 1
        df = df.with_columns(
            pl.when(
                pl.col("avg_post_switch_treat_XX").is_null()
                & pl.col("d_sq_XX").is_not_null()
                & pl.col("S_g_XX").is_null()
            )
            .then(pl.lit(1))
            .otherwise(pl.col("S_g_XX"))
            .alias("S_g_XX")
        )

        # mask = (F_g_XX != T_max_XX + 1)
        # df.loc[~mask, 'S_g_XX'] = np.nan  → when F_g_XX == T_max_XX + 1 → NaN
        df = df.with_columns(
            pl.when(pl.col("F_g_XX") == (T_max_XX + 1))
            .then(pl.lit(None))
            .otherwise(pl.col("S_g_XX"))
            .alias("S_g_XX")
        )

    elif continuous > 0:
        # mask = (
        #   avg_post == d_sq_orig
        #   & avg_post notna
        #   & F_g_XX != T_g_XX + 1
        #   & F_g_XX notna
        #   & T_g_XX notna
        # )
        mask_expr = (
            (pl.col("avg_post_switch_treat_XX") == pl.col("d_sq_XX_orig"))
            & pl.col("avg_post_switch_treat_XX").is_not_null()
            & (pl.col("F_g_XX") != (pl.col("T_g_XX") + 1))
            & pl.col("F_g_XX").is_not_null()
            & pl.col("T_g_XX").is_not_null()
        )
        # df = df.loc[~mask]
        df = df.filter(~mask_expr)

        # df['S_g_XX'] = (avg_post > d_sq_orig).astype(int)
        df = df.with_columns(
            (pl.col("avg_post_switch_treat_XX") > pl.col("d_sq_XX_orig"))
            .cast(pl.Int64)
            .alias("S_g_XX")
        )

        # df.loc[df['F_g_XX'] == T_max_XX + 1, 'S_g_XX'] = np.nan
        df = df.with_columns(
            pl.when(pl.col("F_g_XX") == (T_max_XX + 1))
            .then(pl.lit(None))
            .otherwise(pl.col("S_g_XX"))
            .alias("S_g_XX")
        )

    # aux = df.groupby('group_XX')['avg_post_switch_treat_XX'].transform('min')
    # df.loc[aux.isna(), 'S_g_XX'] = np.nan
    aux_expr = pl.col("avg_post_switch_treat_XX").min().over("group_XX")
    df = df.with_columns(
        pl.when(aux_expr.is_null())
        .then(pl.lit(None))
        .otherwise(pl.col("S_g_XX"))
        .alias("S_g_XX")
    )


    # 5. Define S_g_het_XX if needed
    if (predict_het and len(predict_het) > 0) or continuous > 0:
        df = df.with_columns(
            pl.when(pl.col("S_g_XX").is_null() | (pl.col("S_g_XX") != 0))
            .then(pl.col("S_g_XX"))
            .otherwise(pl.lit(-1))
            .alias("S_g_het_XX")
        )

    import polars as pl

    import polars as pl
    import numpy as np

    if continuous > 0:
        # 6. Continuous‐specific binarization & stagger
        if controls is None:
            controls = []

        # treatment_XX_temp = (F_g_XX <= time_XX) * S_g_het_XX  if S_g_het_XX != NaN/Null, else NaN
        mask_het_notna = (
            pl.col("S_g_het_XX").is_not_null() & pl.col("S_g_het_XX").is_not_nan()
        )

        df = df.with_columns(
            pl.when(mask_het_notna)
            .then(
                (pl.col("F_g_XX") <= pl.col("time_XX"))
                .cast(pl.Float64)
                * pl.col("S_g_het_XX")
            )
            .otherwise(pl.lit(np.nan))
            .alias("treatment_XX_temp")
        )

        # treatment_XX_orig = treatment_XX
        # replace treatment_XX = treatment_XX_temp
        df = df.with_columns([
            pl.col("treatment_XX").alias("treatment_XX_orig"),
            pl.col("treatment_XX_temp").alias("treatment_XX"),
        ])

        # ---- Create time_fe_XX_ dummies / step FEs ----
        max_time_XX = int(df.select(pl.col("time_XX").max()).item())

        for t in range(1, max_time_XX + 1):
            df = df.with_columns(
                (pl.col("time_XX") >= t).cast(pl.Int64).alias(f"time_fe_XX_{t}")
            )

        # ---- Interact period-step FEs with polynomial in d_sq_XX ----
        for t in range(2, max_time_XX + 1):
            var = f"time_fe_XX_{t}"

            exprs = []
            for pol_level in range(1, degree_pol + 1):
                d_col = f"d_sq_{pol_level}_XX"
                newcol = f"{var}_bt{pol_level}_XX"
                exprs.append((pl.col(var) * pl.col(d_col)).alias(newcol))
                controls.append(newcol)

            df = df.with_columns(exprs)

            # drop the original step FE
            if var in df.columns:
                df = df.drop(var)

        # drop time_fe_XX_1 if it exists
        if "time_fe_XX_1" in df.columns:
            df = df.drop("time_fe_XX_1")


    import polars as pl

    # 1. Create treatment at F_g: D_{g,F_g}
    df = df.with_columns(
        pl.when(pl.col("time_XX") == pl.col("F_g_XX"))
        .then(pl.col("treatment_XX"))
        .otherwise(pl.lit(None))
        .alias("d_fg_XX")
    )

    # group‐wise average
    df = df.with_columns(
        pl.col("d_fg_XX").mean().over("group_XX").alias("d_fg_XX")
    )

    # if never switches (F_g == T_max + 1), fill from status‐quo d_sq_XX
    df = df.with_columns(
        pl.when(pl.col("d_fg_XX").is_null() & (pl.col("F_g_XX") == T_max_XX + 1))
        .then(pl.col("d_sq_XX"))
        .otherwise(pl.col("d_fg_XX"))
        .alias("d_fg_XX")
    )

    # 2. Create L_g_XX = T_g_XX - F_g_XX + 1
    df = df.with_columns(
        (pl.col("T_g_XX") - pl.col("F_g_XX") + 1).alias("L_g_XX")
    )

    # 3. Create L_g_placebo_XX if placebos requested
    if placebo > 0:
        df = df.with_columns(
            pl.when(pl.col("F_g_XX") >= 3)
            .then(
                pl.when(pl.col("L_g_XX") > pl.col("F_g_XX") - 2)
                    .then(pl.col("F_g_XX") - 2)
                    .otherwise(pl.col("L_g_XX"))
            )
            .otherwise(pl.lit(None))
            .alias("L_g_placebo_XX")
        )

        # replace infinite with NaN
        df = df.with_columns(
            pl.when(pl.col("L_g_placebo_XX").is_infinite())
            .then(pl.lit(None))
            .otherwise(pl.col("L_g_placebo_XX"))
            .alias("L_g_placebo_XX")
        )

    # 4. Tag first observation within each group_XX
    df = df.sort(["group_XX", "time_XX"])
    df = df.with_columns(
        pl.arange(0, pl.len()).over("group_XX").alias("_row_in_group")
    )
    df = df.with_columns(
        (pl.col("_row_in_group") == 0)
        .cast(pl.Int64)
        .alias("first_obs_by_gp_XX")
    ).drop("_row_in_group")

    # 5. If clustering specified, flag first obs in each cluster and check nesting
    if cluster is not None:
        group_col = "group_XX"
        cluster_col = "cluster_XX"
        time_col = "time_XX"

        # 1. Generate cluster_group_XX = min(cluster) by group
        df = df.with_columns(
            pl.col(cluster_col).min().over(group_col).alias("cluster_group_XX")
        )

        # Replace missing cluster with cluster_group_XX
        df = df.with_columns(
            pl.when(pl.col(cluster_col).is_null())
            .then(pl.col("cluster_group_XX"))
            .otherwise(pl.col(cluster_col))
            .alias(cluster_col)
        )

        # 2. First observation by cluster (sorted within group and time)
        df = df.sort([cluster_col, group_col, time_col])
        df = df.with_columns(
            pl.arange(0, pl.len()).over(cluster_col).alias("_row_in_cluster")
        )
        df = df.with_columns(
            (pl.col("_row_in_cluster") == 0)
            .cast(pl.Int64)
            .alias("first_obs_by_clust_XX")
        ).drop("_row_in_cluster")

        df = df.with_columns(
            pl.when(pl.col(cluster_col).is_null())
            .then(pl.lit(None))
            .otherwise(pl.col("first_obs_by_clust_XX"))
            .alias("first_obs_by_clust_XX")
        )

        # 3. Error check: group must be nested in cluster
        cluster_var = df.group_by(group_col).agg(
            pl.col(cluster_col).n_unique().alias("cluster_var_g_XX")
        )
        max_cluster_var = cluster_var["cluster_var_g_XX"].max()

        if max_cluster_var > 1:
            raise ValueError(
                "❌ The group variable should be nested within the clustering variable."
            )

    # 6. Compute first differences of outcome and treatment
    df = df.sort(["group_XX", "time_XX"])  # ensure sorted like xtset
    df = df.with_columns([
        pl.col("outcome_XX").diff().over("group_XX").alias("diff_y_XX"),
        pl.col("treatment_XX").diff().over("group_XX").alias("diff_d_XX"),
    ])


    import polars as pl

    if controls is not None and len(controls) > 0:

        # 1) Compute first differences of each control and flag missing
        count_controls = 0
        df = df.with_columns(
            pl.lit(1).alias("fd_X_all_non_missing_XX")
        )

        for var in controls:
            count_controls += 1
            diff_col = f"diff_X{count_controls}_XX"

            # group‐wise first difference
            df = df.with_columns(
                pl.col(var).diff().over("group_XX").alias(diff_col)
            )

            # if diff is NaN, mark as missing in fd_X_all_non_missing_XX
            df = df.with_columns(
                pl.when(pl.col(diff_col).is_null())
                .then(0)
                .otherwise(pl.col("fd_X_all_non_missing_XX"))
                .alias("fd_X_all_non_missing_XX")
            )

        # 2) Residualization prep
        count_controls = 0
        mycontrols_XX: list[str] = []

        for var in controls:
            count_controls += 1
            diff_col = f"diff_X{count_controls}_XX"

            # remove any stale helpers
            for tmp in ["sum_weights_control_XX", "avg_diff_temp_XX", "diff_y_wXX"]:
                if tmp in df.columns:
                    df = df.drop(tmp)

            # define mask: not-yet-switched & valid diff_y & all control diffs non-missing
            mask_expr = (
                (pl.col("ever_change_d_XX") == 0)
                & pl.col("diff_y_XX").is_not_null()
                & (pl.col("fd_X_all_non_missing_XX") == 1)
            )

            # grouping keys
            grp_cols = ["time_XX", "d_sq_XX"] + (trends_nonparam or [])

            # 2a) sum of N_gt for controls (within mask)
            df = df.with_columns(
                # N_gt_XX only inside mask, 0 otherwise
                pl.when(mask_expr)
                .then(pl.col("N_gt_XX"))
                .otherwise(0)
                .alias("_N_for_ctrl")
            )

            df = df.with_columns(
                pl.col("_N_for_ctrl").sum().over(grp_cols).alias("sum_weights_control_XX")
            )

            # set to NaN outside mask
            df = df.with_columns(
                pl.when(mask_expr)
                .then(pl.col("sum_weights_control_XX"))
                .otherwise(pl.lit(None))
                .alias("sum_weights_control_XX")
            ).drop("_N_for_ctrl")

            # 2b) weighted sum of first-diffs
            df = df.with_columns(
                (pl.col("N_gt_XX") * pl.col(diff_col)).alias("avg_diff_temp_XX")
            )

            avg_col = f"avg_diff_X{count_controls}_XX"

            df = df.with_columns(
                # only sum inside mask; else 0
                pl.when(mask_expr)
                .then(pl.col("avg_diff_temp_XX"))
                .otherwise(0)
                .alias("_avg_diff_temp_masked")
            )

            df = df.with_columns(
                pl.col("_avg_diff_temp_masked").sum().over(grp_cols).alias(avg_col)
            ).drop("_avg_diff_temp_masked")

            # set to NaN outside mask
            df = df.with_columns(
                pl.when(mask_expr)
                .then(pl.col(avg_col))
                .otherwise(pl.lit(None))
                .alias(avg_col)
            )

            # divide by sum_weights_control_XX
            df = df.with_columns(
                (pl.col(avg_col) / pl.col("sum_weights_control_XX")).alias(avg_col)
            )

            # 2c) residual (√N * (ΔX - avg ΔX))
            resid_col = f"resid_X{count_controls}_time_FE_XX"
            df = df.with_columns(
                (pl.col("N_gt_XX").sqrt() * (pl.col(diff_col) - pl.col(avg_col)))
                .fill_null(0)
                .alias(resid_col)
            )

            mycontrols_XX.append(resid_col)

            # 2d) prepare product with ΔY
            df = df.with_columns(
                (pl.col("N_gt_XX").sqrt() * pl.col("diff_y_XX")).alias("diff_y_wXX"),
                (pl.col(resid_col) * pl.col("N_gt_XX").sqrt())
                .fill_null(0)
                .alias(f"prod_X{count_controls}_Ngt_XX"),
            )

        # Prepare storage
        store_singular: dict[int, bool] = {}
        store_noresidualization_XX: list[int] = []
        levels_d_sq_XX_final: list[int] = []

        # Get unique levels of d_sq_int_XX (similar to categorical categories)
        levels_d_sq_XX = (
            df.select("d_sq_int_XX")
            .get_column("d_sq_int_XX")
            .unique()
            .to_list()
        )
        levels_d_sq_XX = sorted([int(l) for l in levels_d_sq_XX if l is not None])

        # Dictionaries to hold results
        coefs_sq = {}
        inv_Denom = {}

        # Loop over each baseline‐treatment level
        for idx, l in enumerate(levels_d_sq_XX, start=1):
            l = int(l)

            # Count distinct F_g_XX for this level
            useful = (
                df.filter(pl.col("d_sq_int_XX") == l)
                .select(pl.col("F_g_XX").n_unique())
                .item()
            )
            dict_glob[f"useful_res_{l}_XX"] = useful
            store_singular[idx] = False

            if useful > 1:
                # Subset to the observations used for theta_d
                mask_expr = (
                    (pl.col("ever_change_d_XX") == 0)
                    & pl.col("diff_y_XX").is_not_null()
                    & (pl.col("fd_X_all_non_missing_XX") == 1)
                    & (pl.col("d_sq_int_XX") == l)
                )
                data_XX = df.filter(mask_expr)

                # Build YX matrix: [Y, X_residuals..., 1]
                Y_vec = data_XX.select("diff_y_wXX").to_numpy().ravel()
                X_vec = data_XX.select(mycontrols_XX).to_numpy()
                ones = np.ones((len(Y_vec), 1))
                YX = np.hstack([Y_vec.reshape(-1, 1), X_vec, ones])

                # Compute cross‐product matrix
                overall = YX.T @ YX

                # Check if entire matrix is NaN
                e_vec = np.ones((1, overall.shape[1]))
                val = e_vec @ overall @ e_vec.T
                if np.isnan(val)[0, 0]:
                    # Singular: cannot invert or accumulate
                    store_singular[idx] = True
                    store_noresidualization_XX.append(l)
                    dict_glob[f"useful_res_{l}_XX"] = 1
                else:
                    # Extract the (k × k) block for controls
                    k = len(mycontrols_XX)
                    M = overall[1:1 + k, 1:1 + k]
                    v = overall[1:1 + k, 0]

                    # Compute θ_d via Moore-Penrose inverse
                    theta_d = np.linalg.pinv(M) @ v
                    dict_glob[f"coefs_sq_{l}_XX"] = theta_d
                    levels_d_sq_XX_final.append(l)

                    # Check invertibility
                    if abs(np.linalg.det(M)) <= 1e-16:
                        store_singular[idx] = True

                    # rmax = df['F_g_XX'].max()
                    rmax = df.select(pl.col("F_g_XX").max()).item()

                    # Indicator N_c_{l}_temp_XX (not really used except for rsum)
                    col_temp = f"N_c_{l}_temp_XX"
                    df = df.with_columns(
                        (
                            (pl.col("time_XX") >= 2)
                            & (pl.col("time_XX") <= (rmax - 1))
                            & (pl.col("time_XX") < pl.col("F_g_XX"))
                            & pl.col("diff_y_XX").is_not_null()
                        ).alias(col_temp)
                    )

                    # rsum = sum N_gt_XX where this indicator is True
                    rsum = (
                        df.filter(pl.col(col_temp))
                        .select(pl.col("N_gt_XX").sum())
                        .item()
                    )

                    # Store inverse Denominator scaled by G_XX
                    dict_glob[f"inv_Denom_{l}_XX"] = np.linalg.pinv(M) * rsum * G_XX

        # Reconstruct store_singular_XX string using original d_sq_XX levels
        store_singular_XX = ""
        levels_d_sq_bis_XX = (
            df.select("d_sq_XX")
            .get_column("d_sq_XX")
            .unique()
            .to_list()
        )
        levels_d_sq_bis_XX = sorted([int(l) for l in levels_d_sq_bis_XX if l is not None])

        for idx, l in enumerate(levels_d_sq_bis_XX, start=1):
            if store_singular.get(idx, False):
                store_singular_XX += f" {int(l)}"

        # 1. Display warnings if any Den_d^{-1} was singular
        if store_singular_XX.strip():
            warnings.warn(
                "Some control variables are not taken into account for groups with baseline "
                f"treatment equal to:{store_singular_XX}"
            )
            warnings.warn(
                "1. For these groups, the regression of ΔY on ΔX and time‐FE had fewer "
                "observations than regressors."
            )
            warnings.warn(
                "2. For these groups, one or more controls were perfectly collinear (no time variation)."
            )

        # 2. Drop levels where residualization failed entirely
        if store_noresidualization_XX:
            df = df.filter(~pl.col("d_sq_int_XX").is_in(store_noresidualization_XX))

        # 3. Prepare for the “fixed‐effect” residualization regressions
        #    Here we keep time_FE_XX as numeric, and let pandas/statsmodels handle the categorical
        df = df.with_columns(
            pl.col("time_XX").cast(pl.Int64).alias("time_FE_XX")
        )

        # Add a row id for mapping predictions back from pandas
        if "row_id" not in df.columns:
            df = df.with_row_count("row_id")

        # 4. Loop over each baseline‐treatment level we actually residualized
        # 4. Loop over each baseline‐treatment level we actually residualized
        for l in levels_d_sq_XX_final:
            l = int(l)
            outcol = f"E_y_hat_gt_int_{l}_XX"

            # Polars boolean mask for this l
            mask_expr = (pl.col("d_sq_int_XX") == l) & (pl.col("F_g_XX") > pl.col("time_XX"))

            # Subset of rows used in that regression (as Polars)
            data_reg_pl = df.filter(mask_expr)

            # If no rows, just create a null column and continue
            if data_reg_pl.height == 0:
                df = df.with_columns(pl.lit(None).alias(outcol))
                continue

            # Convert subset to pandas for statsmodels
            data_reg = data_reg_pl.to_pandas()

            # Ensure time_FE_XX is categorical
            if not pd.api.types.is_categorical_dtype(data_reg["time_FE_XX"]):
                data_reg["time_FE_XX"] = data_reg["time_FE_XX"].astype("category")

            # reorder the categories so that 2 is the base level
            cats = list(data_reg["time_FE_XX"].cat.categories)
            if 2 in cats:
                new_order = [2] + [c for c in cats if c != 2]
                data_reg["time_FE_XX"] = data_reg["time_FE_XX"].cat.reorder_categories(
                    new_order, ordered=True
                )

            # build the formula: diff_y_XX ~ diff_X1_XX + ... + diff_Xk_XX + C(time_FE_XX) -1
            fe_terms = [f"diff_X{c}_XX" for c in range(1, count_controls + 1)]
            formula = "diff_y_XX ~ " + " + ".join(fe_terms + ["C(time_FE_XX)"]) + " -1"

            # fit weighted least squares
            model = smf.wls(formula, data=data_reg, weights=data_reg["weight_XX"]).fit()
            data_reg["y_hat"] = model.predict(data_reg)

            # Build a small Polars DF with row_id and predictions
            pred_pl = pl.from_pandas(data_reg[["row_id", "y_hat"]]).rename({"y_hat": outcol})

            # Left-join predictions back to the full df on row_id
            df = df.join(pred_pl, on="row_id", how="left")

            # Now df[outcol] is:
            #  - y_hat for rows where mask_expr is True
            #  - null for rows where mask_expr is False
            # which matches:
            # df[outcol] = np.nan; df.loc[mask, outcol] = data_reg["y_hat"]

        # 5. Clean up any numeric dummy columns if you created them earlier:
        for t in range(2, int(T_max_XX) + 1):
            col = f"time_FE_XX{t}"
            if col in df.columns:
                df = df.drop(col)

        # 6. Drop the temporary factor column
        if "time_FE_XX" in df.columns:
            df = df.drop("time_FE_XX")
                

    import polars as pl
    import numpy as np

    # Initialize
    L_u_XX = None
    L_a_XX = None
    L_placebo_u_XX = None
    L_placebo_a_XX = None

    # For switchers in
    if switchers in ("", "in"):
        # S_g_XX == 1
        df_in = df.filter(pl.col("S_g_XX") == 1)
        if df_in.height > 0:
            L_u_XX = df_in.select(pl.col("L_g_XX").max()).item()
        else:
            L_u_XX = 0

        if placebo > 0:
            if df_in.height > 0 and "L_g_placebo_XX" in df.columns:
                L_placebo_u_XX = df_in.select(pl.col("L_g_placebo_XX").max()).item()
            else:
                L_placebo_u_XX = 0
            # Enforce non-negativity
            L_placebo_u_XX = max(L_placebo_u_XX, 0)
            if trends_lin:
                L_placebo_u_XX -= 1

    if switchers in ("", "out"):
        # compute L_a_XX
        vals_df = df.filter(pl.col("S_g_XX") == 0).select("L_g_XX")
        vals = vals_df.get_column("L_g_XX").drop_nulls()
        if vals.len() == 0:
            L_a_XX = 0
        else:
            L_a_XX = float(vals.max())

        # placebo case
        if placebo != 0:
            vals_pl_df = df.filter(pl.col("S_g_XX") == 0).select("L_g_placebo_XX")
            vals_placebo = vals_pl_df.get_column("L_g_placebo_XX").drop_nulls()

            if vals_placebo.len() == 0:
                L_placebo_a_XX = 0
            else:
                L_placebo_a_XX = float(vals_placebo.max())

            # replace negatives with 0
            if L_placebo_a_XX < 0:
                L_placebo_a_XX = 0

            # subtract 1 if trends_lin is True
            if bool(trends_lin):
                L_placebo_a_XX = L_placebo_a_XX - 1


    # Design‐restriction check (unchanged)
    if (
        (switchers == "in" and (L_u_XX is None or L_u_XX == 0)) or
        (switchers == "out" and (L_a_XX is None or L_a_XX == 0)) or
        (switchers == "" and ((L_u_XX is None or L_u_XX == 0) and (L_a_XX is None or L_a_XX == 0)))
    ):
        raise ValueError(
            "No treatment effect can be estimated.\n"
            "This is because Design Restriction 1 in de Chaisemartin & D'Haultfoeuille (2024) "
            "is not satisfied in the data, given the options requested. "
            "This may be due to continuous-period-one treatments or lack of variation. "
            "Try specifying the continuous option or check your data."
        )

    # Determine the number of effects to estimate (unchanged)
    if switchers == "":
        l_XX = int(min(np.nanmax(np.array([L_a_XX, L_u_XX])), effects))
        if placebo != 0:
            l_placebo_XX = np.nanmax(np.array([L_placebo_a_XX, L_placebo_u_XX]))
            l_placebo_XX = np.nanmin(np.array([l_placebo_XX, placebo]))
            l_placebo_XX = np.nanmin(np.array([l_placebo_XX, effects]))
        else:
            l_placebo_XX = 0

    elif switchers == "in":
        l_XX = int(np.nanmin(np.array([effects, L_u_XX])))
        if placebo != 0:
            l_placebo_XX = int(np.nanmin(np.array([placebo, L_placebo_u_XX])))
            l_placebo_XX = int(np.nanmin(np.array([l_placebo_XX, effects])))
        else:
            l_placebo_XX = 0

    else:  # switchers == "out"
        l_XX = int(np.nanmin(np.array([effects, L_a_XX])))
        if placebo != 0:
            l_placebo_XX = int(np.nanmin(np.array([placebo, L_placebo_a_XX])))
            l_placebo_XX = int(np.nanmin(np.array([l_placebo_XX, effects])))
        else:
            l_placebo_XX = 0

    # 1. Warn if the user requested too many effects or placebos (unchanged)
    if l_XX < effects:
        warnings.warn(
            f"The number of effects requested is too large. "
            f"The number of effects which can be estimated is at most {l_XX}. "
            f"Trying to estimate {l_XX} effect(s)."
        )

    if placebo != 0:
        if l_placebo_XX < placebo and effects >= placebo:
            warnings.warn(
                f"The number of placebos which can be estimated is at most {l_placebo_XX}. "
                f"Trying to estimate {l_placebo_XX} placebo(s)."
            )
        if effects < placebo:
            warnings.warn(
                f"The number of placebos requested cannot be larger than the number "
                f"of effects requested. Cannot compute more than {l_placebo_XX} placebo(s)."
            )

    # 2. Compute adjustment windows for placebo tests  (pandas → polars)
    df = df.with_columns(
        pl.when(pl.col("S_g_XX").is_not_null())
        .then(pl.col("F_g_XX") - 2 - pl.col("L_g_XX"))
        .otherwise(pl.lit(np.nan))
        .alias("pl_gap_XX")
    )

    max_pl_u_XX = max_pl_a_XX = max_pl_gap_u_XX = max_pl_gap_a_XX = 0
    if switchers in ("", "in"):
        df_in = df.filter(pl.col("S_g_XX") == 1)
        if df_in.height > 0:
            max_pl_u_XX = df_in.select(pl.col("F_g_XX").max()).item() - 2
            max_pl_gap_u_XX = df_in.select(pl.col("pl_gap_XX").max()).item()
    if switchers in ("", "out"):
        df_out = df.filter(pl.col("S_g_XX") == 0)
        if df_out.height > 0:
            max_pl_a_XX = df_out.select(pl.col("F_g_XX").max()).item() - 2
            max_pl_gap_a_XX = df_out.select(pl.col("pl_gap_XX").max()).item()

    max_pl_XX     = max(max_pl_u_XX,     max_pl_a_XX)
    max_pl_gap_XX = max(max_pl_gap_u_XX, max_pl_gap_a_XX)

    # clean up temporary column
    df = df.drop("pl_gap_XX")

    # 3. Initialize accumulation variables and DataFrame columns
    inh_obj = []

    # a) For each effect k = 1..l_XX, create zeroed columns  (pandas → polars)
    for k in range(1, l_XX + 1):
        k = int(k)
        df = df.with_columns([
            pl.lit(0).alias(f"U_Gg{k}_plus_XX"),
            pl.lit(0).alias(f"U_Gg{k}_minus_XX"),
            pl.lit(0).alias(f"count{k}_plus_XX"),
            pl.lit(0).alias(f"count{k}_minus_XX"),
            pl.lit(0).alias(f"U_Gg_var_{k}_in_XX"),
            pl.lit(0).alias(f"U_Gg_var_{k}_out_XX"),
            pl.lit(0).alias(f"delta_D_g_{k}_plus_XX"),
            pl.lit(0).alias(f"delta_D_g_{k}_minus_XX"),
        ])

    # b) Global counters for in-group and out-group variance sums (unchanged)
    sum_for_var_in_XX  = 0
    sum_for_var_out_XX = 0
    inh_obj.extend(['sum_for_var_in_XX', 'sum_for_var_out_XX'])

    # c) If placebo tests requested, initialize their columns too (pandas → polars)
    if placebo != 0:
        for k in range(1, l_XX + 1):
            k = int(k)
            df = df.with_columns([
                pl.lit(0).alias(f"U_Gg_pl_{k}_plus_XX"),
                pl.lit(0).alias(f"U_Gg_pl_{k}_minus_XX"),
                pl.lit(0).alias(f"count{k}_pl_plus_XX"),
                pl.lit(0).alias(f"count{k}_pl_minus_XX"),
                pl.lit(0).alias(f"U_Gg_var_pl_{k}_in_XX"),
                pl.lit(0).alias(f"U_Gg_var_pl_{k}_out_XX"),
            ])
        sum_for_var_placebo_in_XX  = 0
        sum_for_var_placebo_out_XX = 0
        inh_obj.extend(['sum_for_var_placebo_in_XX', 'sum_for_var_placebo_out_XX'])

    # d) Create de-weighted and raw counters per effect / placebo (unchanged)
    for i in range(1, l_XX + 1):
        i = int(i)
        dict_glob[f"N1_{i}_XX"]     = 0
        dict_glob[f"N1_{i}_XX_new"] = 0
        dict_glob[f"N1_dw_{i}_XX"]  = 0
        dict_glob[f"N0_{i}_XX"]     = 0
        dict_glob[f"N0_{i}_XX_new"] = 0
        dict_glob[f"N0_dw_{i}_XX"]  = 0

        inh_obj.extend([
            f"N1_{i}_XX", f"N1_{i}_XX_new", f"N1_dw_{i}_XX",
            f"N0_{i}_XX", f"N0_{i}_XX_new", f"N0_dw_{i}_XX",
        ])

        if normalized:
            dict_glob[f"delta_D_{i}_in_XX"]  = 0
            dict_glob[f"delta_D_{i}_out_XX"] = 0
            inh_obj.extend([f"delta_D_{i}_in_XX", f"delta_D_{i}_out_XX"])

        if placebo != 0:
            dict_glob[f"N1_placebo_{i}_XX"]     = 0
            dict_glob[f"N1_placebo_{i}_XX_new"] = 0
            dict_glob[f"N1_dw_placebo_{i}_XX"]  = 0
            dict_glob[f"N0_placebo_{i}_XX"]     = 0
            dict_glob[f"N0_placebo_{i}_XX_new"] = 0
            dict_glob[f"N0_dw_placebo_{i}_XX"]  = 0
            inh_obj.extend([
                f"N1_placebo_{i}_XX", f"N1_placebo_{i}_XX_new", f"N1_dw_placebo_{i}_XX",
                f"N0_placebo_{i}_XX", f"N0_placebo_{i}_XX_new", f"N0_dw_placebo_{i}_XX",
            ])
            if normalized:
                dict_glob[f"delta_D_pl_{i}_in_XX"]  = 0
                dict_glob[f"delta_D_pl_{i}_out_XX"] = 0
                inh_obj.extend([f"delta_D_pl_{i}_in_XX", f"delta_D_pl_{i}_out_XX"])

    # 1. Initialize DataFrame columns and scalar counters (pandas → polars)
    df = df.with_columns([
        pl.lit(0).alias("U_Gg_plus_XX"),
        pl.lit(0).alias("U_Gg_minus_XX"),
    ])

    U_Gg_den_plus_XX  = 0
    U_Gg_den_minus_XX = 0
    sum_N1_l_XX       = 0
    sum_N0_l_XX       = 0
    inh_obj.extend([
        "U_Gg_den_plus_XX",
        "U_Gg_den_minus_XX",
        "sum_N1_l_XX",
        "sum_N0_l_XX"
    ])

    df = df.with_columns([
        pl.lit(0).alias("U_Gg_var_plus_XX"),
        pl.lit(0).alias("U_Gg_var_minus_XX"),
    ])

    # 2. Collect inherited scalars into a dict (unchanged)
    const = {}
    for v in inh_obj:
        if v in dict_glob.keys():
            const[v] = dict_glob.get(v)
        elif v in locals().keys():
            const[v] = locals().get(v)
        else:
            const[v] = np.nan

    # 3. Save unchanging globals (unchanged)
    gs = ["L_u_XX", "L_a_XX", "l_XX", "t_min_XX", "T_max_XX", "G_XX"]
    if placebo != 0:
        gs += ["L_placebo_u_XX", "L_placebo_a_XX"]

    globals_dict = {}
    for v in gs:
        if v in dict_glob.keys():
            globals_dict[v] = dict_glob.get(v)
        elif v in locals().keys():
            globals_dict[v] = locals().get(v)
        else:
            globals_dict[v] = np.nan

    # 4. Collect control-specific objects if any (unchanged)
    controls_globals = {}
    if controls is not None:
        for lvl in levels_d_sq_XX:
            controls_globals[f"useful_res_{int(lvl)}_XX"] = dict_glob.get(f"useful_res_{int(lvl)}_XX")
            controls_globals[f"coefs_sq_{int(lvl)}_XX"]   = dict_glob.get(f"coefs_sq_{int(lvl)}_XX")
            controls_globals[f"inv_Denom_{int(lvl)}_XX"]  = dict_glob.get(f"inv_Denom_{int(lvl)}_XX")

    # 5. Tag switchers by event-study effect number (pandas → polars)
    df = df.with_columns(
        pl.lit(np.nan).alias("switcher_tag_XX")
    )

    dict_glob['U_Gg_den_plus_XX'] = 0
    dict_glob['U_Gg_den_minus_XX'] = 0
    dict_glob['sum_N1_l_XX'] = 0
    dict_glob['sum_N0_l_XX'] = 0

    if switchers in ["", "in"]:
        if L_u_XX is not None and L_u_XX != 0:

            # Make sure this column exists before we start assigning to it
            if "switcher_tag_XX" not in df.columns:
                df = df.with_columns(pl.lit(None).alias("switcher_tag_XX"))

            # =========================
            # First run (non-linear trends)
            # =========================
            if not trends_lin:  # if trends lin is false

                data = did_multiplegt_dyn_core_pl(
                    df,
                    outcome="outcome_XX",
                    group="group_XX",
                    time="time_XX",
                    treatment="treatment_XX",
                    effects=l_XX,
                    placebo=l_placebo_XX,
                    switchers_core="in",
                    trends_nonparam=trends_nonparam,
                    controls=controls,
                    same_switchers=same_switchers,
                    same_switchers_pl=same_switchers_pl,
                    only_never_switchers=only_never_switchers,
                    normalized=normalized,
                    globals_dict=globals_dict,
                    dict_glob=dict_glob,
                    const=const,
                    trends_lin=trends_lin,
                    controls_globals=controls_globals,
                    less_conservative_se=less_conservative_se,
                    continuous=continuous,
                    cluster=cluster,
                    **const,
                )

                df = data["df"]

                for keyval in data["const"].keys():
                    const[keyval] = data["const"][keyval]
                    dict_glob[keyval] = data["const"][keyval]

                # Assign switcher_tag_XX where distance_to_switch_k == 1
                for k in range(1, l_XX + 1):
                    col_dist_k = f"distance_to_switch_{k}_XX"
                    df = df.with_columns(
                        pl.when(pl.col(col_dist_k) == 1)
                        .then(pl.lit(k))
                        .otherwise(pl.col("switcher_tag_XX"))
                        .alias("switcher_tag_XX")
                    )

            # =========================
            # Main loop over effects
            # =========================
            for i in range(1, l_XX + 1):
                i = int(i)
                # print(f"Usamos trends lin {trends_lin} and this number effect {i}")

                if trends_lin:
                    data = did_multiplegt_dyn_core_pl(
                        df,
                        outcome="outcome_XX",
                        group="group_XX",
                        time="time_XX",
                        treatment="treatment_XX",
                        effects=i,
                        placebo=0,
                        switchers_core="in",
                        trends_nonparam=trends_nonparam,
                        controls=controls,
                        same_switchers=True,  # change 1
                        same_switchers_pl=same_switchers_pl,
                        only_never_switchers=only_never_switchers,
                        normalized=normalized,
                        globals_dict=globals_dict,
                        dict_glob=dict_glob,
                        const=const,
                        trends_lin=trends_lin,
                        controls_globals=controls_globals,
                        less_conservative_se=less_conservative_se,
                        continuous=continuous,
                        cluster=cluster,
                        **const,
                    )
                    df = data["df"]
                    for keyval in data["const"].keys():
                        const[keyval] = data["const"][keyval]
                        dict_glob[keyval] = data["const"][keyval]

                    # update switcher_tag_XX where distance_to_switch_i == 1
                    col_dist_i = f"distance_to_switch_{i}_XX"
                    df = df.with_columns(
                        pl.when(pl.col(col_dist_i) == 1)
                        .then(pl.lit(i))
                        .otherwise(pl.col("switcher_tag_XX"))
                        .alias("switcher_tag_XX")
                    )

                # ===== N1_i lookup =====
                key_N1 = f"N1_{i}_XX"
                if key_N1 in dict_glob:
                    N1_i = dict_glob.get(key_N1)
                else:
                    N1_i = 0

                # ===== Copy columns (Polars way) =====
                if N1_i != 0:
                    df = df.with_columns([
                        pl.col(f"U_Gg{i}_XX").alias(f"U_Gg{i}_plus_XX"),
                        pl.col(f"count{i}_core_XX").alias(f"count{i}_plus_XX"),
                        pl.col(f"U_Gg{i}_var_XX").alias(f"U_Gg_var_{i}_in_XX"),
                    ])

                    dict_glob[f"N1_{i}_XX_new"] = N1_i
                    const[f"N1_{i}_XX_new"] = N1_i

                    if normalized:
                        dict_glob[f"delta_D_{i}_in_XX"] = dict_glob.get(f"delta_norm_{i}_XX")
                        const[f"delta_D_{i}_in_XX"] = dict_glob[f"delta_D_{i}_in_XX"]

                    if not trends_lin:
                        df = df.with_columns(
                            pl.col(f"delta_D_g_{i}_XX").alias(f"delta_D_g_{i}_plus_XX")
                        )

            # =========================
            # Placebo loop
            # =========================
            if l_placebo_XX != 0:
                for i in range(1, int(l_placebo_XX + 1)):
                    i = int(i)

                    if trends_lin:
                        merged = {**const, **controls_globals}  # Python 3.8 compatible
                        data = did_multiplegt_dyn_core_pl(
                            df,
                            outcome="outcome_XX",
                            group="group_XX",
                            time="time_XX",
                            treatment="treatment_XX",
                            effects=i,
                            placebo=i,
                            switchers_core="in",
                            trends_nonparam=trends_nonparam,
                            controls=controls,
                            same_switchers=True,
                            same_switchers_pl=True,
                            only_never_switchers=only_never_switchers,
                            normalized=normalized,
                            globals_dict=globals_dict,
                            dict_glob=dict_glob,
                            const=const,
                            trends_lin=trends_lin,
                            controls_globals=controls_globals,
                            less_conservative_se=less_conservative_se,
                            continuous=continuous,
                            cluster=cluster,
                            **const,
                        )
                        df = data["df"]
                        for keyval in data["const"].keys():
                            const[keyval] = data["const"][keyval]
                            dict_glob[keyval] = data["const"][keyval]

                        col_dist_i = f"distance_to_switch_{i}_XX"
                        df = df.with_columns(
                            pl.when(pl.col(col_dist_i) == 1)
                            .then(pl.lit(i))
                            .otherwise(pl.col("switcher_tag_XX"))
                            .alias("switcher_tag_XX")
                        )

                    N1_pl = dict_glob.get(f"N1_placebo_{i}_XX", 0)
                    if N1_pl != 0:
                        df = df.with_columns([
                            pl.col(f"U_Gg_placebo_{i}_XX").alias(f"U_Gg_pl_{i}_plus_XX"),
                            pl.col(f"count{i}_pl_core_XX").alias(f"count{i}_pl_plus_XX"),
                            pl.col(f"U_Gg_pl_{i}_var_XX").alias(f"U_Gg_var_pl_{i}_in_XX"),
                        ])

                        dict_glob[f"N1_placebo_{i}_XX_new"] = N1_pl
                        const[f"N1_placebo_{i}_XX_new"] = N1_pl

                        if normalized:
                            dict_glob[f"delta_D_pl_{i}_in_XX"] = dict_glob.get(
                                f"delta_norm_pl_{i}_XX"
                            )
                            const[f"delta_D_pl_{i}_in_XX"] = dict_glob[
                                f"delta_D_pl_{i}_in_XX"
                            ]

            # =========================
            # Last block
            # =========================
            # Python-level condition
            if (not trends_lin) and (dict_glob.get("sum_N1_l_XX", 0) != 0):
                df = df.with_columns(
                    U_Gg_plus_XX     = pl.col("U_Gg_XX"),
                    U_Gg_den_plus_XX = pl.col("U_Gg_den_XX"),
                    U_Gg_var_plus_XX = pl.col("U_Gg_var_XX"),
                )

    if switchers in ["", "out"]:
        if bool(~np.isnan(L_a_XX)) and L_a_XX != 0:

            # ensure the tag column exists once
            if "switcher_tag_XX" not in df.columns:
                df = df.with_columns(pl.lit(None).alias("switcher_tag_XX"))

            if not trends_lin:
                data = did_multiplegt_dyn_core_pl(
                    df,
                    outcome="outcome_XX",
                    group="group_XX",
                    time="time_XX",
                    treatment="treatment_XX",
                    effects=l_XX,
                    placebo=l_placebo_XX,
                    switchers_core="out",
                    trends_nonparam=trends_nonparam,
                    controls=controls,
                    same_switchers=same_switchers,
                    same_switchers_pl=same_switchers_pl,
                    only_never_switchers=only_never_switchers,
                    normalized=normalized,
                    globals_dict=globals_dict,
                    dict_glob=dict_glob,
                    const=const,
                    trends_lin=trends_lin,
                    controls_globals=controls_globals,
                    less_conservative_se=less_conservative_se,
                    continuous=continuous,
                    cluster=cluster,
                    **const,
                )

                df = data["df"]  # should be a Polars DataFrame; if not:
                # df = pl.from_pandas(df)

                for e, val in data["const"].items():
                    const[e] = val
                    dict_glob[e] = val

                # tag switchers for all k
                for k in range(1, int(l_XX) + 1):
                    k = int(k)
                    df = df.with_columns(
                        pl.when(pl.col(f"distance_to_switch_{k}_XX") == 1)
                        .then(k)
                        .otherwise(pl.col("switcher_tag_XX"))
                        .alias("switcher_tag_XX")
                    )

            for i in range(1, int(l_XX) + 1):
                i = int(i)

                if trends_lin:
                    merged = {**const, **controls_globals}
                    data = did_multiplegt_dyn_core_pl(
                        df,
                        outcome="outcome_XX",
                        group="group_XX",
                        time="time_XX",
                        treatment="treatment_XX",
                        effects=i,
                        placebo=0,
                        switchers_core="out",
                        trends_nonparam=trends_nonparam,
                        controls=controls,
                        same_switchers=True,
                        same_switchers_pl=same_switchers_pl,
                        only_never_switchers=only_never_switchers,
                        normalized=normalized,
                        globals_dict=globals_dict,
                        dict_glob=dict_glob,
                        const=const,
                        trends_lin=trends_lin,
                        controls_globals=controls_globals,
                        less_conservative_se=less_conservative_se,
                        continuous=continuous,
                        cluster=cluster,
                        **const,
                    )

                    df = data["df"]  # Polars DF
                    for keyval, val in data["const"].items():
                        const[keyval] = val
                        dict_glob[keyval] = val

                    # update tag for this i
                    df = df.with_columns(
                        pl.when(pl.col(f"distance_to_switch_{i}_XX") == 1)
                        .then(i)
                        .otherwise(pl.col("switcher_tag_XX"))
                        .alias("switcher_tag_XX")
                    )

                # N0_i handling
                key_N0_i = f"N0_{i}_XX"
                if key_N0_i in dict_glob:
                    N0_i = dict_glob[key_N0_i]
                else:
                    N0_i = 0

                if N0_i != 0:
                    # create minus / var / count columns
                    df = df.with_columns([
                        (-pl.col(f"U_Gg{i}_XX")).alias(f"U_Gg{i}_minus_XX"),
                        pl.col(f"count{i}_core_XX").alias(f"count{i}_minus_XX"),
                        (-pl.col(f"U_Gg{i}_var_XX")).alias(f"U_Gg_var_{i}_out_XX"),
                    ])

                    dict_glob[f"N0_{i}_XX_new"] = N0_i
                    const[f"N0_{i}_XX_new"] = N0_i

                    if normalized:
                        dict_glob[f"delta_D_{i}_out_XX"] = dict_glob.get(
                            f"delta_norm_{i}_XX"
                        )
                        const[f"delta_D_{i}_out_XX"] = dict_glob[f"delta_D_{i}_out_XX"]

                    if not trends_lin:
                        df = df.with_columns(
                            pl.col(f"delta_D_g_{i}_XX").alias(f"delta_D_g_{i}_minus_XX")
                        )

            # placebo part
            if l_placebo_XX != 0:
                for i in range(1, int(l_placebo_XX) + 1):
                    i = int(i)

                    if trends_lin:
                        merged = {**const, **controls_globals}
                        data = did_multiplegt_dyn_core_pl(
                            df,
                            outcome="outcome_XX",
                            group="group_XX",
                            time="time_XX",
                            treatment="treatment_XX",
                            effects=i,
                            placebo=i,
                            switchers_core="out",
                            trends_nonparam=trends_nonparam,
                            controls=controls,
                            same_switchers=True,
                            same_switchers_pl=True,
                            only_never_switchers=only_never_switchers,
                            normalized=normalized,
                            globals_dict=globals_dict,
                            dict_glob=dict_glob,
                            const=const,
                            trends_lin=trends_lin,
                            controls_globals=controls_globals,
                            less_conservative_se=less_conservative_se,
                            continuous=continuous,
                            cluster=cluster,
                            merged=merged,
                        )
                        df = data["df"]

                        for keyval, val in data["const"].items():
                            const[keyval] = val
                            dict_glob[keyval] = val

                        # NOTE: your original code used k here; I assume you meant i.
                        df = df.with_columns(
                            pl.when(pl.col(f"distance_to_switch_{i}_XX") == 1)
                            .then(i)
                            .otherwise(pl.col("switcher_tag_XX"))
                            .alias("switcher_tag_XX")
                        )

                    key_N0_pl = f"N0_placebo_{i}_XX"
                    if key_N0_pl in dict_glob:
                        N0_pl = dict_glob[key_N0_pl]
                    else:
                        N0_pl = 0

                    if N0_pl != 0:
                        df = df.with_columns([
                            (-pl.col(f"U_Gg_placebo_{i}_XX")).alias(
                                f"U_Gg_pl_{i}_minus_XX"
                            ),
                            pl.col(f"count{i}_pl_core_XX").alias(
                                f"count{i}_pl_minus_XX"
                            ),
                            (-pl.col(f"U_Gg_pl_{i}_var_XX")).alias(
                                f"U_Gg_var_pl_{i}_out_XX"
                            ),
                        ])

                        dict_glob[f"N0_placebo_{i}_XX_new"] = N0_pl
                        const[f"N0_placebo_{i}_XX_new"] = N0_pl

                        if normalized:
                            dict_glob[f"delta_D_pl_{i}_out_XX"] = dict_glob.get(
                                f"delta_norm_pl_{i}_XX"
                            )
                            const[f"delta_D_pl_{i}_out_XX"] = dict_glob[
                                f"delta_D_pl_{i}_out_XX"
                            ]

            # final minus columns
            if not trends_lin and dict_glob["sum_N0_l_XX"] != 0:
                df = df.with_columns([
                    (-pl.col("U_Gg_XX")).alias("U_Gg_minus_XX"),
                    pl.col("U_Gg_den_XX").alias("U_Gg_den_minus_XX"),
                    (-pl.col("U_Gg_var_XX")).alias("U_Gg_var_minus_XX")
                ])


    # --------------------------------------------------------------
    # Collect results into matrix
    # --------------------------------------------------------------
    rownames = []
    mat_res_XX = np.full((int(l_XX + l_placebo_XX + 1), 9), np.nan)

    # --------------------------
    # Loop over effects
    # --------------------------
    for i in range(1, l_XX + 1):
        i = int(i)
        N1_new = dict_glob.get(f"N1_{i}_XX_new", 0)
        N0_new = dict_glob.get(f"N0_{i}_XX_new", 0)
        # print("Number of N1")
        # print(N1_new)
        # print("Number of N0")
        # print(N0_new)

        # --- U_Gg global ---
        df = df.with_columns(
            ((N1_new / (N1_new + N0_new)) * pl.col(f"U_Gg{i}_plus_XX")
            + (N0_new / (N1_new + N0_new)) * pl.col(f"U_Gg{i}_minus_XX"))
            .alias(f"U_Gg{i}_global_XX")
        )
        df = df.with_columns(
            pl.when(pl.col("first_obs_by_gp_XX") == 0)
            .then(pl.lit(None))
            .otherwise(pl.col(f"U_Gg{i}_global_XX"))
            .alias(f"U_Gg{i}_global_XX")
        )

        # --- Stepwise count selection logic ---
        col_plus = f"count{i}_plus_XX"
        col_minus = f"count{i}_minus_XX"
        col_global = f"count{i}_global_XX"

        df = df.with_columns(
            pl.when(pl.col(col_plus) > pl.col(col_minus))
            .then(pl.col(col_plus))
            .otherwise(pl.col(col_minus))
            .alias(col_global)
        )
        df = df.with_columns(
            pl.when(pl.col(col_plus).is_null())
            .then(pl.col(col_minus))
            .otherwise(pl.col(col_global))
            .alias(col_global)
        )
        df = df.with_columns(
            pl.when(pl.col(col_minus).is_null())
            .then(pl.col(col_plus))
            .otherwise(pl.col(col_global))
            .alias(col_global)
        )
        df = df.with_columns(
            pl.when(pl.col(col_minus).is_null() & pl.col(col_plus).is_null())
            .then(pl.lit(None))
            .otherwise(pl.col(col_global))
            .alias(col_global)
        )
        df = df.with_columns(
            pl.when(pl.col(col_global) == float("-inf"))
            .then(pl.lit(None))
            .otherwise(pl.col(col_global))
            .alias(col_global)
        )

        # --- count_dw indicator ---
        df = df.with_columns(
            ((pl.col(col_global).is_not_null()) & (pl.col(col_global) > 0))
            .cast(pl.Int64)
            .alias(f"count{i}_global_dwXX")
        )

        # --- Normalized deltas ---
        if normalized:
            dict_glob[f"delta_D_{i}_global_XX"] = (
                (N1_new / (N1_new + N0_new)) * dict_glob.get(f"delta_D_{i}_in_XX", 0)
                + (N0_new / (N1_new + N0_new)) * dict_glob.get(f"delta_D_{i}_out_XX", 0)
            )

        # --- switcher counts ---
        dict_glob[f"N_switchers_effect_{i}_XX"] = N1_new + N0_new
        dict_glob[f"N_switchers_effect_{i}_dwXX"] = (
            dict_glob.get(f"N1_dw_{i}_XX", 0) + dict_glob.get(f"N0_dw_{i}_XX", 0)
        )
        mat_res_XX[i - 1, 7] = dict_glob[f"N_switchers_effect_{i}_XX"]
        mat_res_XX[i - 1, 5] = dict_glob[f"N_switchers_effect_{i}_dwXX"]
        mat_res_XX[i - 1, 8] = i

        # --- Number of observations ---
        df_count = df.select(pl.col(col_global)).to_series()
        dict_glob[f"N_effect_{i}_XX"] = float(df_count.sum())
        dict_glob[f"N_effect_{i}_dwXX"] = float(
            df.select(pl.col(f"count{i}_global_dwXX")).to_series().sum()
        )
        mat_res_XX[i - 1, 6] = dict_glob[f"N_effect_{i}_XX"]
        mat_res_XX[i - 1, 4] = int(dict_glob[f"N_effect_{i}_dwXX"])

        # --- Missing check ---
        if dict_glob[f"N_switchers_effect_{i}_XX"] == 0 or dict_glob[f"N_effect_{i}_XX"] == 0:
            warnings.warn(
                f"Effect {i} cannot be estimated. No switcher or control for this effect.",
                UserWarning
            )

        # --- DID computation ---
        df = df.with_columns(
            (pl.col(f"U_Gg{i}_global_XX").sum() / G_XX).alias(f"DID_{i}_XX")
        )
        if normalized:
            df = df.with_columns(
                (pl.col(f"DID_{i}_XX") / dict_glob[f"delta_D_{i}_global_XX"]).alias(f"DID_{i}_XX")
            )
        dict_glob[f"DID_{i}_XX"] = float(df.select(pl.col(f"DID_{i}_XX")).mean()[0, 0])

        # --- Skip missing cases ---
        if (
            (switchers == "" and N1_new == 0 and N0_new == 0)
            or (switchers == "out" and N0_new == 0)
            or (switchers == "in" and N1_new == 0)
        ):
            dict_glob[f"DID_{i}_XX"] = np.nan

        mat_res_XX[i - 1, 0] = dict_glob[f"DID_{i}_XX"]

    import polars as pl
    import numpy as np

    # --------------------------
    # Average total effect (Polars)
    # --------------------------

    # Compute U_Gg_den_plus_XX
    # Compute U_Gg_den_plus_XX
    if "U_Gg_den_plus_XX" in df.columns:
        U_Gg_den_plus_XX = float(
            np.nanmean(df["U_Gg_den_plus_XX"].to_numpy())
        )
        U_Gg_den_plus_XX = 0 if np.isnan(U_Gg_den_plus_XX) else U_Gg_den_plus_XX
    else:
        U_Gg_den_plus_XX = 0

    # Compute U_Gg_den_minus_XX
    # Compute U_Gg_den_minus_XX
    if "U_Gg_den_minus_XX" in df.columns:
        U_Gg_den_minus_XX = float(
            np.nanmean(df["U_Gg_den_minus_XX"].to_numpy())
        )
        U_Gg_den_minus_XX = 0 if np.isnan(U_Gg_den_minus_XX) else U_Gg_den_minus_XX
    else:
        U_Gg_den_minus_XX = 0

    if not trends_lin:
        # weights w_plus_XX depending on switchers
        if switchers == "":
            denom = (
                U_Gg_den_plus_XX * dict_glob["sum_N1_l_XX"]
                + U_Gg_den_minus_XX * dict_glob["sum_N0_l_XX"]
            )
            w_plus_XX = (
                U_Gg_den_plus_XX * dict_glob["sum_N1_l_XX"] / denom
                if denom != 0 else 0
            )
        elif switchers == "out":
            w_plus_XX = 0
        elif switchers == "in":
            w_plus_XX = 1
        else:
            w_plus_XX = 0  # fallback

        # U_Gg_global_XX = weighted combination of plus / minus
        df = df.with_columns(
            (
                w_plus_XX * pl.col("U_Gg_plus_XX")
                + (1 - w_plus_XX) * pl.col("U_Gg_minus_XX")
            ).alias("U_Gg_global_XX")
        )

        # Set to null where first_obs_by_gp_XX == 0
        df = df.with_columns(
            pl.when(pl.col("first_obs_by_gp_XX") == 0)
            .then(None)
            .otherwise(pl.col("U_Gg_global_XX"))
            .alias("U_Gg_global_XX")
        )

        # delta_XX = sum(U_Gg_global_XX) / G_XX
        total_U = df.select(pl.col("U_Gg_global_XX").sum()).to_series()[0]
        delta_XX = total_U / G_XX if G_XX != 0 else np.nan

        # (optional: keep column delta_XX as in original)
        df = df.with_columns(pl.lit(delta_XX).alias("delta_XX"))

        dict_glob["Av_tot_effect"] = delta_XX
        mat_res_XX[l_XX, 0] = delta_XX

        # counts of switchers from dict_glob
        N_switchers_effect_XX = sum(
            dict_glob.get(f"N_switchers_effect_{i}_XX", 0) for i in range(1, l_XX + 1)
        )
        N_switchers_effect_dwXX = sum(
            dict_glob.get(f"N_switchers_effect_{i}_dwXX", 0) for i in range(1, l_XX + 1)
        )
        mat_res_XX[l_XX, 7] = N_switchers_effect_XX
        mat_res_XX[l_XX, 5] = N_switchers_effect_dwXX
        mat_res_XX[l_XX, 8] = 0

        # ------------------------------------
        # count_global_XX (row-wise max)
        # ------------------------------------
        # Collect all count{i}_global_XX columns that exist
        count_exprs = [
            pl.col(f"count{i}_global_XX")
            for i in range(1, l_XX + 1)
            if f"count{i}_global_XX" in df.columns
        ]

        if count_exprs:
            # row-wise max across all count{i}_global_XX and 0, ignoring nulls
            df = df.with_columns(
                pl.max_horizontal([pl.lit(0)] + count_exprs).alias("count_global_XX")
            )
        else:
            # if no count columns exist, just set to 0
            df = df.with_columns(pl.lit(0).alias("count_global_XX"))

        # count_global_dwXX: indicator of count_global_XX > 0
        df = df.with_columns(
            (
                (pl.col("count_global_XX") > 0)
            ).cast(pl.Int64).alias("count_global_dwXX")
        )

        # Aggregate counts
        N_effect_XX = df.select(pl.col("count_global_XX").sum()).to_series()[0]
        N_effect_dwXX = df.select(pl.col("count_global_dwXX").sum()).to_series()[0]

        mat_res_XX[l_XX, 6] = int(N_effect_XX)
        mat_res_XX[l_XX, 4] = int(N_effect_dwXX)

    # --------------------------
    # ✅ Placebos (versión Polars)
    # --------------------------
    if l_placebo_XX != 0:
        for i in range(1, int(l_placebo_XX) + 1):
            # Retrieve placebo counts
            N1_pl_new = dict_glob.get(f"N1_placebo_{i}_XX_new", 0)
            N0_pl_new = dict_glob.get(f"N0_placebo_{i}_XX_new", 0)

            # Weighted average of U_Gg across in/out switchers
            df = df.with_columns(
                (
                    (N1_pl_new / (N1_pl_new + N0_pl_new)) * pl.col(f"U_Gg_pl_{i}_plus_XX") +
                    (N0_pl_new / (N1_pl_new + N0_pl_new)) * pl.col(f"U_Gg_pl_{i}_minus_XX")
                ).alias(f"U_Gg_pl_{i}_global_XX")
            )

            # Drop invalid first obs (set to null)
            df = df.with_columns(
                pl.when(pl.col("first_obs_by_gp_XX") == 0)
                .then(pl.lit(None))
                .otherwise(pl.col(f"U_Gg_pl_{i}_global_XX"))
                .alias(f"U_Gg_pl_{i}_global_XX")
            )

            # ----- Counts -----
            col_plus   = f"count{i}_pl_plus_XX"
            col_minus  = f"count{i}_pl_minus_XX"
            col_global = f"count{i}_pl_global_XX"

            # Take max between plus and minus
            df = df.with_columns(
                pl.max_horizontal(pl.col(col_plus), pl.col(col_minus)).alias(col_global)
            )

            # Binary indicator (count > 0)
            df = df.with_columns(
                ((pl.col(col_global).is_not_null()) & (pl.col(col_global) > 0))
                .cast(pl.Int64)
                .alias(f"count{i}_pl_global_dwXX")
            )

            # ----- Normalized delta -----
            if normalized:
                uax = dict_glob.get(f"delta_D_pl_{i}_out_XX", 0)
                # print(f"N values {N1_pl_new} + {N0_pl_new}, {uax}")

                dict_glob[f"delta_D_pl_{i}_global_XX"] = (
                    (N1_pl_new / (N1_pl_new + N0_pl_new)) * dict_glob.get(f"delta_D_pl_{i}_in_XX", 0)
                    + (N0_pl_new / (N1_pl_new + N0_pl_new)) * dict_glob.get(f"delta_D_pl_{i}_out_XX", 0)
                )

            # ----- DID placebo -----
            df = df.with_columns(
                (pl.col(f"U_Gg_pl_{i}_global_XX").sum() / G_XX).alias(f"DID_placebo_{i}_XX")
            )

            if normalized:
                df = df.with_columns(
                    (pl.col(f"DID_placebo_{i}_XX") / dict_glob[f"delta_D_pl_{i}_global_XX"])
                    .alias(f"DID_placebo_{i}_XX")
                )

            dict_glob[f"DID_placebo_{i}_XX"] = float(
                df.select(pl.col(f"DID_placebo_{i}_XX")).mean()[0, 0]
            )

            # Missing check (if no valid switchers)
            if (
                (switchers == "" and N1_pl_new == 0 and N0_pl_new == 0)
                or (switchers == "out" and N0_pl_new == 0)
                or (switchers == "in" and N1_pl_new == 0)
            ):
                dict_glob[f"DID_placebo_{i}_XX"] = np.nan

            # Store DID in result matrix
            mat_res_XX[l_XX + i, 0] = dict_glob[f"DID_placebo_{i}_XX"]

            # ----- Number of switchers -----
            dict_glob[f"N_switchers_placebo_{i}_XX"] = N1_pl_new + N0_pl_new
            dict_glob[f"N_switchers_placebo_{i}_dwXX"] = (
                dict_glob.get(f"N1_dw_placebo_{i}_XX", 0)
                + dict_glob.get(f"N0_dw_placebo_{i}_XX", 0)
            )

            # Store counts into results matrix
            mat_res_XX[l_XX + i, 7] = dict_glob[f"N_switchers_placebo_{i}_XX"]
            mat_res_XX[l_XX + i, 5] = dict_glob[f"N_switchers_placebo_{i}_dwXX"]
            mat_res_XX[l_XX + i, 8] = -i

            # Compute total observations
            df_count = df.select(pl.col(col_global)).to_series()
            dict_glob[f"N_placebo_{i}_XX"] = float(df_count.sum())

            mat_res_XX[l_XX + i, 6] = dict_glob[f"N_placebo_{i}_XX"]
            mat_res_XX[l_XX + i, 4] = int(
                df.select(pl.col(f"count{i}_pl_global_dwXX")).to_series().sum()
            )

            # Warn if no valid estimations
            if dict_glob[f"N_switchers_placebo_{i}_XX"] == 0 or dict_glob[f"N_placebo_{i}_XX"] == 0:
                warnings.warn(
                    f"Placebo {i} cannot be estimated. No switcher or control for this placebo.",
                    UserWarning
                )


    # Patch significance level
    ci_level = ci_level / 100
    z_level = norm.ppf(ci_level + (1 - ci_level) / 2)

    # If df is still pandas, do this once *before*:
    # df = pl.from_pandas(df)

    for i in range(1, int(l_XX) + 1):
        N1_new = dict_glob.get(f"N1_{i}_XX_new", 0)
        N0_new = dict_glob.get(f"N0_{i}_XX_new", 0)

        valid_case = (
            (switchers == "" and (N1_new != 0 or N0_new != 0))
            or (switchers == "out" and N0_new != 0)
            or (switchers == "in" and N1_new != 0)
        )

        if not valid_case:
            continue

        # Weights for "in" and "out"
        denom = N1_new + N0_new
        weight_in = N1_new / denom if denom != 0 else 0
        weight_out = N0_new / denom if denom != 0 else 0

        col_glob = f"U_Gg_var_glob_{i}_XX"

        # 1) Aggregate U_Gg_var for switchers in and out
        df = df.with_columns(
            (
                pl.col(f"U_Gg_var_{i}_in_XX") * weight_in
                + pl.col(f"U_Gg_var_{i}_out_XX") * weight_out
            ).alias(col_glob)
        )

        # 2) Variance aggregation with or without clustering
        if cluster is None:
            # Without clustering
            col_eff_sq = f"U_Gg_var_glob_eff{i}_sqrd_XX"

            df = df.with_columns(
                (pl.col(col_glob) ** 2 * pl.col("first_obs_by_gp_XX")).alias(col_eff_sq)
            )

            if G_XX is not None:
                sum_for_var_i = (
                    df.select(pl.col(col_eff_sq).sum()).item() / (G_XX ** 2)
                )
            else:
                sum_for_var_i = np.nan

            dict_glob[f"sum_for_var_{i}_XX"] = sum_for_var_i

        else:
            # With clustering
            col_base = col_glob
            col_clust = f"clust_U_Gg_var_glob_{i}_XX"
            col_clust_sq = f"clust_U_Gg_var_glob_{i}_2_XX"

            # 1. Multiply U_Gg_var_glob by first_obs_by_gp_XX
            df = df.with_columns(
                (pl.col(col_base) * pl.col("first_obs_by_gp_XX")).alias(col_base)
            )

            # 2. Sum within cluster: total(U_Gg_var_glob_i_XX) over cluster_XX
            df = df.with_columns(
                pl.col(col_base).sum().over("cluster_XX").alias(col_clust)
            )

            # 3. Average of square * first_obs_by_clust_XX
            df = df.with_columns(
                ((pl.col(col_clust) ** 2) * pl.col("first_obs_by_clust_XX")).alias(
                    col_clust_sq
                )
            )

            # 4. Scalar sum_for_var = sum / G_XX^2
            if G_XX is not None:
                sum_for_var_i = (
                    df.select(pl.col(col_clust_sq).sum()).item() / (G_XX ** 2)
                )
            else:
                sum_for_var_i = np.nan

            dict_glob[f"sum_for_var_{i}_XX"] = sum_for_var_i

            # 5. Replace U_Gg_var_glob with cluster total
            df = df.with_columns(
                pl.col(col_clust).alias(col_base)
            )

        # 3) Compute SE
        se_i = np.sqrt(dict_glob[f"sum_for_var_{i}_XX"])
        dict_glob[f"se_{i}_XX"] = se_i

        # 4) Normalize SE if requested
        if normalized:
            denom_se = dict_glob.get(f"delta_D_{i}_global_XX", 1)
            se_i = se_i / denom_se
            dict_glob[f"se_{i}_XX"] = se_i

        # 5) Store results
        mat_res_XX[i - 1, 1] = se_i
        dict_glob[f"se_effect_{i}"] = se_i

        lb_ci = dict_glob[f"DID_{i}_XX"] - z_level * se_i
        ub_ci = dict_glob[f"DID_{i}_XX"] + z_level * se_i

        dict_glob[f"LB_CI_{i}_XX"] = lb_ci
        dict_glob[f"UB_CI_{i}_XX"] = ub_ci

        mat_res_XX[i - 1, 2] = lb_ci
        mat_res_XX[i - 1, 3] = ub_ci


    # ----------------------------------------
    # Variances of placebo estimators (Polars)
    # ----------------------------------------
    if l_placebo_XX != 0:
        for i in range(1, int(l_placebo_XX) + 1):
            N1_pl_new = dict_glob.get(f"N1_placebo_{i}_XX_new", 0)
            N0_pl_new = dict_glob.get(f"N0_placebo_{i}_XX_new", 0)

            valid_case = (
                (switchers == "" and (N1_pl_new != 0 or N0_pl_new != 0))
                or (switchers == "out" and N0_pl_new != 0)
                or (switchers == "in" and N1_pl_new != 0)
            )

            if not valid_case:
                continue

            denom = N1_pl_new + N0_pl_new
            weight_in = N1_pl_new / denom if denom != 0 else 0
            weight_out = N0_pl_new / denom if denom != 0 else 0

            col_glob = f"U_Gg_var_glob_pl_{i}_XX"

            # 1) Aggregate U_Gg_var for placebo
            df = df.with_columns(
                (
                    pl.col(f"U_Gg_var_pl_{i}_in_XX") * weight_in
                    + pl.col(f"U_Gg_var_pl_{i}_out_XX") * weight_out
                ).alias(col_glob)
            )

            # 2) Variance aggregation with or without clustering
            if cluster is None:
                col_glob2 = f"U_Gg_var_glob_pl_{i}_2_XX"
                df = df.with_columns(
                    ((pl.col(col_glob) ** 2) * pl.col("first_obs_by_gp_XX")).alias(
                        col_glob2
                    )
                )

                sum_for_var_pl = (
                    df.select(pl.col(col_glob2).sum())
                    .to_series()[0] / (G_XX ** 2)
                )
                dict_glob[f"sum_for_var_placebo_{i}_XX"] = sum_for_var_pl

            else:
                col_U = col_glob
                col_clust = f"clust_U_Gg_var_glob_pl_{i}_XX"
                col_clust_sq = f"clust_U_Gg_var_glob_pl_{i}_2_XX"

                # 1. Multiply by first_obs_by_gp_XX
                df = df.with_columns(
                    (pl.col(col_U) * pl.col("first_obs_by_gp_XX")).alias(col_U)
                )

                # 2. Sum within cluster
                df = df.with_columns(
                    pl.col(col_U).sum().over("cluster_XX").alias(col_clust)
                )

                # 3. Square cluster total * first_obs_by_clust_XX
                df = df.with_columns(
                    ((pl.col(col_clust) ** 2) * pl.col("first_obs_by_clust_XX")).alias(
                        col_clust_sq
                    )
                )

                sum_for_var_pl = (
                    df.select(pl.col(col_clust_sq).sum())
                    .to_series()[0] / (G_XX ** 2)
                )
                dict_glob[f"sum_for_var_placebo_{i}_XX"] = sum_for_var_pl

                # 4. Replace U_ column by cluster total
                df = df.with_columns(pl.col(col_clust).alias(col_U))

            # 3) SE
            se_pl = np.sqrt(dict_glob[f"sum_for_var_placebo_{i}_XX"])
            dict_glob[f"se_placebo_{i}_XX"] = se_pl

            # 4) Normalize SE if requested
            if normalized:
                denom_se = dict_glob.get(f"delta_D_pl_{i}_global_XX", 1)
                se_pl = se_pl / denom_se
                dict_glob[f"se_placebo_{i}_XX"] = se_pl

            # 5) Store results
            mat_res_XX[l_XX + i, 1] = se_pl
            dict_glob[f"se_placebo_{i}"] = se_pl

            lb_ci = dict_glob[f"DID_placebo_{i}_XX"] - z_level * se_pl
            ub_ci = dict_glob[f"DID_placebo_{i}_XX"] + z_level * se_pl

            dict_glob[f"LB_CI_placebo_{i}_XX"] = lb_ci
            dict_glob[f"UB_CI_placebo_{i}_XX"] = ub_ci

            mat_res_XX[l_XX + i, 2] = lb_ci
            mat_res_XX[l_XX + i, 3] = ub_ci


    # ----------------------------------------
    # Variance of average total effect
    # ----------------------------------------
    if not trends_lin:
        valid_case = (
            (switchers == "" and (dict_glob["sum_N1_l_XX"] != 0 or dict_glob["sum_N0_l_XX"] != 0))
            or (switchers == "out" and dict_glob["sum_N0_l_XX"] != 0)
            or (switchers == "in" and dict_glob["sum_N1_l_XX"] != 0)
        )

        if valid_case:
            # U_Gg_var_global_XX = weighted mix of plus / minus
            df = df.with_columns(
                (
                    w_plus_XX * pl.col("U_Gg_var_plus_XX")
                    + (1 - w_plus_XX) * pl.col("U_Gg_var_minus_XX")
                ).alias("U_Gg_var_global_XX")
            )

            if cluster is None:
                # No clustering
                df = df.with_columns(
                    (pl.col("U_Gg_var_global_XX") ** 2 * pl.col("first_obs_by_gp_XX")).alias(
                        "U_Gg_var_global_2_XX"
                    )
                )

                sum_for_var_XX = (
                    df.select(pl.col("U_Gg_var_global_2_XX").sum())
                    .to_series()[0] / (G_XX ** 2)
                )
            else:
                # With clustering
                df = df.with_columns(
                    (pl.col("U_Gg_var_global_XX") * pl.col("first_obs_by_gp_XX")).alias(
                        "U_Gg_var_global_XX"
                    )
                )

                # Sum over clusters
                df = df.with_columns(
                    pl.col("U_Gg_var_global_XX").sum().over("cluster_XX").alias(
                        "clust_U_Gg_var_global_XX"
                    )
                )

                # Square cluster total * first_obs_by_clust_XX
                df = df.with_columns(
                    (
                        pl.col("clust_U_Gg_var_global_XX") ** 2
                        * pl.col("first_obs_by_clust_XX")
                    ).alias("clust_U_Gg_var_global_XX")
                )

                sum_for_var_XX = (
                    df.select(pl.col("clust_U_Gg_var_global_XX").sum())
                    .to_series()[0] / (G_XX ** 2)
                )

            se_XX = np.sqrt(sum_for_var_XX)
            mat_res_XX[l_XX, 1] = se_XX
            dict_glob["se_avg_total_effect"] = se_XX

            LB_CI_XX = delta_XX - z_level * se_XX
            UB_CI_XX = delta_XX + z_level * se_XX

            mat_res_XX[l_XX, 2] = LB_CI_XX
            mat_res_XX[l_XX, 3] = UB_CI_XX


    # ----------------------------------------
    # Average number of cumulated effects
    # ----------------------------------------
    for i in range(1, int(l_XX) + 1):
        col = f"delta_D_g_{i}_XX"
        if col in df.columns:
            df = df.drop(col)

    # M_g_XX
    df = df.with_columns(
        pl.when(
            pl.lit(l_XX) <= (pl.col("T_g_XX") - pl.col("F_g_XX") + 1)
        )
        .then(pl.lit(l_XX))
        .otherwise(pl.col("T_g_XX") - pl.col("F_g_XX") + 1)
        .alias("M_g_XX")
    )

    import polars as pl

    # Build delta_D_g_XX
    df = df.with_columns(
        pl.lit(0.0).alias("delta_D_g_XX")
    )

    for j in range(1, l_XX + 1):
        temp_col = "delta_D_g_XX_temp"
        plus_col = f"delta_D_g_{j}_plus_XX"
        minus_col = f"delta_D_g_{j}_minus_XX"

        # delta_D_g_XX_temp = delta_D_g_j_plus if != 0, else delta_D_g_j_minus
        df = df.with_columns(
            pl.when(pl.col(plus_col) != 0)
            .then(pl.col(plus_col))
            .otherwise(pl.col(minus_col))
            .alias(temp_col)
        )

        # delta_D_g_XX_temp = NaN where == 0
        df = df.with_columns(
            pl.when(pl.col(temp_col) == 0)
            .then(None)        # Polars null ≈ NaN in this context
            .otherwise(pl.col(temp_col))
            .alias(temp_col)
        )

        # delta_D_g_XX = delta_D_g_XX + temp where switcher_tag_XX == j, else unchanged
        df = df.with_columns(
            pl.when(pl.col("switcher_tag_XX") == j)
            .then(pl.col("delta_D_g_XX") + pl.col(temp_col))
            .otherwise(pl.col("delta_D_g_XX"))
            .alias("delta_D_g_XX")
        )

    # drop the temp column if you don't need it
    if "delta_D_g_XX_temp" in df.columns:
        df = df.drop("delta_D_g_XX_temp")

    # delta_D_g_num_XX = delta_D_g_XX * (M_g_XX - (switcher_tag_XX - 1))
    df = df.with_columns(
        (
            pl.col("delta_D_g_XX")
            * (pl.col("M_g_XX") - (pl.col("switcher_tag_XX") - 1))
        ).alias("delta_D_g_num_XX")
    )


    delta_D_num_total = (
        df.select(
            pl.col("delta_D_g_num_XX")
            .fill_nan(0)   # turn NaNs into nulls
            .sum()
        )
        .to_series()[0]
    )
    delta_D_denom_total = (
        df.select(
            pl.col("delta_D_g_XX")
            .fill_nan(0)   # turn NaNs into nulls
            .sum()
        )
        .to_series()[0]
    )
    delta_D_avg_total = np.float64(delta_D_num_total) / np.float64(delta_D_denom_total)



    # ----------------------------------------
    # Cluster adjustment
    # ----------------------------------------
    if cluster is not None:
        df = df.with_columns(
            pl.col("first_obs_by_clust_XX").alias("first_obs_by_gp_XX")
        )

        

    # --------------------------------------------------------------------------------
    ###### Performing a test to see whether all effects are jointly equal to 0
    # --------------------------------------------------------------------------------

    # Initialize
    all_Ns_not_zero = np.nan
    all_delta_not_zero = np.nan
    p_jointeffects = None

    # Test can only be run when at least two effects requested
    if l_XX != 0 and l_XX > 1:
        all_Ns_not_zero = 0
        all_delta_not_zero = 0

        # Count number of estimable effects
        for i in range(1, l_XX + 1):
            N1_new = dict_glob.get(f"N1_{i}_XX_new", 0)
            N0_new = dict_glob.get(f"N0_{i}_XX_new", 0)

            if (
                (switchers == "" and (N1_new != 0 or N0_new != 0))
                or (switchers == "out" and N0_new != 0)
                or (switchers == "in" and N1_new != 0)
            ):
                all_Ns_not_zero += 1

            if normalized:
                delta_val = dict_glob.get(f"delta_D_{i}_global_XX", np.nan)
                if delta_val != 0 and not np.isnan(delta_val):
                    all_delta_not_zero += 1

        # Test feasible only if all requested effects were computed
        feasible = (
            (all_Ns_not_zero == l_XX and not normalized)
            or (normalized and all_Ns_not_zero == l_XX and all_delta_not_zero == l_XX)
        )

        if feasible:
            # Collect DID estimates and variances
            didmgt_Effects = np.zeros((l_XX, 1))
            didmgt_Var_Effects = np.zeros((l_XX, l_XX))

            for i in range(1, l_XX + 1):
                didmgt_Effects[i - 1, 0] = dict_glob.get(f"DID_{i}_XX", 0)
                didmgt_Var_Effects[i - 1, i - 1] = dict_glob.get(f"se_{i}_XX", 0) ** 2

                if i < l_XX:
                    for j in range(i + 1, l_XX + 1):
                        # U_Gg_var_{i}_{j}_XX
                        if not normalized:
                            df = df.with_columns(
                                (
                                    pl.col(f"U_Gg_var_glob_{i}_XX")
                                    + pl.col(f"U_Gg_var_glob_{j}_XX")
                                ).alias(f"U_Gg_var_{i}_{j}_XX")
                            )
                        else:
                            df = df.with_columns(
                                (
                                    pl.col(f"U_Gg_var_glob_{i}_XX")
                                    / dict_glob.get(f"delta_D_{i}_global_XX", 1)
                                    + pl.col(f"U_Gg_var_glob_{j}_XX")
                                    / dict_glob.get(f"delta_D_{j}_global_XX", 1)
                                ).alias(f"U_Gg_var_{i}_{j}_XX")
                            )

                        # U_Gg_var_{i}_{j}_2_XX = (U_Gg_var_{i}_{j}_XX^2) * first_obs_by_gp_XX
                        df = df.with_columns(
                            (
                                pl.col(f"U_Gg_var_{i}_{j}_XX") ** 2
                                * pl.col("first_obs_by_gp_XX")
                            ).alias(f"U_Gg_var_{i}_{j}_2_XX")
                        )

                        # Sum, ignoring NaN (pandas skipna=True)
                        var_sum_series = df.select(
                            pl.col(f"U_Gg_var_{i}_{j}_2_XX").fill_nan(0).sum()
                        ).to_series()
                        var_sum = var_sum_series[0] / (G_XX ** 2)

                        se_i = dict_glob.get(f"se_{i}_XX", 0)
                        se_j = dict_glob.get(f"se_{j}_XX", 0)

                        cov_ij = (var_sum - se_i**2 - se_j**2) / 2
                        dict_glob[f"cov_{i}_{j}_XX"] = cov_ij

                        didmgt_Var_Effects[i - 1, j - 1] = cov_ij
                        didmgt_Var_Effects[j - 1, i - 1] = cov_ij

            # Inverse covariance matrix (pseudo-inverse if singular)
            didmgt_Var_Effects_inv = np.linalg.pinv(didmgt_Var_Effects)

            # Wald χ² statistic
            didmgt_chi2effects = (
                didmgt_Effects.T @ didmgt_Var_Effects_inv @ didmgt_Effects
            )

            # p-value
            p_jointeffects = 1 - chi2.cdf(didmgt_chi2effects[0, 0], df=l_XX)

        else:
            p_jointeffects = np.nan
            warnings.warn(
                "Some effects could not be estimated. Therefore, the test of joint nullity of the effects could not be computed.",
                UserWarning
            )



    # --------------------------------------------------------------------------------
    ###### Performing a test to see whether all placebos are jointly equal to 0
    # --------------------------------------------------------------------------------

    

    all_Ns_pl_not_zero = np.nan
    all_delta_pl_not_zero = np.nan
    p_jointplacebo = None

    # Test can only be run when at least two placebos requested
    if l_placebo_XX != 0 and l_placebo_XX > 1:
        all_Ns_pl_not_zero = 0
        all_delta_pl_not_zero = 0

        # Count number of estimable placebos
        for i in range(1, int(l_placebo_XX) + 1):
            N1_pl_new = dict_glob.get(f"N1_placebo_{i}_XX_new", 0)
            N0_pl_new = dict_glob.get(f"N0_placebo_{i}_XX_new", 0)

            if (
                (switchers == "" and (N1_pl_new != 0 or N0_pl_new != 0))
                or (switchers == "out" and N0_pl_new != 0)
                or (switchers == "in" and N1_pl_new != 0)
            ):
                all_Ns_pl_not_zero += 1

            if normalized:
                delta_val = dict_glob.get(f"delta_D_pl_{i}_global_XX", np.nan)
                if delta_val != 0 and not np.isnan(delta_val):
                    all_delta_pl_not_zero += 1

        # Test feasible only if all requested placebos were computed
        feasible = (
            (all_Ns_pl_not_zero == l_placebo_XX and not normalized)
            or (normalized and all_Ns_pl_not_zero == l_placebo_XX and all_delta_pl_not_zero == l_placebo_XX)
        )

        if feasible:
            # Collect DID placebo estimates and variances
            l_placebo_int = int(l_placebo_XX)

            didmgt_Placebo = np.zeros((l_placebo_int, 1))
            didmgt_Var_Placebo = np.zeros((l_placebo_int, l_placebo_int))

            for i in range(1, l_placebo_int + 1):
                didmgt_Placebo[i - 1, 0] = dict_glob.get(f"DID_placebo_{i}_XX", 0.0)
                se_i = dict_glob.get(f"se_placebo_{i}_XX", 0.0)
                didmgt_Var_Placebo[i - 1, i - 1] = se_i**2

                if i < l_placebo_int:
                    for j in range(i + 1, l_placebo_int + 1):

                        # --- build U_Gg_var_pl_{i}_{j}_XX ---
                        if not normalized:
                            df = df.with_columns(
                                (
                                    pl.col(f"U_Gg_var_glob_pl_{i}_XX")
                                    + pl.col(f"U_Gg_var_glob_pl_{j}_XX")
                                ).alias(f"U_Gg_var_pl_{i}_{j}_XX")
                            )
                        else:
                            df = df.with_columns(
                                (
                                    pl.col(f"U_Gg_var_glob_pl_{i}_XX")
                                    / dict_glob.get(f"delta_D_pl_{i}_global_XX", 1.0)
                                    + pl.col(f"U_Gg_var_glob_pl_{j}_XX")
                                    / dict_glob.get(f"delta_D_pl_{j}_global_XX", 1.0)
                                ).alias(f"U_Gg_var_pl_{i}_{j}_XX")
                            )

                        # --- build U_Gg_var_pl_{i}_{j}_2_XX = (U_Gg_var_pl_{i}_{j}_XX)^2 * first_obs_by_gp_XX ---
                        df = df.with_columns(
                            (
                                pl.col(f"U_Gg_var_pl_{i}_{j}_XX") ** 2
                                * pl.col("first_obs_by_gp_XX")
                            ).alias(f"U_Gg_var_pl_{i}_{j}_2_XX")
                        )

                        # sum over column (nulls ignored by default)
                        var_sum = (
                            df.select(pl.col(f"U_Gg_var_pl_{i}_{j}_2_XX").sum())
                            .item()              # returns Python scalar
                            / (G_XX ** 2)
                        )

                        se_i = dict_glob.get(f"se_placebo_{i}_XX", 0.0)
                        se_j = dict_glob.get(f"se_placebo_{j}_XX", 0.0)

                        cov_ij = (var_sum - se_i**2 - se_j**2) / 2.0
                        dict_glob[f"cov_pl_{i}_{j}_XX"] = cov_ij

                        didmgt_Var_Placebo[i - 1, j - 1] = cov_ij
                        didmgt_Var_Placebo[j - 1, i - 1] = cov_ij

            # Inverse covariance matrix (pseudo-inverse if singular)
            didmgt_Var_Placebo_inv = Ginv(didmgt_Var_Placebo)

            # Wald χ² statistic
            didmgt_chi2placebo = didmgt_Placebo.T @ didmgt_Var_Placebo_inv @ didmgt_Placebo

            # p-value
            p_jointplacebo = 1 - chi2.cdf(didmgt_chi2placebo[0, 0], df=l_placebo_int)
        else:
            p_jointplacebo = np.nan
            warnings.warn(
                "Some placebos could not be estimated. Therefore, the test of joint nullity of the placebos could not be computed.",
                UserWarning
            )

    het_res = pd.DataFrame()

    if predict_het is not None and len(predict_het_good) > 0:
        # Define which effects to calculate
        # Handle 'all', -1, or list of effect numbers
        if het_effects == 'all' or het_effects == -1 or (isinstance(het_effects, (list, tuple)) and -1 in het_effects):
            het_effects = list(range(1, l_XX + 1))
        elif isinstance(het_effects, (int, float)):
            het_effects = [int(het_effects)]
        all_effects_XX = [i for i in range(1, l_XX + 1) if i in het_effects]

        if any(np.isnan(all_effects_XX)):
            raise ValueError(
                "Error in predict_het second argument: please specify only numbers ≤ number of effects requested"
            )

        # Preliminaries: Yg, Fg-1 (using Polars syntax)
        df = df.with_columns(
            pl.when(pl.col("time_XX") == pl.col("F_g_XX") - 1)
            .then(pl.col("outcome_non_diff_XX"))
            .otherwise(None)
            .alias("Yg_Fg_min1_XX_temp")
        )
        df = df.with_columns(
            pl.col("Yg_Fg_min1_XX_temp").mean().over("group_XX").alias("Yg_Fg_min1_XX")
        )
        df = df.with_columns(
            pl.col("Yg_Fg_min1_XX").is_not_null().alias("feasible_het_XX")
        )
        df = df.drop("Yg_Fg_min1_XX_temp")

        if trends_lin:
            df = df.with_columns(
                pl.when(pl.col("time_XX") == pl.col("F_g_XX") - 2)
                .then(pl.col("outcome_non_diff_XX"))
                .otherwise(None)
                .alias("Yg_Fg_min2_XX_temp")
            )
            df = df.with_columns(
                pl.col("Yg_Fg_min2_XX_temp").mean().over("group_XX").alias("Yg_Fg_min2_XX")
            )
            df = df.with_columns(
                (pl.col("feasible_het_XX") & pl.col("Yg_Fg_min2_XX").is_not_null()).alias("feasible_het_XX")
            )
            df = df.drop("Yg_Fg_min2_XX_temp")

        # Order and group index
        df = df.sort(["group_XX", "time_XX"])
        df = df.with_columns(
            (pl.col("time_XX").cum_count().over("group_XX")).alias("gr_id")
        )

        lhyp = [f"{v}=0" for v in predict_het_good]

        # Loop over requested effects
        for i in all_effects_XX:
            # Yg, Fg-1 + i
            df = df.with_columns(
                pl.when(pl.col("time_XX") == pl.col("F_g_XX") - 1 + i)
                .then(pl.col("outcome_non_diff_XX"))
                .otherwise(None)
                .alias(f"Yg_Fg_{i}_XX_temp")
            )
            df = df.with_columns(
                pl.col(f"Yg_Fg_{i}_XX_temp").mean().over("group_XX").alias(f"Yg_Fg_{i}_XX")
            )
            df = df.drop(f"Yg_Fg_{i}_XX_temp")

            # Difference calculation
            df = df.with_columns(
                (pl.col(f"Yg_Fg_{i}_XX") - pl.col("Yg_Fg_min1_XX")).alias("diff_het_XX")
            )
            if trends_lin:
                df = df.with_columns(
                    (pl.col("diff_het_XX") - i * (pl.col("Yg_Fg_min1_XX") - pl.col("Yg_Fg_min2_XX"))).alias("diff_het_XX")
                )

            # Interaction term
            df = df.with_columns(
                pl.when(pl.col("gr_id") == 1)
                .then(pl.col("S_g_het_XX") * pl.col("diff_het_XX"))
                .otherwise(None)
                .alias(f"prod_het_{i}_XX")
            )

            # Sample restriction - filter and convert to pandas for regression
            het_sample_pl = df.filter(
                (pl.col("F_g_XX") - 1 + i <= pl.col("T_g_XX")) & pl.col("feasible_het_XX")
            )
            het_sample = het_sample_pl.to_pandas().reset_index(drop=True)

            # Regression formula
            het_reg = f"prod_het_{i}_XX ~ {' + '.join(predict_het_good)}"

            # Add categorical dummies
            cat_vars = ["F_g_XX", "d_sq_XX", "S_g_XX"]
            if trends_nonparam is not None:
                cat_vars.extend(trends_nonparam if isinstance(trends_nonparam, list) else [trends_nonparam])

            for v in cat_vars:
                if v in het_sample.columns and het_sample[v].nunique() > 1:
                    het_reg += f" + C({v})"

            # Run regression with robust SE
            # HC2 by default, or HC2 with cluster and dfadjust if predict_het_hc2bm
            if predict_het_hc2bm:
                # Bell-McCaffrey degrees of freedom adjustment with clustering
                cluster_var = cluster if cluster else "group_XX"
                if cluster_var in het_sample.columns:
                    # Drop rows with missing values in regression variables to avoid mismatch
                    # between cluster groups and statsmodels internal filtering
                    reg_vars = [f"prod_het_{i}_XX", "weight_XX", cluster_var] + predict_het_good
                    reg_vars = [v for v in reg_vars if v in het_sample.columns]
                    het_sample_clean = het_sample.dropna(subset=reg_vars).reset_index(drop=True)

                    # Use cluster-robust with small sample adjustment
                    cluster_groups = het_sample_clean[cluster_var].values
                    model = smf.wls(het_reg, data=het_sample_clean, weights=het_sample_clean["weight_XX"].values).fit(
                        cov_type="cluster",
                        cov_kwds={"groups": cluster_groups, "use_correction": True}
                    )
                else:
                    model = smf.wls(het_reg, data=het_sample, weights=het_sample["weight_XX"].values).fit(
                        cov_type="HC2"
                    )
            else:
                model = smf.wls(het_reg, data=het_sample, weights=het_sample["weight_XX"].values).fit(
                    cov_type="HC2"
                )

            # Extract results
            coefs = model.params[predict_het_good]
            ses = model.bse[predict_het_good]
            ts = model.tvalues[predict_het_good]

            t_stat = student_t.ppf(0.975, model.df_resid)
            lb = coefs - t_stat * ses
            ub = coefs + t_stat * ses

            f_test = model.f_test(lhyp)
            f_stat = f_test.pvalue

            # Append to het_res
            het_res = pd.concat(
                [
                    het_res,
                    pd.DataFrame(
                        {
                            "effect": i,
                            "covariate": predict_het_good,
                            "Estimate": coefs.values,
                            "SE": ses.values,
                            "t": ts.values,
                            "LB": lb.values,
                            "UB": ub.values,
                            "N": [int(model.nobs)] * len(predict_het_good),
                            "pF": [f_stat] * len(predict_het_good),
                        }
                    ),
                ],
                ignore_index=True,
            )

        het_res = het_res.sort_values(["covariate", "effect"])

    # ----------------------------
    # Test that all DID_l effects are equal
    # ----------------------------
    if effects_equal and l_XX > 1:
        all_Ns_not_zero = 0
        for i in range(1, l_XX + 1):
            N1_new = dict_glob.get(f"N1_{i}_XX_new", 0)
            N0_new = dict_glob.get(f"N0_{i}_XX_new", 0)
            if (
                (switchers == "" and (N1_new != 0 or N0_new != 0))
                or (switchers == "out" and N0_new != 0)
                or (switchers == "in" and N1_new != 0)
            ):
                all_Ns_not_zero += 1

        if all_Ns_not_zero == l_XX:
            didmgt_Effects = mat_res_XX[:l_XX, 0]
            didmgt_Var_Effects = np.zeros((l_XX, l_XX))
            didmgt_identity = np.zeros((l_XX - 1, l_XX))

            for i in range(1, l_XX + 1):
                N1_new = dict_glob.get(f"N1_{i}_XX_new", 0)
                N0_new = dict_glob.get(f"N0_{i}_XX_new", 0)
                if (
                    (switchers == "" and (N1_new != 0 or N0_new != 0))
                    or (switchers == "out" and N0_new != 0)
                    or (switchers == "in" and N1_new != 0)
                ):
                    didmgt_Var_Effects[i - 1, i - 1] = dict_glob.get(f"se_{i}_XX", 0) ** 2
                    if i < l_XX:
                        didmgt_identity[i - 1, i - 1] = 1

                    if i < l_XX:
                        for j in range(i + 1, l_XX + 1):
                            col_U = f"U_Gg_var_{i}_{j}_XX"
                            col_U2 = f"U_Gg_var_{i}_{j}_2_XX"

                            # 1. Create U_Gg_var_{i}_{j}_XX
                            if not normalized:
                                expr_U = (
                                    pl.col(f"U_Gg_var_glob_{i}_XX")
                                    + pl.col(f"U_Gg_var_glob_{j}_XX")
                                ).alias(col_U)
                            else:
                                expr_U = (
                                    pl.col(f"U_Gg_var_glob_{i}_XX") / dict_glob.get(f"delta_D_{i}_global_XX", 1)
                                    + pl.col(f"U_Gg_var_glob_{j}_XX") / dict_glob.get(f"delta_D_{j}_global_XX", 1)
                                ).alias(col_U)

                            # 2. Create squared/weighted version U_Gg_var_{i}_{j}_2_XX
                            expr_U2 = (
                                (pl.col(col_U) ** 2) * pl.col("first_obs_by_gp_XX")
                            ).alias(col_U2)

                            # Add both columns to df
                            df = df.with_columns([expr_U, expr_U2])

                            # 3. Compute var_sum
                            var_sum = (
                                df.select(
                                    (pl.col(col_U2).sum()) / (G_XX ** 2)
                                )
                                .to_series()[0]
                            )

                            # 4. cov_ij and store in dict_glob and matrix
                            se_i = dict_glob.get(f"se_{i}_XX", 0)
                            se_j = dict_glob.get(f"se_{j}_XX", 0)

                            cov_ij = (var_sum - se_i**2 - se_j**2) / 2.0
                            dict_glob[f"cov_{i}_{j}_XX"] = cov_ij

                            didmgt_Var_Effects[i - 1, j - 1] = cov_ij
                            didmgt_Var_Effects[j - 1, i - 1] = cov_ij

            # Demeaned effects: test equality
            didmgt_D = didmgt_identity - np.full((l_XX - 1, l_XX), 1 / l_XX)
            didmgt_test_effects = didmgt_D @ didmgt_Effects
            didmgt_test_var = didmgt_D @ didmgt_Var_Effects @ didmgt_D.T
            # enforce symmetry
            didmgt_test_var = (didmgt_test_var + didmgt_test_var.T) / 2

            # Wald χ² statistic
            quad_form = didmgt_test_effects.T @ np.linalg.pinv(didmgt_test_var) @ didmgt_test_effects
            # Robustly get a scalar even if it's 0-dim, 1x1, or shape (1,)
            didmgt_chi2_equal_ef = np.asarray(quad_form).ravel()[0]

            p_equality_effects = 1 - chi2.cdf(didmgt_chi2_equal_ef, df=l_XX - 1)
            dict_glob["p_equality_effects"] = p_equality_effects

        else:
            warnings.warn(
                "Some effects could not be estimated. Therefore, the test of equality of effects could not be computed.",
                UserWarning
            )

    # assume df is a pandas DataFrame
    # assume l_XX, l_placebo_XX, normalized, G_XX, mat_res_XX are already defined

    # 1. Total length
    l_tot_XX = l_XX + l_placebo_XX

    # 2. Initialize covariance matrix with NaNs
    didmgt_vcov = np.full((int(l_tot_XX), int(l_tot_XX)), np.nan)

    # 3. Build row/col names
    mat_names = [
        f"Effect_{i}" if i <= l_XX else f"Placebo_{i - l_XX}"
        for i in range(1, int(l_tot_XX) + 1)
    ]
    # print(mat_names)

    # Optionally wrap in DataFrame for labeled covariance matrix
    didmgt_vcov = pd.DataFrame(didmgt_vcov, index=mat_names, columns=mat_names)

    # 4. Loop for main effects
    for i in range(1, l_XX + 1):
        col_glob = f"U_Gg_var_glob_{i}_XX"
        col_comb = f"U_Gg_var_comb_{i}_XX"

        if not normalized:
            if col_glob in df.columns:
                df = df.with_columns(
                    pl.col(col_glob).alias(col_comb)
                )
            else:
                df = df.with_columns(
                    pl.lit(None).alias(col_comb)
                )
        else:
            delta_name = f"delta_D_{i}_global_XX"
            if col_glob in df.columns:
                df = df.with_columns(
                    (pl.col(col_glob) / dict_glob[delta_name]).alias(col_comb)
                )
            else:
                df = df.with_columns(
                    pl.lit(None).alias(col_comb)
                )

    # 5. Loop for placebos
    if l_placebo_XX != 0:
        for i in range(1, int(l_placebo_XX) + 1):
            col_glob_pl = f"U_Gg_var_glob_pl_{i}_XX"
            col_comb = f"U_Gg_var_comb_{l_XX + i}_XX"

            if not normalized:
                if col_glob_pl in df.columns:
                    df = df.with_columns(
                        pl.col(col_glob_pl).alias(col_comb)
                    )
                else:
                    df = df.with_columns(
                        pl.lit(None).alias(col_comb)
                    )
            else:
                delta_name = f"delta_D_pl_{i}_global_XX"
                if col_glob_pl in df.columns:
                    df = df.with_columns(
                        (pl.col(col_glob_pl) / dict_glob[delta_name]).alias(col_comb)
                    )
                else:
                    df = df.with_columns(
                        pl.lit(None).alias(col_comb)
                    )

    # 6. Fill the covariance matrix
    for i in range(1, int(l_tot_XX) + 1):
        # this line stays as-is (didmgt_vcov, mat_res_XX are pandas/numpy)
        didmgt_vcov.iloc[i - 1, i - 1] = mat_res_XX[i + (i > l_XX) - 1, 1] ** 2

        j = 1
        while j < i:
            col_i = f"U_Gg_var_comb_{i}_XX"
            col_j = f"U_Gg_var_comb_{j}_XX"
            col_temp = f"U_Gg_var_comb_{i}_{j}_2_XX"

            # polars version of:
            # df[col_temp] = (df[col_i] + df[col_j]) ** 2 * df["first_obs_by_gp_XX"]
            df = df.with_columns(
                (
                    (pl.col(col_i) + pl.col(col_j)) ** 2
                    * pl.col("first_obs_by_gp_XX")
                ).alias(col_temp)
            )

            # polars version of:
            # var_temp = df[col_temp].sum(skipna=True) / (G_XX ** 2)
            var_sum = df.select(pl.col(col_temp).sum()).item()   # skip_nulls=True by default
            var_temp = var_sum / (G_XX ** 2)

            didmgt_vcov.iloc[i - 1, j - 1] = didmgt_vcov.iloc[j - 1, i - 1] = (
                var_temp
                - mat_res_XX[i + (i > l_XX) - 1, 1] ** 2
                - mat_res_XX[j + (j > l_XX) - 1, 1] ** 2
            ) / 2

            # polars version of:
            # df.drop(columns=[col_temp], inplace=True)
            df = df.drop(col_temp)

            j += 1


    # -------------------------------------
    # Format results matrix
    # -------------------------------------

    rownames_arr = np.array(rownames)
    colnames_arr = [
        "Estimate", "SE", "LB CI", "UB CI",
        "N", "Switchers", "N.w", "Switchers.w", "Time"
    ]

    mat_res_df = pd.DataFrame(mat_res_XX)
    mat_res_df.columns = colnames_arr 

    # Save results if requested
    if save_results is not None:
        mat_res_df.to_csv(save_results, index=True)

    # -------------------------------------
    # Separate Effect matrix and ATE matrix
    # -------------------------------------
    Effect_mat = mat_res_df.iloc[:l_XX, :-1].copy()
    ATE_mat = mat_res_df.iloc[[l_XX], :-1].copy()
    ATE_mat.index = ["Average_Total_Effect"]
    Effect_mat.index = mat_names[:l_XX]


    # -------------------------------------
    # Assemble did_multiplegt_dyn
    # -------------------------------------
    out_names = [
        "N_Effects", "N_Placebos", "Effects", "ATE",
        "delta_D_avg_total", "max_pl", "max_pl_gap"
    ]

    did_multiplegt_dyn = [
        l_XX,
        int(l_placebo_XX),
        Effect_mat,
        ATE_mat,
        delta_D_avg_total,
        max_pl_XX,
        max_pl_gap_XX,
    ]

    if p_jointeffects is not None:
        did_multiplegt_dyn.append(p_jointeffects)
        out_names.append("p_jointeffects")

    if effects_equal:
        did_multiplegt_dyn.append(p_equality_effects)
        out_names.append("p_equality_effects")

    if placebo != 0:
        Placebo_mat = mat_res_df.iloc[(l_XX + 1):, :-1].copy()
        Placebo_mat.index = mat_names[l_XX:]
        did_multiplegt_dyn.append(Placebo_mat)
        out_names.append("Placebos")

        if placebo > 1 and l_placebo_XX > 1:
            did_multiplegt_dyn.append(p_jointplacebo)
            out_names.append("p_jointplacebo")

    if predict_het is not None and len(predict_het_good) > 0:
        did_multiplegt_dyn.append(het_res)
        out_names.append("predict_het")

    
    # -------------------------------------
    # Collect delta if normalized
    # -------------------------------------
    delta = {}
    if normalized:
        for i in range(1, l_XX + 1):
            delta[f"delta_D_{i}_global_XX"] = dict_glob.get(f"delta_D_{i}_global_XX")

    # -------------------------------------
    # Collect coefficients and vcov
    # -------------------------------------
    coef = {
        "b": mat_res_df.iloc[np.r_[0:l_XX, (l_XX + 1):], 0].values,
        "vcov": didmgt_vcov,
    }



    # -------------------------------------
    # Average Time on Effects
    # -------------------------------------
    if not trends_lin:

        # ---- FIXED LINE: no clip_max, use when/then/otherwise ----
        expr_M = pl.col("T_g_XX") - pl.col("F_g_XX") + 1
        df = df.with_columns(
            pl.when(expr_M > l_XX)
            .then(pl.lit(l_XX))
            .otherwise(expr_M)
            .alias("M_g_XX")
        )

        # Initialize delta_D_g_XX
        df = df.with_columns(
            pl.lit(None).cast(pl.Float64).alias("delta_D_g_XX")
        )

        for j in range(1, l_XX + 1):
            plus_col = f"delta_D_g_{j}_plus_XX"
            minus_col = f"delta_D_g_{j}_minus_XX"
            tmp_col = f"delta_D_g_{j}_XX"

            # delta_D_g_j = plus, or minus if plus==0
            df = df.with_columns(
                pl.when(pl.col(plus_col) == 0)
                .then(pl.col(minus_col))
                .otherwise(pl.col(plus_col))
                .alias(tmp_col)
            )

            # set 0 to null
            df = df.with_columns(
                pl.when(pl.col(tmp_col) == 0)
                .then(pl.lit(None))
                .otherwise(pl.col(tmp_col))
                .alias(tmp_col)
            )

            # delta_D_g_XX = delta_D_g_j where switcher_tag_XX == j
            df = df.with_columns(
                pl.when(pl.col("switcher_tag_XX") == j)
                .then(pl.col(tmp_col))
                .otherwise(pl.col("delta_D_g_XX"))
                .alias("delta_D_g_XX")
            )

            df = df.drop(tmp_col)

        # delta_D_g_num_XX = delta_D_g_XX * (M_g_XX - (switcher_tag_XX - 1))
        df = df.with_columns(
            (
                pl.col("delta_D_g_XX")
                * (pl.col("M_g_XX") - (pl.col("switcher_tag_XX") - 1))
            ).alias("delta_D_g_num_XX")
        )

        delta_D_num_total_XX = df.select(
            pl.col("delta_D_g_num_XX").sum()
        ).item()

        delta_D_denom_total_XX = df.select(
            pl.col("delta_D_g_XX").sum()
        ).item()

        avg_periods = delta_D_num_total_XX / delta_D_denom_total_XX


    # -------------------------------------
    # Adding agregation of Effects
    # -------------------------------------

    if trends_lin is None:
        did_multiplegt_dyn.append(avg_periods)
        out_names.append("avg_period")


    # -------------------------------------
    # Converting to Dict
    # -------------------------------------
    did_multiplegt_dyn = dict(zip(out_names, did_multiplegt_dyn))


    # -------------------------------------
    # Design option: detect treatment paths
    # -------------------------------------
    design_output = None
    if design is not None:
        try:
            from ._design import run_design_analysis
            design_output = run_design_analysis(
                df=df,
                design=design,
                l_XX=l_XX,
                T_max_XX=T_max_XX,
                weight_col="weight_XX" if weight else None
            )
        except Exception as e:
            warnings.warn(f"Design option failed: {str(e)}", UserWarning)

    # -------------------------------------
    # Normalized weights option
    # -------------------------------------
    normalized_weights_output = None
    if normalized_weights and normalized:
        try:
            normalized_weights_output = print_normalized_weights(
                df=df,
                l_XX=l_XX,
                group_col="group_XX",
                time_col="time_XX",
                treatment_col="treatment_XX",
                f_g_col="F_g_XX",
                t_g_col="T_g_XX",
                d_sq_col="d_sq_XX",
                n_gt_col="N_gt_XX"
            )
        except Exception as e:
            warnings.warn(f"Normalized weights option failed: {str(e)}", UserWarning)

    # -------------------------------------
    # Date first switch option
    # -------------------------------------
    date_first_switch_output = None
    if date_first_switch is not None:
        try:
            # Add T_max_XX as a column for the date_first_switch function
            df_with_tmax = df.with_columns(
                pl.lit(T_max_XX).alias("T_max_XX")
            )
            date_first_switch_output = run_date_first_switch(
                df=df_with_tmax,
                date_first_switch=date_first_switch,
                group_col="group_XX",
                time_col="time_XX",
                f_g_col="F_g_XX",
                t_max_col="T_max_XX",
                d_sq_col="d_sq_XX"
            )
        except Exception as e:
            warnings.warn(f"Date first switch option failed: {str(e)}", UserWarning)

    # -------------------------------------
    # Assemble return object
    # -------------------------------------
    ret = {
        "df": df,
        "did_multiplegt_dyn": did_multiplegt_dyn,
        "delta": delta,
        "l_XX": l_XX,
        "T_max_XX": T_max_XX,
        "mat_res_XX": mat_res_df,
        'dict_glob' : dict_glob
    }

    if placebo != 0:
        ret["l_placebo_XX"] = l_placebo_XX

    ret["coef"] = coef

    if design_output is not None:
        ret["design"] = design_output

    if normalized_weights_output is not None:
        ret["normalized_weights"] = normalized_weights_output

    if date_first_switch_output is not None:
        ret["date_first_switch"] = date_first_switch_output

    return ret