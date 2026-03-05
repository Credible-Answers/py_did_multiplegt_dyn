from .did_multiplegt_main import did_multiplegt_main
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import polars as pl
from ._utils import *

class DidMultiplegtDyn:
    """
    Difference-in-Differences estimator with multiple groups and periods.

    Estimates event-study DID effects in designs with multiple groups and periods,
    potentially non-binary treatment that may increase or decrease multiple times.

    Parameters
    ----------
    df : DataFrame
        Input data (Polars or Pandas DataFrame)
    outcome : str
        Name of outcome variable
    group : str
        Name of group identifier variable
    time : str
        Name of time variable
    treatment : str
        Name of treatment variable
    cluster : str, optional
        Clustering variable for standard errors
    effects : int, default 1
        Number of dynamic effects to estimate
    placebo : int, default 1
        Number of pre-treatment placebo periods
    normalized : bool, default False
        Whether to normalize effects by treatment intensity
    effects_equal : bool, default False
        Test for equality of all effects
    controls : list, optional
        List of control variable names
    trends_nonparam : list, optional
        Variables for non-parametric trends
    trends_lin : bool, default False
        Include linear time trends
    continuous : int, default 0
        Polynomial degree for continuous treatment (0 = binary)
    weight : str, optional
        Name of weight variable
    predict_het : list, optional
        Heterogeneous effects specification [variables, effect_numbers]
    same_switchers : bool, default False
        Restrict to same switchers across effects
    same_switchers_pl : bool, default False
        Restrict to same switchers for placebos
    switchers : str, default ""
        Filter switchers: "" (all), "in" (switchers in), "out" (switchers out)
    only_never_switchers : bool, default False
        Use only never-switchers as controls
    ci_level : int, default 95
        Confidence interval level (e.g., 95 for 95% CI)
    save_results : str, optional
        Path to save results CSV
    less_conservative_se : bool, default False
        Use less conservative standard error adjustment
    more_granular_demeaning : bool, default False
        Use more granular demeaning for variance calculations. Automatically
        enables less_conservative_se with adaptive cohort selection.
    dont_drop_larger_lower : bool, default False
        Do not drop larger/lower observations
    drop_if_d_miss_before_first_switch : bool, default False
        Drop observations with missing treatment before first switch
    bootstrap : tuple, optional
        Bootstrap standard errors: (n_reps, seed). Example: (50, 12345)
    by_path : str or int, optional
        Stratify by treatment path: "all" or number of paths
    design : tuple, optional
        Detect treatment paths: (percentage, output_path). Example: (0.9, "console")
    by : str, optional
        Stratify analysis by a grouping variable. When specified, the estimation
        runs separately for each unique value of this variable. Results are stored
        with headers indicating "By: {varname} = {value}".
    normalized_weights : bool, default False
        When combined with normalized=True, reports the weights that normalized
        effects place on current and lagged treatments.
    predict_het_hc2bm : bool, default False
        When using predict_het, compute HC2 standard errors with Bell-McCaffrey
        degrees of freedom adjustment for more conservative inference.
    date_first_switch : tuple, optional
        Reports dates of first treatment changes and number of groups switching.
        Format: (by_baseline_treat, output_path) where by_baseline_treat is bool
        and output_path is "console" or a file path. Example: (False, "console")
    """
    def __init__(self,
        df,
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

        # more_granular_demeaning automatically enables less_conservative_se
        if more_granular_demeaning:
            less_conservative_se = True

        # Validate normalized_weights requires normalized
        if normalized_weights and not normalized:
            raise ValueError("normalized_weights option requires normalized=True")

        # Validate predict_het_hc2bm requires predict_het
        if predict_het_hc2bm and predict_het is None:
            raise ValueError("predict_het_hc2bm option requires predict_het to be specified")

        #### Getting the initial conditions
        self.args = dict(
            df=df,
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
            save_results=save_results,
            less_conservative_se=less_conservative_se,
            more_granular_demeaning=more_granular_demeaning,
            dont_drop_larger_lower=dont_drop_larger_lower,
            drop_if_d_miss_before_first_switch=drop_if_d_miss_before_first_switch,
            bootstrap=bootstrap,
            by_path=by_path,
            design=design,
            by=by,
            normalized_weights=normalized_weights,
            predict_het_hc2bm=predict_het_hc2bm,
            date_first_switch=date_first_switch
        )

        validated = validate_inputs( **self.args )
    
    def fit( self ):
        ret = did_multiplegt_main( **self.args )
        self.result = ret
        return self

    def summary(self):
        """
        Collect Effects, ATE, Placebos, and predict_het from result['did_multiplegt_dyn'],
        append them into one DataFrame, and print it nicely.

        Mimics the Stata output format for did_multiplegt_dyn.

        When 'by' or 'by_path' options are used, prints results for all subgroups.

        Returns
        -------
        summary_df : pandas.DataFrame or list
            Combined table(s) with results. Returns a list if multiple subgroups exist.
        """
        import pandas as pd

        hide_ate_nans = True
        dyn = self.result["did_multiplegt_dyn"]

        def _to_df(obj, block_name):
            """Coerce obj to DataFrame and add a 'Block' column."""
            if obj is None:
                return pd.DataFrame()
            if isinstance(obj, pd.DataFrame):
                df = obj.copy()
            elif isinstance(obj, pd.Series):
                df = obj.to_frame().T
            else:
                df = pd.DataFrame({"value": [obj]})
            df = df.reset_index().rename(columns={'index': 'Block'})
            return df

        def _print_single_result(result_dyn, by_var=None, by_value=None, path_info=None):
            """Print results for a single estimation (or subgroup)."""
            effects = result_dyn.get("Effects")
            ate = result_dyn.get("ATE")
            placebos = result_dyn.get("Placebos")
            predict_het = result_dyn.get("predict_het", None)

            # Print header
            print("=" * 80)
            print(" " * 13 + "Estimation of treatment effects: Event-study effects")

            # Print by option header if applicable
            if by_var is not None and by_value is not None:
                print(" " * 35 + f"By: {by_var} = {by_value}")

            # Print by_path header if applicable
            if path_info is not None:
                path_str = path_info.get('treatment_sequence', [])
                if isinstance(path_str, list):
                    path_str = ', '.join([str(x) for x in path_str])
                n_groups = path_info.get('n_groups', '')
                print(" " * 30 + f"Path ({path_str})")
                if n_groups:
                    print(" " * 30 + f"{n_groups} switchers")

            print("=" * 80)

            if effects is None or (isinstance(effects, pd.DataFrame) and effects.empty):
                print("No effects estimated for this subgroup.")
                print("=" * 80)
                return pd.DataFrame()

            eff_df = _to_df(effects, "Effects")
            ate_df = _to_df(ate, "ATE")
            pl_df = _to_df(placebos, "Placebos")

            # Hide NaNs in the ATE block only
            if hide_ate_nans and not ate_df.empty:
                ate_df = ate_df.where(~ate_df.isna(), "")

            summary_df = pd.concat([eff_df, ate_df, pl_df], ignore_index=True, sort=False)
            print(summary_df.to_string(index=False))
            print("=" * 80)

            # Print joint test p-values if available
            p_jointeffects = result_dyn.get("p_jointeffects", None)
            if p_jointeffects is not None:
                if not np.isnan(p_jointeffects):
                    print(f"Test of joint nullity of the effects: p-value = {p_jointeffects:.6f}")

            p_jointplacebo = result_dyn.get("p_jointplacebo", None)
            if p_jointplacebo is not None:
                if not np.isnan(p_jointplacebo):
                    print(f"Test of joint nullity of the placebos: p-value = {p_jointplacebo:.6f}")

            p_equality_effects = result_dyn.get("p_equality_effects", None)
            if p_equality_effects is not None:
                if not np.isnan(p_equality_effects):
                    print(f"Test of equality of the effects: p-value = {p_equality_effects:.6f}")

            # Print variance matrix warnings if present
            effects_var_warning = result_dyn.get("effects_var_warning", None)
            if effects_var_warning is not None:
                print()
                print("WARNING: " + effects_var_warning)

            placebo_var_warning = result_dyn.get("placebo_var_warning", None)
            if placebo_var_warning is not None:
                print()
                print("WARNING: " + placebo_var_warning)

            effects_equal_var_warning = result_dyn.get("effects_equal_var_warning", None)
            if effects_equal_var_warning is not None:
                print()
                print("WARNING: " + effects_equal_var_warning)

            # Print predict_het results if available
            if predict_het is not None and len(predict_het) > 0:
                print("\n" + "=" * 60)
                print(" " * 15 + "Heterogeneous Effects (predict_het)")
                print("=" * 60)
                if isinstance(predict_het, pd.DataFrame):
                    print(predict_het.to_string(index=True))
                else:
                    print(predict_het)

            return summary_df

        # Check if we have multiple subgroups from 'by' option
        all_by_results = dyn.get("all_by_results", None)
        if all_by_results is not None and len(all_by_results) > 0:
            all_summaries = []
            for item in all_by_results:
                by_var = item.get('by_var')
                by_value = item.get('by_value')
                sub_result = item.get('result', {})
                sub_dyn = sub_result.get('did_multiplegt_dyn', {})
                summary_df = _print_single_result(sub_dyn, by_var=by_var, by_value=by_value)
                all_summaries.append(summary_df)
                print()  # Empty line between subgroups

            print("The development of this package was funded by the European Union.")
            print("ERC REALLYCREDIBLE - GA N. 101043899")
            return all_summaries

        # Check if we have multiple subgroups from 'by_path' option
        all_path_results = dyn.get("all_path_results", None)
        if all_path_results is not None and len(all_path_results) > 0:
            all_summaries = []
            for item in all_path_results:
                path_info = {
                    'treatment_sequence': item.get('treatment_sequence', []),
                    'n_groups': item.get('n_groups', 0)
                }
                sub_result = item.get('result', {})
                sub_dyn = sub_result.get('did_multiplegt_dyn', {})
                summary_df = _print_single_result(sub_dyn, path_info=path_info)
                all_summaries.append(summary_df)
                print()  # Empty line between paths

            print("The development of this package was funded by the European Union.")
            print("ERC REALLYCREDIBLE - GA N. 101043899")
            return all_summaries

        # Single result (no by or by_path)
        summary_df = _print_single_result(dyn)
        print("\nThe development of this package was funded by the European Union.")
        print("ERC REALLYCREDIBLE - GA N. 101043899")

        return summary_df



    def plot(self, 
        *,
        n_placebos=None,          # number of pre-treatment periods to show (closest to 0)
        n_effects=None,           # number of post-treatment periods to show
        x_label="Time from Treatment",
        y_label="Estimate",
        title=None,
        note=None,
        label_last_post_as_plus=True,
        fit_pretrend_line=False,
        report_pretrend_in_note=False,
        rotate_by_pretrend=False,
        pretrend_line_kwargs=None,
        figsize=(6.5, 4.5),
        pretrend_decimals=3
    ):
        """
        Plot an event-study figure (placebos + effects) using the did_multiplegt_dyn
        result object.

        Parameters
        ----------
        result : dict-like
            Object containing result['did_multiplegt_dyn']['Effects'] and
            result['did_multiplegt_dyn']['Placebos'].

        n_placebos : int or None
            How many pre-treatment periods (placebos) to display. If None, show all.
            If k > 0, shows the k periods closest to 0 (e.g., -k,...,-1).
            If 0, no pre-periods are shown.

        n_effects : int or None
            How many post-treatment periods (effects) to display. If None, show all.
            If k > 0, shows 1,...,k.
            If 0, no post-periods are shown.

        x_label, y_label : str
            Axis labels.

        title : str or None
            Title for the figure.

        note : str or None
            Text at the bottom (e.g., "Notes: ...").

        label_last_post_as_plus : bool
            If True and there are positive event times, the largest positive
            x-tick label is shown as "k+" instead of "k".

        fit_pretrend_line : bool
            If True AND rotate_by_pretrend is False, fit a line on displayed
            pre-treatment coefficients, constrained to pass through (0,0),
            and draw it.

        report_pretrend_in_note : bool
            If True and a pretrend slope can be estimated, append
            "Pre-trend slope = ..." to the note.

        rotate_by_pretrend : bool
            If True, subtract the predicted value from the pretrend line
            from all displayed coefficients (both pre and post) and also
            shift the CI bounds by the same amount. Uses the slope estimated
            from the displayed pre-periods.
            If rotate_by_pretrend is True, fit_pretrend_line is ignored
            (the pretrend would be zero after rotation).

        pretrend_line_kwargs : dict or None
            Extra kwargs to pass to ax.plot for the pretrend line.

        figsize : tuple
            Matplotlib figure size.

        pretrend_decimals : int
            Number of decimals for reporting the pretrend slope in the note.

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        result = self.result
        col_est = "Estimate"
        col_lb = "LB CI"
        col_ub = "UB CI"
        time_col = "time"

        # --- extract tables ---
        effects = result["did_multiplegt_dyn"]["Effects"].copy()
        placebos = result["did_multiplegt_dyn"]["Placebos"].copy()

        # --- construct time columns ---
        n_pl_all = placebos.shape[0]
        placebos[time_col] = np.arange(1, n_pl_all+1)*-1
        n_eff_all = effects.shape[0]
        effects[time_col] = np.arange(1, n_eff_all + 1)

        # sort by time
        placebos = placebos.sort_values(time_col)
        effects = effects.sort_values(time_col)

        # --- subset by requested number of placebos/effects ---
        pl = placebos.copy()
        eff = effects.copy()

        if n_placebos is not None:
            if n_placebos > 0:
                pl = pl[pl[time_col] < 0].iloc[-n_placebos:]
            else:
                pl = pl.iloc[0:0]  # empty

        if n_effects is not None:
            if n_effects > 0:
                eff = eff[eff[time_col] > 0].iloc[:n_effects]
            else:
                eff = eff.iloc[0:0]

        # recompute counts after subsetting
        n_pl = pl.shape[0]
        n_eff = eff.shape[0]

        # --- estimate pretrend slope (on displayed pre-periods) ---
        beta_pretrend = None
        if n_pl > 0:
            t_pre = pl[time_col].to_numpy().astype(float)
            y_pre = pl[col_est].to_numpy().astype(float)
            mask = ~np.isnan(t_pre) & ~np.isnan(y_pre)
            t_pre = t_pre[mask]
            y_pre = y_pre[mask]
            denom = np.sum(t_pre ** 2)
            if t_pre.size > 0 and denom > 0:
                beta_pretrend = float(np.sum(t_pre * y_pre) / denom)

        # --- rotate coefficients by pretrend, if requested ---
        if rotate_by_pretrend:
            if beta_pretrend is None:
                raise ValueError(
                    "Cannot rotate by pretrend: no valid pre-periods to estimate the slope."
                )
            for df in (pl, eff):
                if df.shape[0] == 0:
                    continue
                t = df[time_col].to_numpy().astype(float)
                pred = beta_pretrend * t
                df[col_est] = df[col_est] - pred
                df[col_lb] = df[col_lb] - pred
                df[col_ub] = df[col_ub] - pred
            # After rotation, it doesn't make sense to plot the original sloped line
            fit_pretrend_line = False

        # --- style (similar to journal/event-study figure) ---
        sns.set_theme(style="white", context="paper")
        plt.rcParams["font.family"] = "serif"
        plt.rcParams["font.size"] = 12

        fig, ax = plt.subplots(figsize=figsize)

        # --- pre-treatment (placebos): vertical CI + dots ---
        if n_pl > 0:
            ax.vlines(
                x=pl[time_col],
                ymin=pl[col_lb],
                ymax=pl[col_ub],
                color="black",
                linewidth=2,
            )
            ax.scatter(
                pl[time_col],
                pl[col_est],
                color="black",
                s=35,
                zorder=3,
            )

        # --- post-treatment (effects): vertical CI + dots ---
        if n_eff > 0:
            ax.vlines(
                x=eff[time_col],
                ymin=eff[col_lb],
                ymax=eff[col_ub],
                color="black",
                linewidth=2,
            )
            ax.scatter(
                eff[time_col],
                eff[col_est],
                color="black",
                s=35,
                zorder=3,
            )

        # --- always show effect at 0 with CI [0,0] ---
        ax.vlines(0, 0, 0, color="black", linewidth=2)
        ax.scatter(0, 0, color="black", s=35, zorder=4)

        # --- pretrend line (optional, unrotated coordinates) ---
        if fit_pretrend_line and (beta_pretrend is not None):
            # Determine x range just from displayed periods plus 0
            x_candidates = [0]
            if n_pl > 0:
                x_candidates.append(int(pl[time_col].min()))
            if n_eff > 0:
                x_candidates.append(int(eff[time_col].max()))
            x_min = min(x_candidates)
            x_max = max(x_candidates)

            x_line = np.linspace(x_min, x_max, 200)
            y_line = beta_pretrend * x_line

            if pretrend_line_kwargs is None:
                pretrend_line_kwargs = {}
            pretrend_default = {
                "linestyle": "--",
                "linewidth": 1.2,
                "color": "black",
                "alpha": 0.7,
            }
            pretrend_default.update(pretrend_line_kwargs)
            ax.plot(x_line, y_line, **pretrend_default)

        # --- zero lines (axes) ---
        ax.axhline(0, color="black", linewidth=1)
        ax.axvline(0, color="black", linewidth=1)

        # --- ticks ---
        x_candidates = [0]
        if n_pl > 0:
            x_candidates.append(int(pl[time_col].min()))
        if n_eff > 0:
            x_candidates.append(int(eff[time_col].max()))
        x_min = min(x_candidates)
        x_max = max(x_candidates)

        xticks = list(range(x_min, x_max + 1))
        xtick_labels = [str(x) for x in xticks]

        if label_last_post_as_plus and x_max > 0:
            idx = xticks.index(x_max)
            xtick_labels[idx] = f"{x_max}"

        ax.set_xticks(xticks)
        ax.set_xticklabels(xtick_labels)

        # only horizontal dashed grid
        ax.yaxis.grid(True, linestyle="--", linewidth=0.7, color="0.8")
        ax.xaxis.grid(False)

        # remove top and right spines
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        ax.tick_params(axis="both", direction="out", length=4)

        # labels
        ax.set_xlabel(x_label, labelpad=10)
        ax.set_ylabel(y_label)

        # --- build note (possibly add pretrend info) ---
        final_note = note
        if report_pretrend_in_note and (beta_pretrend is not None):
            slope_txt = f"Pre-trend slope (based on displayed pre-treatment coefficients) = {beta_pretrend:.{pretrend_decimals}f}."
            if final_note is None:
                final_note = slope_txt
            else:
                final_note = final_note.rstrip() + " " + slope_txt

        # title
        if title is not None:
            ax.set_title(title, loc="center", pad=15)

        # note at bottom
        if final_note is not None:
            fig.text(0.5, 0.02, final_note, ha="center", va="bottom", fontsize=9)
            fig.subplots_adjust(top=0.82, bottom=0.20, left=0.10, right=0.97)
        else:
            fig.tight_layout()

        return self

    def plot_panelview(
        self,
        *,
        show_for: str = "effect_1",
        sample_pct: float = 0.10,
        figsize: tuple = (14, 10),
        title: str = None,
        show_legend: bool = True
    ):
        """
        Plot a panelview-style visualization showing which observations are used
        for constructing a specific effect or placebo.

        Shows a sample (default 10%) of switchers marked with "T" and controls
        marked with "C" to illustrate which observations contribute to the estimate.

        Parameters
        ----------
        show_for : str
            Which effect or placebo to highlight. Format: "effect_N" or "placebo_N"
            Examples: "effect_1", "placebo_2". Default: "effect_1"

        sample_pct : float
            Percentage of switchers and controls to display (0.0 to 1.0). Default: 0.10

        figsize : tuple
            Figure size. Default: (14, 10)

        title : str, optional
            Custom title. If None, auto-generates.

        show_legend : bool
            Whether to show legend. Default: True

        Returns
        -------
        fig, ax : matplotlib Figure and Axes
        """
        import matplotlib.patches as mpatches
        import numpy as np

        # Parse show_for parameter
        show_for_lower = show_for.lower().strip()
        if show_for_lower.startswith("effect_"):
            try:
                l = int(show_for_lower.replace("effect_", ""))
                is_effect = True
            except ValueError:
                raise ValueError(f"Invalid format: {show_for}. Use 'effect_N' or 'placebo_N'")
        elif show_for_lower.startswith("placebo_"):
            try:
                l = int(show_for_lower.replace("placebo_", ""))
                is_effect = False
            except ValueError:
                raise ValueError(f"Invalid format: {show_for}. Use 'effect_N' or 'placebo_N'")
        else:
            raise ValueError(f"Invalid format: {show_for}. Use 'effect_N' or 'placebo_N'")

        # Get the result
        if self.result is None:
            raise ValueError("No results available. Call fit() first.")

        df_internal = self.result.get("df", None)
        if df_internal is None:
            raise ValueError("Internal data not available.")

        # Convert to pandas
        if hasattr(df_internal, 'to_pandas'):
            df = df_internal.to_pandas()
        else:
            df = df_internal.copy()

        # Get column mappings
        group_col = self.args["group"]
        time_col = self.args["time"]

        # Get internal time and group info
        times = sorted(df['time_XX'].unique())
        n_times = len(times)
        max_time = max(times)
        min_time = min(times)

        # Map internal time to original for display
        # The internal df stores original time in 'time' column
        if 'time' in df.columns:
            time_to_orig = df.groupby('time_XX')['time'].first().to_dict()
        else:
            time_to_orig = {t: t for t in times}

        # Get group info - use 'group' column for original group IDs
        agg_cols = {'F_g_XX': 'first', 'T_g_XX': 'first', 'd_sq_XX': 'first'}
        if 'group' in df.columns:
            agg_cols['group'] = 'first'
        group_info = df.groupby('group_XX').agg(agg_cols).to_dict('index')

        # Add original group ID if not present
        for g in group_info:
            if 'group' not in group_info[g]:
                group_info[g]['group'] = g

        all_groups = sorted(group_info.keys())

        # Identify switcher groups and control groups for this effect/placebo
        switcher_groups = []  # Groups that are switchers
        switcher_cells = []   # (group, time) cells for switchers
        control_cells = []    # (group, time) cells for controls

        if is_effect:
            # For effect l: switchers are at time F_g - 1 + l
            for g in all_groups:
                info = group_info[g]
                f_g = info['F_g_XX']
                t_g = info['T_g_XX']

                # Switcher condition: F_g <= T_g and l <= T_g - F_g + 1
                if f_g <= t_g and l <= (t_g - f_g + 1):
                    effect_time = int(f_g - 1 + l)
                    baseline_time = int(f_g - 1)

                    if effect_time <= max_time and baseline_time >= min_time:
                        switcher_groups.append(g)
                        switcher_cells.append((g, effect_time))
                        switcher_cells.append((g, baseline_time))

            # Get time periods used by switchers, grouped by d_sq
            switcher_times_by_dsq = {}
            for g in switcher_groups:
                d_sq = group_info[g]['d_sq_XX']
                f_g = group_info[g]['F_g_XX']
                effect_time = int(f_g - 1 + l)
                baseline_time = int(f_g - 1)

                if d_sq not in switcher_times_by_dsq:
                    switcher_times_by_dsq[d_sq] = set()
                switcher_times_by_dsq[d_sq].add(effect_time)
                switcher_times_by_dsq[d_sq].add(baseline_time)

            # Find controls: groups that haven't switched at those times
            for g in all_groups:
                if g in switcher_groups:
                    continue
                info = group_info[g]
                f_g = info['F_g_XX']
                d_sq = info['d_sq_XX']

                times_needed = switcher_times_by_dsq.get(d_sq, set())
                for t in times_needed:
                    # Control if F_g > t (haven't switched yet)
                    if f_g > t:
                        control_cells.append((g, t))

        else:
            # For placebo l: use times F_g - 1 and F_g - 1 - l
            for g in all_groups:
                info = group_info[g]
                f_g = info['F_g_XX']
                t_g = info['T_g_XX']

                baseline_time = int(f_g - 1)
                placebo_time = int(f_g - 1 - l)

                if f_g <= t_g and placebo_time >= min_time and baseline_time >= min_time:
                    switcher_groups.append(g)
                    switcher_cells.append((g, baseline_time))
                    switcher_cells.append((g, placebo_time))

            # Controls for placebo
            switcher_times_by_dsq = {}
            for g in switcher_groups:
                d_sq = group_info[g]['d_sq_XX']
                f_g = group_info[g]['F_g_XX']
                baseline_time = int(f_g - 1)
                placebo_time = int(f_g - 1 - l)

                if d_sq not in switcher_times_by_dsq:
                    switcher_times_by_dsq[d_sq] = set()
                switcher_times_by_dsq[d_sq].add(baseline_time)
                switcher_times_by_dsq[d_sq].add(placebo_time)

            for g in all_groups:
                if g in switcher_groups:
                    continue
                info = group_info[g]
                f_g = info['F_g_XX']
                d_sq = info['d_sq_XX']

                times_needed = switcher_times_by_dsq.get(d_sq, set())
                for t in times_needed:
                    if f_g > t:
                        control_cells.append((g, t))

        # Get unique control groups
        control_groups = list(set(g for g, t in control_cells))

        # Sample groups to display
        n_switchers_sample = max(1, int(len(switcher_groups) * sample_pct))
        n_controls_sample = max(1, int(len(control_groups) * sample_pct))

        # Sort by F_g and take sample
        switcher_groups_sorted = sorted(switcher_groups, key=lambda g: group_info[g]['F_g_XX'])
        control_groups_sorted = sorted(control_groups, key=lambda g: group_info[g]['F_g_XX'])

        # Take evenly spaced sample
        if len(switcher_groups_sorted) > n_switchers_sample:
            indices = np.linspace(0, len(switcher_groups_sorted) - 1, n_switchers_sample, dtype=int)
            sampled_switchers = [switcher_groups_sorted[i] for i in indices]
        else:
            sampled_switchers = switcher_groups_sorted

        if len(control_groups_sorted) > n_controls_sample:
            indices = np.linspace(0, len(control_groups_sorted) - 1, n_controls_sample, dtype=int)
            sampled_controls = [control_groups_sorted[i] for i in indices]
        else:
            sampled_controls = control_groups_sorted

        # Combine and sort all groups to display
        groups_to_show = sorted(
            set(sampled_switchers + sampled_controls),
            key=lambda g: (group_info[g]['F_g_XX'], g)
        )

        n_groups = len(groups_to_show)

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Create mappings
        group_to_idx = {g: i for i, g in enumerate(groups_to_show)}
        time_to_idx = {t: j for j, t in enumerate(times)}

        # Get cells for sampled groups
        sampled_switcher_cells = [(g, t) for g, t in switcher_cells if g in sampled_switchers]
        sampled_control_cells = [(g, t) for g, t in control_cells if g in sampled_controls]

        # Draw all cells with treatment coloring
        for g in groups_to_show:
            i = group_to_idx[g]
            for t in times:
                j = time_to_idx[t]

                # Get treatment value
                mask = (df['group_XX'] == g) & (df['time_XX'] == t)
                if mask.sum() == 0:
                    color = 'white'
                else:
                    d = df.loc[mask, 'treatment_XX'].iloc[0]
                    if d > 0:
                        color = plt.cm.Oranges(0.3 + 0.4 * min(float(d), 1))
                    else:
                        color = plt.cm.Greys(0.15)

                rect = mpatches.Rectangle(
                    (j - 0.5, i - 0.5), 1, 1,
                    facecolor=color, edgecolor='gray', linewidth=0.5
                )
                ax.add_patch(rect)

        # Mark switcher cells with "T" and red border
        for (g, t) in sampled_switcher_cells:
            if g not in group_to_idx or t not in time_to_idx:
                continue
            i = group_to_idx[g]
            j = time_to_idx[t]

            # Red border
            rect = mpatches.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                facecolor='none', edgecolor='red', linewidth=2.5
            )
            ax.add_patch(rect)

            # "T" label
            ax.text(j, i, 'T', ha='center', va='center', fontsize=8,
                   fontweight='bold', color='darkred')

        # Mark control cells with "C" and blue border
        for (g, t) in sampled_control_cells:
            if g not in group_to_idx or t not in time_to_idx:
                continue
            i = group_to_idx[g]
            j = time_to_idx[t]

            # Blue border
            rect = mpatches.Rectangle(
                (j - 0.5, i - 0.5), 1, 1,
                facecolor='none', edgecolor='blue', linewidth=2.5
            )
            ax.add_patch(rect)

            # "C" label
            ax.text(j, i, 'C', ha='center', va='center', fontsize=8,
                   fontweight='bold', color='darkblue')

        # Draw vertical line at F_g for each group
        for g in groups_to_show:
            i = group_to_idx[g]
            f_g = group_info[g]['F_g_XX']
            if f_g <= max_time and f_g in time_to_idx:
                j = time_to_idx[f_g]
                ax.axvline(x=j - 0.5, ymin=(i) / n_groups, ymax=(i + 1) / n_groups,
                          color='black', linewidth=2)

        # Set limits and labels
        ax.set_xlim(-0.5, n_times - 0.5)
        ax.set_ylim(-0.5, n_groups - 0.5)

        # X-axis: time periods
        ax.set_xticks(range(n_times))
        ax.set_xticklabels([str(int(time_to_orig.get(t, t))) for t in times], fontsize=9)
        ax.set_xlabel("Time Period", fontsize=11)

        # Y-axis: group IDs
        ax.set_yticks(range(n_groups))
        ylabels = []
        for g in groups_to_show:
            orig_id = group_info[g].get('group', g)
            marker = "T" if g in sampled_switchers else "C"
            ylabels.append(f"{orig_id} [{marker}]")
        ax.set_yticklabels(ylabels, fontsize=7)
        ax.set_ylabel("Groups [T=Switcher, C=Control]", fontsize=11)

        # Title
        if title is None:
            effect_type = "Effect" if is_effect else "Placebo"
            title = (f"Panel View: {effect_type} {l}\n"
                    f"Total: {len(switcher_groups)} switchers, {len(control_groups)} controls\n"
                    f"Showing {int(sample_pct*100)}% sample: {len(sampled_switchers)} switchers (T), "
                    f"{len(sampled_controls)} controls (C)")
        ax.set_title(title, fontsize=11)

        # Legend
        if show_legend:
            treated_patch = mpatches.Patch(color=plt.cm.Oranges(0.5), label='Treated (D>0)')
            untreated_patch = mpatches.Patch(color=plt.cm.Greys(0.2), label='Untreated (D=0)')
            switcher_patch = mpatches.Patch(
                facecolor='none', edgecolor='red', linewidth=2,
                label='Switcher obs (T)'
            )
            control_patch = mpatches.Patch(
                facecolor='none', edgecolor='blue', linewidth=2,
                label='Control obs (C)'
            )
            switch_line = plt.Line2D([0], [0], color='black', linewidth=2,
                                     label='First switch (F_g)')
            ax.legend(handles=[treated_patch, untreated_patch, switcher_patch,
                              control_patch, switch_line],
                     loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=9)

        plt.tight_layout()
        return fig, ax
