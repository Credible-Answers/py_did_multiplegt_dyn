import polars as pl
import numpy as np
import math
import warnings
from typing import Optional, Sequence, Union, Dict, Any
import polars as pl


def validate_inputs(
    df: pl.DataFrame,
    outcome,
    group,
    time,
    treatment,
    *,
    cluster: Optional[str] = None,
    effects: Union[int, float] = 1,
    placebo: Union[int, float] = 1,
    normalized: bool = False,
    effects_equal: bool = False,
    controls: Optional[Sequence[str]] = None,
    trends_nonparam: Optional[Sequence[str]] = None,
    trends_lin: bool = False,
    continuous: Union[int, float] = 0,
    weight: Optional[str] = None,
    predict_het=None,
    same_switchers: bool = False,
    same_switchers_pl: bool = False,
    switchers: str = "",
    only_never_switchers: bool = False,
    ci_level: Union[int, float] = 95,
    save_results: Optional[str] = None,
    less_conservative_se: bool = False,
    more_granular_demeaning: bool = False,
    dont_drop_larger_lower: bool = False,
    drop_if_d_miss_before_first_switch: bool = False,
    bootstrap=None,
    by_path=None,
    design=None,
    by: Optional[str] = None,
    normalized_weights: bool = False,
    predict_het_hc2bm: bool = False,
    date_first_switch=None,
) -> Dict[str, Any]:
    """
    Validate constructor arguments for the DID class.

    Rules enforced:
      1. df must be a polars.DataFrame
      2. outcome, group, time, treatment must be strings naming columns in df
         (cluster and weight are optional but, if given, must be column names)
      3. effects and placebo must be numbers
      4. several flags must be booleans
      5. controls and trends_nonparam must be list/sequence of column-name strings
      6. ci_level must be >= 50
    """

    # 1. df must be a polars DataFrame
    if not isinstance(df, pl.DataFrame):
        raise TypeError(f"`df` must be a polars.DataFrame, got {type(df)}")

    df_cols = set(df.columns)

    # helper for required string column
    def _check_str_col(name, label):
        if not isinstance(name, str):
            raise TypeError(f"`{label}` must be a string with a column name, got {type(name)}")
        if name not in df_cols:
            raise ValueError(f"`{label}`='{name}' is not a column of df")
        return name

    # 2. outcome, group, time, treatment: required string column names
    outcome = _check_str_col(outcome, "outcome")
    group = _check_str_col(group, "group")
    time = _check_str_col(time, "time")
    treatment = _check_str_col(treatment, "treatment")

    # cluster & weight: optional string column names
    def _check_optional_str_col(name, label):
        if name is None:
            return None
        if not isinstance(name, str):
            raise TypeError(f"`{label}` must be a string with a column name or None, got {type(name)}")
        if name not in df_cols:
            raise ValueError(f"`{label}`='{name}' is not a column of df")
        return name

    cluster = _check_optional_str_col(cluster, "cluster")
    weight = _check_optional_str_col(weight, "weight")
    by = _check_optional_str_col(by, "by")

    # 3. effects and placebo must be numbers
    if not isinstance(effects, (int, float)):
        raise TypeError(f"`effects` must be a number, got {type(effects)}")
    if not isinstance(placebo, (int, float)):
        raise TypeError(f"`placebo` must be a number, got {type(placebo)}")

    # 4. boolean flags
    bool_params = {
        "normalized": normalized,
        "effects_equal": effects_equal,
        "trends_lin": trends_lin,
        "same_switchers": same_switchers,
        "same_switchers_pl": same_switchers_pl,
        "only_never_switchers": only_never_switchers,
        "less_conservative_se": less_conservative_se,
        "more_granular_demeaning": more_granular_demeaning,
        "dont_drop_larger_lower": dont_drop_larger_lower,
        "drop_if_d_miss_before_first_switch": drop_if_d_miss_before_first_switch,
        "normalized_weights": normalized_weights,
        "predict_het_hc2bm": predict_het_hc2bm,
    }
    for name, value in bool_params.items():
        if not isinstance(value, bool):
            raise TypeError(f"`{name}` must be boolean, got {type(value)}")

    # 5. controls and trends_nonparam: list/sequence of string column names
    def _check_str_list(cols, label):
        if cols is None:
            return None
        if isinstance(cols, str):
            # disallow a single string (ambiguous) to avoid mistakes
            raise TypeError(
                f"`{label}` must be a list/sequence of column-name strings, "
                f"not a single string. Use [{label!r}] if you want one column."
            )
        try:
            cols_list = list(cols)
        except TypeError:
            raise TypeError(f"`{label}` must be a sequence of strings or None")

        for c in cols_list:
            if not isinstance(c, str):
                raise TypeError(f"All elements of `{label}` must be strings, got {type(c)}")
            if c not in df_cols:
                raise ValueError(f"`{label}` contains '{c}', which is not a column of df")
        return cols_list

    controls = _check_str_list(controls, "controls")
    trends_nonparam = _check_str_list(trends_nonparam, "trends_nonparam")

    # 6. ci_level must be >= 50
    if not isinstance(ci_level, (int, float)):
        raise TypeError(f"`ci_level` must be numeric, got {type(ci_level)}")
    if ci_level < 50:
        raise ValueError(f"`ci_level` must be >= 50, got {ci_level}")

    # optional params with light checks
    if save_results is not None and not isinstance(save_results, str):
        raise TypeError("`save_results` must be a string path or None")

    if not isinstance(switchers, str):
        raise TypeError("`switchers` must be a string (possibly empty)")

    # we leave `continuous` and `predict_het` largely unchecked on purpose,
    # since you did not specify constraints for them.

    # Validate bootstrap/continuous combination
    if bootstrap is not None and continuous == 0:
        warnings.warn(
            "You specified the bootstrap option without the continuous option. "
            "Bootstrap is typically used with continuous treatments for valid inference.",
            UserWarning
        )

    # Validate by/by_path mutual exclusivity
    if by is not None and by_path is not None:
        raise ValueError(
            "Cannot specify both 'by' and 'by_path' options. "
            "Use 'by' to stratify by a grouping variable, or 'by_path' to stratify by treatment paths."
        )

    # ---- Build validated dict to return ----
    validated: Dict[str, Any] = dict(
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
        date_first_switch=date_first_switch,
    )

    # # ---- User-facing summary printout ----
    # print("✅ Input check passed.")
    # print(f"   • df: polars.DataFrame with shape {df.shape}")
    # print(f"   • outcome: '{outcome}', group: '{group}', time: '{time}', treatment: '{treatment}'")
    # if cluster is not None:
    #     print(f"   • cluster: '{cluster}'")
    # else:
    #     print(f"   • cluster: None (no clustering)")

    # if weight is not None:
    #     print(f"   • weight: '{weight}'")
    # else:
    #     print(f"   • weight: None (unweighted)")

    # print(f"   • effects: {effects}, placebo: {placebo}")
    # print(f"   • normalized={normalized}, effects_equal={effects_equal}")
    # print(f"   • trends_lin={trends_lin}, same_switchers={same_switchers}, same_switchers_pl={same_switchers_pl}")
    # print(f"   • only_never_switchers={only_never_switchers}")
    # print(f"   • less_conservative_se={less_conservative_se}, dont_drop_larger_lower={dont_drop_larger_lower}")
    # print(f"   • drop_if_d_miss_before_first_switch={drop_if_d_miss_before_first_switch}")
    # print(f"   • ci_level={ci_level}")
    # if controls:
    #     print(f"   • controls: {controls}")
    # else:
    #     print(f"   • controls: None")
    # if trends_nonparam:
    #     print(f"   • trends_nonparam: {trends_nonparam}")
    # else:
    #     print(f"   • trends_nonparam: None")

    return validated



# ------------------------------------------------------------
# 1️⃣ — BASIC UTILITY FUNCTIONS
# ------------------------------------------------------------
def gaussian_elimination(A, b, tol=1e-7, verbose=False):
    """Perform Gaussian elimination on augmented matrix [A|b]."""
    A = A.astype(float).copy()
    b = b.astype(float).copy()
    m, n = A.shape
    Ab = np.hstack([A, b])
    
    for i in range(min(m, n)):
        # Pivot
        max_row = np.argmax(np.abs(Ab[i:, i])) + i
        if abs(Ab[max_row, i]) < tol:
            continue
        if max_row != i:
            Ab[[i, max_row]] = Ab[[max_row, i]]
        
        # Normalize pivot row
        Ab[i] = Ab[i] / Ab[i, i]
        
        # Eliminate other rows
        for j in range(m):
            if j != i:
                Ab[j] -= Ab[j, i] * Ab[i]
    
    return Ab

def Ginv(A, tol=np.sqrt(np.finfo(float).eps), verbose=False):
    """Generalized inverse using Gaussian elimination (R's Ginv)."""
    A = np.asarray(A, dtype=float)
    m, n = A.shape
    
    # First elimination
    B = gaussian_elimination(A, np.eye(m), tol=tol, verbose=verbose)
    L = B[:, n:]      # columns after first n
    AR = B[:, :n]     # first n columns
    
    # Second elimination
    C = gaussian_elimination(AR.T, np.eye(n), tol=tol, verbose=verbose)
    R = C[:, m:].T    # columns after first m, then transpose
    AC = C[:, :m].T   # first m columns, then transpose
    
    # Construct generalized inverse
    ginv = R @ AC.T @ L
    return ginv

def _flatten_vars(extra):
    """
    Flatten nested lists/tuples/sets/dicts into a simple list.
    Examples:
        _flatten_vars(['x', ['y','z']]) -> ['x', 'y', 'z']
        _flatten_vars('x') -> ['x']
    """
    if extra is None:
        return []
    if isinstance(extra, str):
        return [extra]
    if isinstance(extra, dict):
        return list(extra.keys())
    flat = []
    for x in extra:
        if isinstance(x, (list, tuple, set)):
            flat.extend(list(x))
        else:
            flat.append(x)
    return flat


def _safe_div(numerator, denominator):
    """
    Safe division usable with either column names (str) or Polars expressions.
    Returns numerator/denominator when valid, otherwise None.
    """
    num_expr = numerator if isinstance(numerator, pl.Expr) else pl.col(numerator)
    den_expr = denominator if isinstance(denominator, pl.Expr) else pl.col(denominator)
    return pl.when((den_expr != 0) & den_expr.is_not_null()) \
             .then(num_expr / den_expr) \
             .otherwise(None)


def _replace_nulls(df, cols=None, value=0):
    """
    Replace null values in specified columns with a given value (default 0).
    Safely ignores missing columns or empty lists.
    """
    if not cols:
        return df
    existing = [c for c in cols if c in df.columns]
    if existing:
        df = df.with_columns([pl.col(c).fill_null(value).alias(c) for c in existing])
    return df


# ------------------------------------------------------------
# 2️⃣ — COLUMN CONTROL & VALIDATION HELPERS
# ------------------------------------------------------------

def _ensure_columns(df: pl.DataFrame, cols, dtype=pl.Float64):
    """
    Ensure each column in cols exists in df. If missing, create it as NULLs.
    """
    for c in cols:
        if c not in df.columns:
            df = df.with_columns(pl.lit(None).cast(dtype).alias(c))
    return df


def _drop_temp(df: pl.DataFrame, *cols):
    """
    Drop temporary columns if they exist (safe drop).
    """
    existing = [c for c in cols if c in df.columns]
    if existing:
        df = df.drop(existing)
    return df


def _check_columns_exist(df, cols):
    """
    Raise an error if any required column is missing.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Required columns are missing: {missing}")


# ------------------------------------------------------------
# 3️⃣ — GROUPED AGGREGATION HELPERS
# ------------------------------------------------------------

def _group_weighted_mean(df, group_cols, value_col, weight_col="weight_XX"):
    """
    Compute a grouped weighted mean safely using _safe_div.
    Returns DataFrame with group columns and '{value_col}_wmean'.
    """
    weighted_sum = (pl.col(value_col) * pl.col(weight_col)).sum()
    total_weight = pl.col(weight_col).sum()
    return (
        df.filter(pl.col(value_col).is_not_null())
          .group_by(group_cols)
          .agg(_safe_div(weighted_sum, total_weight).alias(f"{value_col}_wmean"))
    )


def _group_sum(df, group_cols, value_col, alias=None):
    """
    Sum value_col by group_cols. Returns aggregated DataFrame.
    """
    alias = alias or f"sum_{value_col}"
    return df.group_by(group_cols).agg(pl.col(value_col).sum().alias(alias))


# ------------------------------------------------------------
# 4️⃣ — DIAGNOSTIC / WARNING HELPERS
# ------------------------------------------------------------

def _warn_missing_cols(df, cols):
    """
    Warn if any requested column is missing from the DataFrame.
    """
    missing = [c for c in cols if c not in df.columns]
    if missing:
        warnings.warn(f"Missing columns detected: {missing}", UserWarning)


def _warn_once(msg):
    """
    Emit a single warning pointing to the caller location.
    """
    warnings.warn(msg, stacklevel=2)


def _describe_polars(df: pl.DataFrame, max_rows=5):
    """
    Quick overview of a Polars DataFrame: prints columns, dtypes, shape, and sample.
    """
    print("🔹 Shape:", df.shape)
    print("🔹 Columns:", df.columns)
    print("🔹 Dtypes:", df.dtypes)
    print("🔹 Preview:")
    print(df.head(max_rows))


# ------------------------------------------------------------
# 5️⃣ — DYNAMIC VARIABLE HELPERS
# ------------------------------------------------------------

def make_var(prefix, i=None, suffix="_XX"):
    """
    Build a dynamic variable name.
    Examples:
        make_var("E_hat_gt", 2) -> "E_hat_gt_2_XX"
        make_var("sample") -> "sample_XX"
    """
    if i is None:
        return f"{prefix}{suffix}"
    return f"{prefix}_{i}{suffix}"


def add_indexed_column(df, base_col, i, func):
    """
    Create a new indexed column from base_col applying func(pl.col()).
    Example:
        df = add_indexed_column(df, "y", 3, lambda c: c**2)
    """
    new_col = f"{base_col}_{i}_XX"
    df = df.with_columns(func(pl.col(base_col)).alias(new_col))
    return df


