"""
Pytest fixtures for did_multiplegt_dyn tests.
"""

import pytest
import pandas as pd
import polars as pl
from pathlib import Path


@pytest.fixture(scope="module")
def data_dir():
    """Return path to test data directory."""
    return Path(__file__).parent / "data"


@pytest.fixture(scope="module")
def wagepan_pandas(data_dir):
    """Load wagepan dataset as pandas DataFrame."""
    return pd.read_stata(data_dir / "wagepan.dta")


@pytest.fixture(scope="module")
def wagepan_polars(wagepan_pandas):
    """Convert wagepan dataset to Polars DataFrame."""
    return pl.from_pandas(wagepan_pandas)


@pytest.fixture
def baseline_args():
    """Default arguments for baseline estimation."""
    return dict(
        outcome="lwage",
        group="nr",
        time="year",
        treatment="union",
        effects=5,
        placebo=2
    )


@pytest.fixture
def controls_args(baseline_args):
    """Arguments with controls."""
    args = baseline_args.copy()
    args["controls"] = ["hours"]
    return args


@pytest.fixture
def cluster_args(baseline_args):
    """Arguments with clustering."""
    args = baseline_args.copy()
    args["cluster"] = "hisp"
    return args


@pytest.fixture
def weight_args(baseline_args):
    """Arguments with weights."""
    args = baseline_args.copy()
    args["weight"] = "educ"
    return args


@pytest.fixture
def normalized_args(baseline_args):
    """Arguments for normalized estimation."""
    args = baseline_args.copy()
    args["normalized"] = True
    return args


@pytest.fixture
def trends_args(baseline_args):
    """Arguments with nonparametric trends."""
    args = baseline_args.copy()
    args["trends_nonparam"] = ["black"]
    return args


@pytest.fixture
def continuous_args(baseline_args):
    """Arguments for continuous treatment."""
    args = baseline_args.copy()
    args["continuous"] = 1
    return args


# Test model specifications for comprehensive testing
TEST_MODELS = {
    "Baseline": {},
    "Placebos": {},  # Uses default placebo=2
    "Normalized": {"normalized": True},
    "Controls": {"controls": ["hours"]},
    "Trends_Nonparam": {"trends_nonparam": ["black"]},
    "Trends_Lin": {"trends_lin": True},
    "Continuous": {"continuous": 1},
    "Weight": {"weight": "educ"},
    "Cluster": {"cluster": "hisp"},
    "Same_Switchers": {"same_switchers": True},
    "Same_Switchers_Placebo": {"same_switchers": True, "same_switchers_pl": True},
    "Switchers_In": {"switchers": "in"},
    "Switchers_Out": {"switchers": "out"},
    "Only_Never_Switchers": {"only_never_switchers": True},
    "CI_Level_90": {"ci_level": 90},
    "CI_Level_99": {"ci_level": 99},
    "Less_Conservative_SE": {"less_conservative_se": True},
    "Bootstrap": {"bootstrap": (20, 1234)},  # 20 reps, seed 1234 (matching Stata)
    "Dont_Drop_Larger_Lower": {"dont_drop_larger_lower": True},
    "Effects_Equal": {"effects_equal": True},
    "By_Black": {"by": "black"},  # Stratify by black variable
}


@pytest.fixture(params=TEST_MODELS.keys())
def model_name(request):
    """Parametrized fixture for model names."""
    return request.param


@pytest.fixture
def model_args(model_name, baseline_args):
    """Get arguments for a specific model."""
    args = baseline_args.copy()
    args.update(TEST_MODELS[model_name])
    return args
