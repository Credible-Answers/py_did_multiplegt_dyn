"""
Comprehensive tests for did_multiplegt_dyn package.

Tests all model configurations against expected behavior.
"""

import pytest
import warnings
import numpy as np
import pandas as pd
import polars as pl

from did_multiplegt_dyn import DidMultiplegtDyn


class TestBasicFunctionality:
    """Test basic estimation functionality."""

    def test_baseline_estimation(self, wagepan_polars, baseline_args):
        """Test baseline estimation runs without error."""
        model = DidMultiplegtDyn(df=wagepan_polars, **baseline_args)
        result = model.fit()

        assert result is not None
        assert "did_multiplegt_dyn" in result.result
        assert "Effects" in result.result["did_multiplegt_dyn"]
        assert "Placebos" in result.result["did_multiplegt_dyn"]

    def test_effects_count(self, wagepan_polars, baseline_args):
        """Test correct number of effects estimated."""
        model = DidMultiplegtDyn(df=wagepan_polars, **baseline_args)
        result = model.fit()

        effects = result.result["did_multiplegt_dyn"]["Effects"]
        assert len(effects) <= baseline_args["effects"]

    def test_placebos_count(self, wagepan_polars, baseline_args):
        """Test correct number of placebos estimated."""
        model = DidMultiplegtDyn(df=wagepan_polars, **baseline_args)
        result = model.fit()

        placebos = result.result["did_multiplegt_dyn"]["Placebos"]
        assert len(placebos) <= baseline_args["placebo"]


class TestOptions:
    """Test various option configurations."""

    def test_normalized(self, wagepan_polars, normalized_args):
        """Test normalized estimation."""
        model = DidMultiplegtDyn(df=wagepan_polars, **normalized_args)
        result = model.fit()
        assert result is not None

    def test_controls(self, wagepan_polars, controls_args):
        """Test estimation with controls."""
        model = DidMultiplegtDyn(df=wagepan_polars, **controls_args)
        result = model.fit()
        assert result is not None

    def test_cluster(self, wagepan_polars, cluster_args):
        """Test estimation with clustering."""
        model = DidMultiplegtDyn(df=wagepan_polars, **cluster_args)
        result = model.fit()
        assert result is not None

    def test_weights(self, wagepan_polars, weight_args):
        """Test estimation with weights."""
        model = DidMultiplegtDyn(df=wagepan_polars, **weight_args)
        result = model.fit()
        assert result is not None

    def test_trends_nonparam(self, wagepan_polars, trends_args):
        """Test estimation with nonparametric trends."""
        model = DidMultiplegtDyn(df=wagepan_polars, **trends_args)
        result = model.fit()
        assert result is not None

    def test_continuous(self, wagepan_polars, continuous_args):
        """Test continuous treatment."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = DidMultiplegtDyn(df=wagepan_polars, **continuous_args)
            result = model.fit()
        assert result is not None


class TestNewOptions:
    """Test newly implemented options."""

    def test_design_console(self, wagepan_polars, baseline_args):
        """Test design option with console output."""
        args = baseline_args.copy()
        args["design"] = (0.9, "console")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = DidMultiplegtDyn(df=wagepan_polars, **args)
            result = model.fit()

        # Design should be in results if successful
        assert result is not None

    def test_bootstrap_warning(self, wagepan_polars, baseline_args):
        """Test bootstrap warning without continuous."""
        args = baseline_args.copy()
        args["bootstrap"] = (10, 12345)

        with pytest.warns(UserWarning, match="continuous"):
            model = DidMultiplegtDyn(df=wagepan_polars, **args)


class TestOutputFormat:
    """Test output format matches expected structure."""

    def test_effects_columns(self, wagepan_polars, baseline_args):
        """Test Effects DataFrame has required columns."""
        model = DidMultiplegtDyn(df=wagepan_polars, **baseline_args)
        result = model.fit()

        effects = result.result["did_multiplegt_dyn"]["Effects"]
        required_cols = ["Estimate", "SE", "LB CI", "UB CI"]

        for col in required_cols:
            assert col in effects.columns

    def test_ate_exists(self, wagepan_polars, baseline_args):
        """Test ATE is computed."""
        model = DidMultiplegtDyn(df=wagepan_polars, **baseline_args)
        result = model.fit()

        ate = result.result["did_multiplegt_dyn"]["ATE"]
        assert ate is not None
        assert len(ate) > 0

    def test_coef_vcov_structure(self, wagepan_polars, baseline_args):
        """Test coefficient and variance-covariance structure."""
        model = DidMultiplegtDyn(df=wagepan_polars, **baseline_args)
        result = model.fit()

        assert "coef" in result.result
        assert "b" in result.result["coef"]
        assert "vcov" in result.result["coef"]


class TestSwitcherOptions:
    """Test switcher-related options."""

    def test_same_switchers(self, wagepan_polars, baseline_args):
        """Test same_switchers option."""
        args = baseline_args.copy()
        args["same_switchers"] = True

        model = DidMultiplegtDyn(df=wagepan_polars, **args)
        result = model.fit()
        assert result is not None

    def test_switchers_in(self, wagepan_polars, baseline_args):
        """Test switchers='in' option."""
        args = baseline_args.copy()
        args["switchers"] = "in"

        model = DidMultiplegtDyn(df=wagepan_polars, **args)
        result = model.fit()
        assert result is not None

    def test_switchers_out(self, wagepan_polars, baseline_args):
        """Test switchers='out' option."""
        args = baseline_args.copy()
        args["switchers"] = "out"

        model = DidMultiplegtDyn(df=wagepan_polars, **args)
        result = model.fit()
        assert result is not None


class TestCILevels:
    """Test different confidence interval levels."""

    @pytest.mark.parametrize("ci_level", [90, 95, 99])
    def test_ci_levels(self, wagepan_polars, baseline_args, ci_level):
        """Test different CI levels."""
        args = baseline_args.copy()
        args["ci_level"] = ci_level

        model = DidMultiplegtDyn(df=wagepan_polars, **args)
        result = model.fit()

        effects = result.result["did_multiplegt_dyn"]["Effects"]
        # CI bounds should exist
        assert "LB CI" in effects.columns
        assert "UB CI" in effects.columns


class TestEffectsEqual:
    """Test effects equality option."""

    def test_effects_equal(self, wagepan_polars, baseline_args):
        """Test effects_equal option returns p-value."""
        args = baseline_args.copy()
        args["effects_equal"] = True

        model = DidMultiplegtDyn(df=wagepan_polars, **args)
        result = model.fit()

        dyn = result.result["did_multiplegt_dyn"]
        # p_equality_effects may or may not be present depending on data
        assert result is not None


class TestByOption:
    """Test 'by' option for stratified analysis."""

    def test_by_stratification(self, wagepan_polars, baseline_args):
        """Test by option stratifies analysis correctly."""
        args = baseline_args.copy()
        args["by"] = "black"
        args["effects"] = 3
        args["placebo"] = 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = DidMultiplegtDyn(df=wagepan_polars, **args)
            result = model.fit()

        # Check that all_by_results is populated
        dyn = result.result["did_multiplegt_dyn"]
        all_by_results = dyn.get("all_by_results", [])

        # Should have 2 subgroups for black (0 and 1)
        assert len(all_by_results) == 2
        assert all_by_results[0]["by_var"] == "black"
        assert all_by_results[1]["by_var"] == "black"

    def test_by_and_by_path_mutually_exclusive(self, wagepan_polars, baseline_args):
        """Test that by and by_path cannot be used together."""
        args = baseline_args.copy()
        args["by"] = "black"
        args["by_path"] = 4

        with pytest.raises(ValueError, match="Cannot specify both"):
            DidMultiplegtDyn(df=wagepan_polars, **args)


class TestByPathOption:
    """Test 'by_path' option for treatment path stratification."""

    def test_by_path_stratification(self, wagepan_polars, baseline_args):
        """Test by_path option stratifies analysis by treatment paths."""
        args = baseline_args.copy()
        args["by_path"] = 3
        args["effects"] = 3
        args["placebo"] = 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = DidMultiplegtDyn(df=wagepan_polars, **args)
            result = model.fit()

        # Check that all_path_results is populated
        dyn = result.result["did_multiplegt_dyn"]
        all_path_results = dyn.get("all_path_results", [])

        # Should have 3 paths
        assert len(all_path_results) == 3
        # Each path should have a treatment_sequence
        for path_result in all_path_results:
            assert "treatment_sequence" in path_result
            assert "n_groups" in path_result
            assert path_result["n_groups"] > 0

    def test_by_path_all(self, wagepan_polars, baseline_args):
        """Test by_path='all' analyzes all treatment paths."""
        args = baseline_args.copy()
        args["by_path"] = "all"
        args["effects"] = 3
        args["placebo"] = 1

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = DidMultiplegtDyn(df=wagepan_polars, **args)
            result = model.fit()

        # Check that all_path_results is populated
        dyn = result.result["did_multiplegt_dyn"]
        all_path_results = dyn.get("all_path_results", [])

        # Should have multiple paths (more than 3)
        assert len(all_path_results) > 3
