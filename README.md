# py_did_multiplegt_dyn

This is the Python implementation of the Stata package [did_multiplegt_dyn](https://github.com/chaisemartinPackages/did_multiplegt_dyn/tree/main/Stata).


[![PyPI version](https://img.shields.io/pypi/v/py-did-multiplegt-dyn.svg?color=blue)](https://pypi.org/project/py-did-multiplegt-dyn/)
[![Downloads](https://static.pepy.tech/personalized-badge/py-did-multiplegt-dyn?period=total&units=international_system&left_color=blue&right_color=grey&left_text=Downloads)](https://pepy.tech/project/py-did-multiplegt-dyn)
[![Last commit](https://img.shields.io/github/last-commit/Credible-Answers/py_did_multiplegt_dyn.svg)](https://github.com/Credible-Answers/py_did_multiplegt_dyn/commits/main)
[![GitHub stars](https://img.shields.io/github/stars/Credible-Answers/py_did_multiplegt_dyn.svg?style=social)](https://github.com/Credible-Answers/py_did_multiplegt_dyn/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/Credible-Answers/py_did_multiplegt_dyn.svg)](https://github.com/Credible-Answers/py_did_multiplegt_dyn/issues)
[![License](https://img.shields.io/github/license/Credible-Answers/py_did_multiplegt_dyn.svg)](https://github.com/Credible-Answers/py_did_multiplegt_dyn/blob/main/LICENSE)



> See [Using did_multiplegt_dyn to Estimate Event-Study Effects in Complex Designs](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5337463) for worked examples covering non-absorbing binary treatment, continuous absorbing treatment, multivalued treatment with repeated changes, and designs with two binary absorbing treatments.



## DidMultiplegtDyn
Estimation in staggered first switch designs, where groups experience their first treatment change at different points in time.

[Installation](#installation) | [Syntax](#syntax) | [Description](#description)

[Options](#options) | [Example](#example) | [FAQ](#faq) | [References](#references) | [Authors](#authors)

## What's new in 0.1.8

Version 0.1.8 adds the Stata-compatible `reset` and `avg_time_periods` options.
`reset` makes long-panel control groups reusable after a configurable treatment
spell, while `avg_time_periods` reports how long incremental treatment effects
can accumulate. See the [changelog](https://github.com/Credible-Answers/py_did_multiplegt_dyn/blob/main/CHANGELOG.md)
for the complete release notes.

## Installation

```bash
python -m pip install py-did-multiplegt-dyn
```

The estimator currently expects `df` to be a [Polars](https://pola.rs/) `DataFrame`.

## Syntax

```python
from did_multiplegt_dyn import DidMultiplegtDyn

model = DidMultiplegtDyn(
    df=df,
    outcome="outcome_column",
    group="group_column",
    time="time_column",
    treatment="treatment_column",
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
    by_path=None,
    design=None,
    by=None,
    normalized_weights=False,
    predict_het_hc2bm=False,
    date_first_switch=None,
    reset=0,
    avg_time_periods=False,
)

model.fit()
model.summary()
model.plot()
```

## Description

**did_multiplegt_dyn** computes the heterogeneity-robust DID event-study estimators introduced in de Chaisemartin and D'Haultfoeuille (2026). Like other recently proposed DID estimation commands (**csdid**, **didimputation**, ...), **did_multiplegt_dyn** can be used with a binary and staggered (absorbing) treatment. But unlike those other commands, **did_multiplegt_dyn** can also be used if the treatment is non-binary (discrete or continuous) and/or non-absorbing (the treatment can increase or decrease multiple times). It is applicable to any "staggered first switch design", where groups experience their first treatment change at different points in time. Lagged treatments may affect the outcome, and the current and lagged treatments may have heterogeneous effects, across space and/or over time. The event-study estimators computed by the command compare the outcome evolutions of switchers, namely units that experience a change in their treatment, and of not-yet-switchers, namely units whose treatment has not changed yet. Those estimators rely on a no-anticipation and parallel-trends assumptions, which can be partly tested by computing pre-trend estimators. The panel may be unbalanced: not all groups have to be observed at every period. The data may also be at a more disaggregated level than the group level (e.g. individual-level wage data to measure the effect of a regional-level minimum-wage on individuals' wages). See Section 8.3 of "Causal Inference with Differences-in-Differences: Credible Answers to Hard Questions" by Chaisemartin and D'Haultfoeuille for a thorough presentation of the estimators computed by the command.

**Y** is the outcome variable.

**G** is the group variable, which identifies the panel's cross-sectional units (e.g.: counties, municipalities...)

**T** is the time period variable. The command assumes that the time variable is evenly spaced (for example, annual data with no years missing for all groups). However, it can also be used when some time periods are missing for all groups (for example, annual data with three consecutive years missing from the panel for all groups). See the FAQ section below for details.

**D** is the treatment variable.

### Further Detail

**Non-normalized event-study estimators (default)** Intuitively, those effects compare groups' outcomes under their actual treatment path to what their outcome would have been under the status-quo path where they would have kept their period-one treatment throughout the panel. Formally, for all "switchers", namely groups that experience a change of their treatment over the study period, let $F_g$ denote the first time period when g's treatment changes. The command computes the non-normalized event-study estimators $DID_\ell$. $DID_1$ is the average, across all switchers, of DID estimators comparing the $F_{g}-1$ to $F_{g}$ outcome evolution of g to that of groups with the same period-one treatment as g but whose treatment has not changed yet at $F_{g}$. More generally, $DID_\ell$ is the average, across all switchers, of DID estimators comparing the $F_{g}-1$ to $F_{g}-1+\ell$ outcome evolution of g to that of groups with the same period-one treatment as g but whose treatment has not changed yet at $F_{g}-1+\ell$. Those estimators are unbiased for non-normalized event-study effects, which are average effects of having been exposed to a weakly higher treatment dose for $\ell$ periods. However, the magnitude and timing of the incremental treatment doses received under the actual treatment path relative to the status-quo path can vary across groups, so non-normalized effects can generally not be interpreted as effects of a one unit increase in the treatment.

**Normalized event-study estimators** The command also computes the normalized event-study estimators $DID^n_\ell$, that normalize $DID_\ell$ by the average of the sum of the incremental treatment doses received by switchers under their actual path, relative to the doses they would have received under their status-quo path. This normalization ensures that $DID^n_\ell$ estimates a weighted average of the effects of the current treatment and of its $\ell-1$ first lags on the outcome. Thus, normalized effects can be interpreted as effects of a one unit increase in the treatment. While the effects of the current and lagged treatments cannot be separately estimated, the weight that $DID^n_\ell$ puts on the effect of each lag can be estimated.

**Average cumulative (total) effect per dose** The command also computes an estimated average cumulative (total) effect per unit of treatment, where "cumulative effect" refers to the sum of the effects of a treatment dose, at the time when it takes place and at later periods, see Section 3.3 of de Chaisemartin and D'Haultfoeuille (2026) for further details. The command also shows the number of time periods over which the effect of a dose is accumulated, on average across all incremental doses received by switchers over the study period. By dividing the average cumulative effect by the average number of periods across which effects are accumulated, one can get an estimator of the effect of being exposed to one more unit of treatment for one period.

**Placebos** The command also computes placebo estimators, that average DIDs comparing the outcome evolution of switcher g and of its control groups, from $F_{g}-1$ to $F_{g}-1-\ell$, namely before g's treatment changes for the first time. Those placebos can be used to test the parallel trends and no-anticipation assumptions under which the estimators computed by **did_multiplegt_dyn** are unbiased.

**Designs compatible with the command** The command can be used in staggered first switch designs, where groups experience their first treatment change at different points in time. Such designs encompass the canonical binary and absorbing treatment case. But they also encompass more complicated designs: groups may have heterogeneous treatments at period one, their treatment may change at different dates, some groups may experience increases in their treatment while other groups experience decreases, some groups may experience more than one change of their treatment, and finally some groups may experience larger treatment changes than others. The command can also be used to separately estimate the effects of several treatment variables, see references in the FAQ section. The only requirement is that not all groups experience their first treatment change at the same date.

**Relaxing the parallel-trends assumption** The command allows for many relaxations of the parallel-trends assumption: see the **controls** option for estimators allowing for time-varying covariates, see the **trends_lin** option for estimators allowing for group-specific linear trends, and see the **trends_nonparam** option for estimators allowing to interact time fixed effects with time-invariant variables (e.g. industry×year effect with firm-level panel data).

## Options

### Main Options

**effects = #**: gives the number of event-study effects to be estimated. With effects = 5, the command estimates event-study effects 1 through 5 periods after the first treatment change.

**normalized = True**: when this option is specified, the command estimates normalized event-study effects, that are equal to a weighted average of the effects of the current treatment and of its $\ell-1$ first lags on the outcome. See Section 3.2 of de Chaisemartin and D'Haultfoeuille (2026) for further details.

**normalized_weights = True**: when this option and the **normalized** option are specified, the command reports the weights that normalized effect $\ell$ puts on the effect of the current treatment, on the effect of the first treatment lag, etc.

**placebo = #**: gives the number of placebo estimators to be computed. Placebos compare the outcome evolution of switchers and of their controls, before switchers' treatment changes for the first time. Under the parallel trends and no-anticipation assumptions underlying the event-study estimators computed by **did_multiplegt_dyn**, the expectation of the placebos is equal to zero. Thus, placebos can be used to test those assumptions, by testing the null that all placebos are equal to zero. If the user requests that at least two placebos be estimated, the command computes the p-value of a joint test of that null hypothesis. The number of placebos requested can be at most equal to the number of time periods in the data minus 2, though most often only a smaller number of placebos can be computed. Also, the number of placebos requested cannot be larger than the number of effects requested.

**reset = #**: specifies that treatment spells are reset after # consecutive periods without a treatment change. When this option is used, the command partitions each original group into a sequence of subgroups. A new subgroup starts whenever the treatment has remained unchanged for # consecutive periods since the last treatment change. For example, *reset = 5* starts a new subgroup whenever a group's treatment has not changed for five consecutive periods. Treatment changes occurring after such a reset are treated as belonging to a new treatment spell. Thus, a group that has experienced its last treatment change more than # periods ago can again be used as a control group by the command. This option can be useful in long panels where all groups eventually experience a treatment change. Without that option, treatment effects can only be estimated until there is still one group that has never experienced a treatment change, while with this option it may be possible to estimate treatment effects throughout the panel. With this option, the estimators computed by the command allow for effects of the first # treatment lags on the outcome, but they assume that older lags do not affect the outcome (instead, without this option the command allows for effects of lagged treatments up to any lag). When this option is specified, standard errors remain clustered at the level of the original groups, or at a coarser level if the user specifies a coarser clustering variable in the **cluster** option. When this option is specified, the command restricts the estimation sample to groups whose treatment is observed at all dates.

**continuous = #**: allows to use the command even when groups' period-one treatment is continuous, meaning that all groups have a different period-one treatment value. With a discrete period-one treatment, the command compares the outcome evolution of switchers and non-switchers with the same period-one treatment. But with a truly continuous period-one treatment, there will be no two groups with the same period-one treatment. Then, the command assumes that groups' counterfactual outcome evolution if their treatment does not change is a polynomial in their period-one treatment. The user's chosen polynomial order is the option's argument. See Section 1.10 of the Web Appendix of de Chaisemartin and D'Haultfoeuille (2026) for further details. Unlike the other variance estimators computed by the command, those computed when the **continuous** option is specified are not backed by a proven asymptotic normality result. Preliminary simulation evidence indicates that when the option is used with a correctly-specified polynomial order, those variance estimators are conservative. On the other hand, when the specified polynomial order is strictly larger than needed, those variance estimators can become liberal. Thus, when this option is specified, we recommend using an external bootstrap procedure for inference. At least, one should perform a robustness check where one compares the analytic variance computed by the command to a bootstrapped variance. This option cannot be combined with the **design** option. This option only needs to be used when groups' period-one treatment is continuous: if all groups are initially untreated and then start receiving continuous treatment doses, using this option is unnecessary.

**same_switchers = True**: if this option is specified and the user requests that at least two event-study effects be estimated, the command will restrict the estimation of the effects to switchers for which all effects can be estimated, to avoid compositional changes.

**same_switchers_pl = True**: this option can be specified when **same_switchers** is also specified. Then, the placebos are estimated only for switchers for which all the requested effects and placebos can be estimated.

**only_never_switchers = True**: if this option is specified, the command estimates the event-study effects using only never-switchers as control units, instead of using all not-yet-switchers (a larger control group than just never-switchers).

**avg_time_periods = True**: if this option is specified, the command reports the average number of time periods over which the effect of a treatment dose is cumulated. Each time a switcher group receives an incremental dose of treatment relative to its baseline, the effect of that dose is tracked from the period it is received until the last period for which a valid control group exists for that switcher. Groups that can be followed for longer, or that receive more doses, contribute more to this average. The result is stored in `model.result["did_multiplegt_dyn"]["avg_cumul"]`.

### Options to understand and leverage your design

**design = (*float*, *string*)**: this option reports switchers' period-one and subsequent treatments, thus helping the analyst understand the treatment paths whose effect is aggregated in the non-normalized event-study effects. When the number of treatment paths is low, or when there are paths shared by a reasonably large number of switchers, one may consider estimating treatment-path-specific event-study effects, using the **by_path** option. When the number of treatment paths is large, one may specify a number included between 0 and 1 in the *float* argument. Then the command reports the treatment paths common to at least (*float* × 100)% of switchers. Results can be printed in the Python console specifying `"console"` as the string argument. For example, *DidMultiplegtDyn(..., effects = 5, design = (0.5, "console"))* reports the treatment paths experienced by at least 50% of the switchers and prints the output in the Python console. Alternatively, the output can be stored in an Excel file providing a valid file path as the string argument.

**by_path = #**: when this option is specified, the command estimates all the effects separately for the # most common treatment paths from $F_{g-1}$ to $F_{g-1+\ell}$, where $\ell$ is the argument inputted to the *effects* option. If you want to estimate effects separately for all treatment paths, you can input `by_path = "all"`. This option can not be combined with the **by** option. For instance, with a binary and non-absorbing treatment, it may be interesting to estimate event-study effects separately for groups experiencing a 01000... path, for groups experiencing a 011000... path, for groups experiencing a 0111000... path, etc. This analysis can shed light on whether treatment effects vary with the number of periods of exposure to treatment.

**switchers = *string***: one may be interested in estimating separately the treatment effect of switchers-in, whose treatment after they switch is larger than their period-one treatment, and of switchers-out, whose treatment after they switch is lower than their period-one treatment. In that case, one should run the command first with the **switchers = "in"** option, and then with the **switchers = "out"** option.

**date_first_switch = (*by_baseline_treat*, *string*)**: the option reports the dates at which switchers experience their first treatment change, and how many groups experienced a first change at each date. The reference population are switchers for which the last event-study effect can be estimated. If `True` is specified as the first argument, separate tables are displayed for each level of the period-one treatment. Results can be printed in the Python console specifying `"console"` in the second argument. Alternatively, the output can be stored in an Excel file providing a valid file path in the second argument.

### Options to relax the parallel-trends assumption

Before describing the command's options to include covariates in the estimation, let us emphasize that there is evidence that when they choose their control variables, researchers engage in p-hacking: they choose the covariates that make their event-study estimates more significant, rather than choosing those that make the parallel-trends assumption more plausible. Thus, while DID estimators with control variables rely, in principle, on a weaker identifying assumption than DID estimators without controls, in practice they should be considered as less reliable. Accordingly, we recommend that by default, researchers do not include any control variable in their estimation. When pre-trend coefficients are precisely estimated and not significantly different from zero, there is no reason to include control variables in the estimation. On the other hand, when pre-trend estimators are smaller, less significant, and/or more precisely estimated with than without control variables, then it may make sense to include some control variables. See Section 4.1 of "Causal Inference with Differences-in-Differences: Credible Answers to Hard Questions" by Chaisemartin and D'Haultfoeuille for further details.

**controls = [*varlist*]**: gives the names of the control variables to be included in the estimation. Estimators with controls are similar to those without controls, except that the first-difference of the outcome is replaced by residuals from regressions of the first-difference of the outcome on the first-differences of the controls and time fixed effects. Those regressions are estimated in the sample of control $(g,t)$s: $(g,t)$s such that group g's treatment has not changed yet at t. Those regressions are also estimated separately for each value of the period-one treatment. Estimators with controls are unbiased even if groups experience differential trends, provided such differential trends can be fully explained by a linear model in covariates changes. To control for time-invariant covariates, one can for instance input the product of those covariates and of the time variable **T** into the option. See Section 1.2 of the Web Appendix of de Chaisemartin and D'Haultfoeuille (2026) for further details.

**trends_lin = True**: when this option is specified, the estimation of the treatment effects allows for group-specific linear trends. Estimators with linear trends start by computing event-study effects on the outcome's first-difference, rather than on the outcome itself, thus allowing for group-specific linear trends. Then, to recover event-study effect $\ell$ on the outcome, event-study effects on the outcome's first-difference are summed from 1 to $\ell$. See Section 1.3 of the Web Appendix of de Chaisemartin and D'Haultfoeuille (2026) for further details. When this option is specified, the estimated average cumulative (total) effect per unit of treatment is not computed.

**trends_nonparam = [*varlist*]**: when this option is specified, the DID estimators computed by the command only compare switchers to not-yet-switchers with the same period-one treatment and with the same value of *varlist*. Estimators with the **trends_nonparam = [*varlist*]** option are unbiased even if groups experience differential trends, provided all groups with the same value of *varlist* experience parallel trends. *varlist* can only include time-invariant variables, and the interaction of those variables has to be coarser than the group variable. For instance, if one works with a county×year data set and one wants to allow for state-specific trends, one should specify **trends_nonparam = ["state"]**, where state is the state identifier. Similarly, if one works with a firm×year data and one wants to allow for industry-specific trends, one should specify **trends_nonparam = ["industry"]**. See Section 1.4 of the Web Appendix of de Chaisemartin and D'Haultfoeuille (2026) for further details.

### Options for standard errors and confidence intervals

**cluster = *varname***: can be used to cluster the estimators' standard errors. Only one clustering variable is allowed. A common practice in DID analysis is to cluster standard errors at the group level. Such clustering is implemented by default by the command. Standard errors can be clustered at a more aggregated level than the group level, but they cannot be clustered at a more disaggregated level.

**ci_level = #**: with this option, one can change the level of the confidence intervals shown in the output tables and on the graph. The default value is 95, thus yielding 95% level confidence intervals.

**more_granular_demeaning = True**: when groups' treatment can change multiple times, the standard errors reported by default by the command may be conservative. Then, standard errors that may be less conservative when the sample size is large enough can be obtained by specifying this option. See de Chaisemartin et al. (2025) for further details.

### Options to investigate heterogeneous effects

**by = *varname***: when this option is specified, the command estimates all the effects separately by the levels of *varname*, a group-level and time-invariant variable. If *varname* is a binary variable for example, then the estimation is carried out once for groups with *varname*=0 and once for groups with *varname*=1. Then, the command reports on a graph event-study plots for all values of *varname*, thus allowing to assess effect heterogeneity by *varname*.

**predict_het = [*varlist*, *numlist*]**: when this option is specified, the command outputs tables showing whether the group-level and time-invariant variables in *varlist* predict groups' estimated event-study effects. By default, with this option the command produces one table per event-study effect estimated, each displaying the coefficients from regressions of the group-level estimate of the event-study effect on the variables in *varlist*. This method to analyze heterogeneous treatment effects assumes that switchers' counterfactual outcome evolutions is uncorrelated with the variables in *varlist*. To placebo test this condition, the command also shows placebo regression tables, where switchers' outcome evolutions before their treatment changed is regressed on the covariates. The p-value of a test that all coefficients are equal to zero is shown below each table. If you are interested in predicting all the event-study effects estimated, you can specify `"all"` as the option's second argument, instead of *numlist*. This option cannot be specified together with **normalized** or **controls**. See Section 1.5 of the Web Appendix of de Chaisemartin and D'Haultfoeuille (2026) for further details.

**predict_het_hc2bm = True**: when this option is specified with the **predict_het** option, the command computes HC2 standard errors that allow for intragroup correlation within groups defined by the variable specified in the **cluster** option. Degrees of freedom are adjusted following Bell and McCaffrey (2002) based on the variable specified in the **cluster** option. If no variable is specified in **cluster**, it will be clustered at the (g) level.

**effects_equal = True**: when this option is specified, the command performs an F-test to determine whether all estimated effects are equal.

### Other options

**weight = *varname***: specifies the name of a variable used to weight the data. For instance, if one works with a district×year data set and one wants to weight the estimation by each district×year's population, one should specify **weight = "population"**. If the data set is at a more disaggregated level than group×time, the command aggregates it at the group×time level internally, and weights the estimation by the number of observations in each group×time cell if the weight option is not specified, or by the sum of the weights of the observations in each group×time cell if the weight option is specified.

**save_results = *path***: if this option is specified, the command saves the estimators requested, their standard error, their 95% confidence interval, and the number of observations used in the estimation in a separate data set, at the location specified in *path*.

**dont_drop_larger_lower = True**: by default, the command drops all the $(g,t)$ cells such that at $t$, group $g$ has experienced both a strictly larger and a strictly lower treatment than its period-one treatment. de Chaisemartin and D'Haultfoeuille (2026) show that dropping those cells is necessary to ensure that non-normalized event-study effects can be interpreted as effects of having been exposed to a weakly larger treatment for $\ell$ periods. The option **dont_drop_larger_lower** allows one to keep those cells.

**drop_if_d_miss_before_first_switch = True**: This option is relevant when the treatment of some groups is missing at some time periods. Then, the command imputes some of those missing treatments. Those imputations are detailed in Appendix A of de Chaisemartin et al (2025). In designs where groups' treatments can change at most once, all those imputations are justified by the design. In other designs, some of those imputations may be liberal. **drop_if_d_miss_before_first_switch** can be used to overrule liberal imputations that are not innocuous for the non-normalized event-study estimators. See Appendix A of de Chaisemartin et al (2025) for further details.

## Example

This example is estimating the effect of banking deregulations on loans volume, using the data of Favara and Imbs (2015)

### Install

```python
!pip install py-did-multiplegt-dyn
```

Estimating eight non-normalized event-study effects and three placebo effects of banking
deregulations on loans volume:

```python
import pandas as pd
import polars as pl
from did_multiplegt_dyn import DidMultiplegtDyn

favara_imbs = pl.read_csv('https://raw.githubusercontent.com/Credible-Answers/py_did_multiplegt_dyn/main/data/favara_imbs_did_multiplegt_dyn.csv',
                 ignore_errors = True)
model1 = DidMultiplegtDyn(
    df=favara_imbs,
    outcome = "Dl_vloans_b",
    group = "county",
    time = "year",
    treatment = "inter_bra",
    effects = 8,
    placebo = 3,
    cluster = "state_n",
    normalized = True,
    same_switchers = True,
    effects_equal = True
)
model1.fit()
```

The two additional Stata-compatible options can be requested directly in the constructor:

```python
model_with_reset = DidMultiplegtDyn(
    df=favara_imbs,
    outcome="Dl_vloans_b",
    group="county",
    time="year",
    treatment="inter_bra",
    effects=8,
    placebo=3,
    reset=5,
    avg_time_periods=True,
)
model_with_reset.fit()
model_with_reset.summary()

avg_cumul = model_with_reset.result["did_multiplegt_dyn"]["avg_cumul"]
effect_1_switchers = model_with_reset.result["did_multiplegt_dyn"]["N_switch_avg_1"]
```

`reset=5` starts a new subgroup after five consecutive periods without a treatment change. `avg_time_periods=True` does not change the event-study estimates; it adds the Stata-compatible accumulation diagnostic to the returned results and summary.

```python
model1_sum = model1.summary()
```

| Block                |    Estimate |         SE |        LB CI |     UB CI |    N |   Switchers |   N.w |   Switchers.w |
|:---------------------|------------:|-----------:|-------------:|----------:|-----:|------------:|------:|--------------:|
| Effect_1             |  0.0240393  | 0.0184241  | -0.0120712   | 0.0601498 | 3368 |         773 |  3368 |           773 |
| Effect_2             |  0.0116702  | 0.0106714  | -0.00924526  | 0.0325857 | 2604 |         773 |  2604 |           773 |
| Effect_3             |  0.0143443  | 0.00828821 | -0.00190033  | 0.0305889 | 2031 |         773 |  2031 |           773 |
| Effect_4             |  0.0115211  | 0.00819515 | -0.00454109  | 0.0275833 | 1541 |         773 |  1541 |           773 |
| Effect_5             |  0.0161627  | 0.00684205 |  0.00275253  | 0.0295729 | 1412 |         773 |  1412 |           773 |
| Effect_6             |  0.0133861  | 0.00625129 |  0.00113379  | 0.0256384 | 1293 |         773 |  1293 |           773 |
| Effect_7             |  0.00981002 | 0.00495722 |  9.40507e-05 | 0.019526  | 1248 |         773 |  1248 |           773 |
| Effect_8             |  0.0100838  | 0.00430933 |  0.0016377   | 0.01853   | 1248 |         773 |  1248 |           773 |
| Average_Total_Effect |  0.055063   | 0.0268082  |  0.00251994  | 0.107606  | 9861 |        6184 |  9861 |          6184 |
| Placebo_1            |  0.0303785  | 0.0264052  | -0.0213746   | 0.0821317 | 2335 |         764 |  2335 |           764 |
| Placebo_2            | -0.0217763  | 0.0253522  | -0.0714658   | 0.0279132 |  957 |         497 |   957 |           497 |
| Placebo_3            | -0.0137113  | 0.0263437  | -0.065344    | 0.0379214 |  511 |         358 |   511 |           358 |

```python
model1.plot()
```

![](images/model1_plot.png)



## FAQ

 > :question: *did_multiplegt_dyn does not output exactly the same results as did_multiplegt, is this normal?*

Yes, the two commands can sometimes output different results.  This is mostly due to different
    conventions in the way the two commands deal with missing values.  See Appendix B of de Chaisemartin et al
    (2024) for further details.

> :question: *Do I have to include group and time fixed effects as controls when using the did_multiplegt_dyn
    package?*

No, you do not have to.  Group and time fixed effects are automatically controlled for.

> :question: *My group-level panel is unbalanced: some groups (e.g. counties) are not observed in every year.
    Can I still use the command?*

You can. A frequent case of unbalancedness is when some groups are not observed over the full
    duration of the panel.  For instance, your data may be a yearly county-level panel from 1990 to
    2000, where some counties appear after 1990 while some exit before 2000.  Then, the command just
    redefines group's baseline treatment as their treatment at the first period when they are
    observed.

It may also be that some groups enter and exit the data multiple times.  For instance, you
    observe a county in 1990, 1991, 1994, 1996, and 2000. Then, the command may impute some of that
    county's missing treatments.  Those imputations are detailed in de Chaisemartin et al (2023a).
    In designs where groups' treatments can change at most once, all those imputations are justified
    by the design.  In other designs, some of those imputations may be liberal.
    **drop_if_d_miss_before_first_switch** can be used to overrule the potentially liberal imputations
    that are not innocuous for the non-normalized event-study estimators.  See de Chaisemartin et al
    (2023a) for further details.

Finally, it may also be the case that the data is fully missing at one or several time periods.
    For instance, you have data for 1990, 1991, and 1993, but 1992 is missing for every group.  Then,
    it is important to fill the gap in the data, as otherwise the estimation will assume that 1991
    and 1993 are as far apart as 1990 and 1991.  There are two ways of doing so.  First, you can
    append to your data a data set identical to your 1991 data, but with the year equal to 1992, and
    the outcome missing for every observation.  This is a conservative solution, where no first
    treatment change occurring between 1991 and 1993 will be used in the estimation, which may be
    reasonable because the year in which the change occurred is effectively unknown.  Second, you can
    append to your data a data set identical to your 1993 data, with the year equal to 1992, and the
    outcome missing for every observation.  Then, treatment changes occurring between 1991 and 1993
    will be used in the estimation, assuming they all took place between 1991 and 1992.

> :question: *Related to imbalanced panels, my outcomes (and potentially the control variables) are measured
    less frequently than the treatment.  For instance, the outcome is measured every two years, but I
    know the treatment of every group in every year.  How should I proceed?*

To fix ideas, let us first assume that the outcome is measured every two years, but you know the treatment of every group in every year. Then, you should split the sample into two subsamples, and run the command twice, one time on each of the subsamples. In the first estimation, you should include all group $\times$ time cells $(g,t)$ such that at $t$, $g$'s treatment has never changed since the start of the panel, and all $(g,t)$s such that (i) $g$'s treatment has changed at least once at $t$ and (ii) the change occurred at a period where the outcome is observed. Since the outcome is measured every two years, in that subsample the first event-study effect (denoted effect_1) is the effect of being exposed to a higher treatment for one period, the second effect (effect_2) is the effect of being exposed to a higher treatment for three periods, etc. In the second estimation, you should include all group $\times$ time cells $(g,t)$ such that at $t$, $g$'s treatment has never changed since the start of the panel, and all $(g,t)$s such that (i) $g$'s treatment has changed at least once at $t$ and (ii) the change occurred at a period where the outcome is not observed. In that subsample, the first event-study effect (denoted effect_1) is the effect of being exposed to a higher treatment for two periods, the second effect (effect_2) is the effect of being exposed to a higher treatment for four periods, etc. You may then combine the two sets of estimated effects into one event-study graph, with the only caveat that the "odd" and "even" effects are estimated on different subsamples. Importantly, the two estimations have to be run on a dataset at the same bi-yearly level as the outcome variable: the yearly level treatment information should only be used to select the relevant subsamples.

If the treatment is observed three times more often than the treatment, you can follow the same
    logic, splitting the sample into three subsamples and running the command three times, etc.

A short do file with a simple example where the treatment status is observed in each period while
    the outcome is only observed every second period can be found [here](https://drive.google.com/uc?export=download&id=1NBwfsFeNltU3XSOsORdthUW49LIezm1z).

> :question: *What is the maximum number of event-study effects I can estimate?*

With a balanced panel of groups, the maximum number of event-study effects one can estimate can be determined as follows. For each value of the period-one treatment $d$, start by computing the difference between the last period at which at least one group has had treatment $d$ since period 1, and the first period at which a group with treatment $d$ at period 1 changed its treatment. Add one to this difference. Then, the maximum number of event-study effects is equal to the maximum of the obtained values, across all values of the period-one treatment. With an unbalanced panel, this method can still be used to derive an upper bound of the maximum number of event-study effects one can estimate.

> :question: *How many control variables can I include in the estimation?*

Estimators with control variables are similar to those without controls, except that the
    first-difference of the outcome is replaced by residuals from regressions of the first-difference
    of the outcome on the first-differences of the controls and time fixed effects.  Those
    regressions are estimated in the sample of control $(g,t)$s:  $(g,t)$s such that group $g$'s treatment
    has not changed yet at period $t$.  Those regressions are also estimated separately for each value
    of the baseline treatment.  If the treatment takes values 0, 1, 2, 3, and 4, one regression is
    estimated for control $(g,t)$s with a treatment equal to 0, one regression is estimated for control
    (g,t)s with a treatment equal to 1, etc.  The number of control variables needs to be
    significantly smaller than the number of control $(g,t)$s in each of those regressions.  Otherwise,
    those regressions will overfit and produce noisy estimates.  If the number of observations is
    lower than the number of variables in one of those regressions, the command will run but will not
    take into account all the controls for all values of the baseline treatment.  An error message
    will let the user know that they are encountering this situation, and may thus want to reduce
    their number of control variables.

> :question: *In my application, groups' baseline treatment is a continuous variable, meaning that all groups
    have a different period-one treatment.  Therefore, Assumption 1 in de Chaisemartin and
    D'Haultfoeuille (2020a) fails. Can I still use the command?*

 Yes you can estimate non-normalized event-study effects.  Essentially, you just need to define a
    new treatment variable equal to 0 if g's treatment has never changed at t, to 1 if g's treatment
    has changed at t and g's period-t treatment is larger than its baseline treatment, and to -1 if
    g's treatment has changed at t and g's period-t treatment is lower than its baseline treatment.
    Then, you run the command with this new treatment, including interactions of period fixed effects
    and a polynomial in the baseline treatment as control variables.  For instance, if one wants to
    model the relationship between the counterfactual outcome trend and the baseline treatment as
    quadratic, and the data has 12 periods, one needs to include 22 variables as controls: the
    baseline treatment interacted with the period 2 to 12 fixed effects, and the baseline treatment
    squared interacted with the period 2 to 12 fixed effects.  See Section 1.5 of de Chaisemartin et
    al (2022) for further details. If groups' baseline treatment is not continuous but takes many
    values, pursuing this strategy may yield more precise estimators, applying to a larger number of
    switchers, than just running the command with the original treatment, at the expense of incurring
    a potential bias if the model for the counterfactual outcome trend is misspecified.

> :question: *My design is such that treatment is binary, and groups can enter the treatment, and then leave it
    once.  Can I use the command to separately estimate the effect of joining and leaving the
    treatment?*

 Yes you can. See Section 1.5 of the Web Appendix of de Chaisemartin and D'Haultfoeuille (2024)
    for further details.

> :question: *My design has several treatments that may all have dynamic effects.  Can I use the command to
    estimate the effect of a treatment controlling for other treatments?*

 Yes, if those treatments follow binary and staggered designs.  See Section 3.2 of the Web
    Appendix of de Chaisemartin and D'Haultfoeuille (2023) for further details.

> :question: *Can I perform triple difference-in-differences with the command?*

Yes. Suppose for instance your third difference is across men and women in the same $(g,t)$ cell.
    Then, for each $(g,t)$ cell, you just need to compute the difference between the average outcome of
    men and women in cell $(g,t)$.  Then, you simply run the command with this new outcome.

> :question: *Is it possible to compute the percentage increase in the outcome of the treated relative to its
    counterfactual?*

Yes. The command already outputs the average total treatment effect, that is, the numerator in
    the percentage increase.  To compute the denominator, you need to define a new outcome variable
    $\tilde{Y} = - Y * 1\{t < F_g\}$, where F_g is the first date at which $g$'s treatment has changed.
    Essentially, you replace the outcome by 0 after the treatment change, and by -Y before the
    treatment change.  Finally, you can just compute un-normalized event-study estimators with
    $\tilde{Y}$ as the outcome.

## References

de Chaisemartin, C, D'Haultfoeuille, X (2024).  [Difference-in-Differences Estimators of
Intertemporal Treatment Effects](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3731856).
    
de Chaisemartin, C, D'Haultfoeuille, X (2020b).  [Two-way fixed effects regressions with several
treatments](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3751060).

de Chaisemartin, C, D'Haultfoeuille, X, Pasquier, F, Vazquez-Bare, G (2022).
[Difference-in-Differences Estimators for Treatments Continuously Distributed at Every Period](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=4011782).
    
de Chaisemartin et al. (2025). [Using did_multiplegt_dyn to Estimate Event-Study Effects in Complex Designs: Overview, and Four Examples Based on Real Datasets](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5337463).

## Authors

Clément de Chaisemartin, Economics Department, Sciences Po, France.  
Xavier D'Haultfoeuille, CREST-ENSAE, France.  
Anzony Quispe, Economics Department, Sciences Po, France.

**<ins>Contact:</ins>**  
[chaisemartin.packages@gmail.com](mailto:chaisemartin.packages@gmail.com)
