# py_did_multiplegt_dyn

This is the python version of the main Stata package [did_multiplegt_dyn](https://github.com/chaisemartinPackages/did_multiplegt_dyn/tree/main/Stata).


[![PyPI version](https://img.shields.io/pypi/v/py-did-multiplegt-dyn.svg?color=blue)](https://pypi.org/project/py-did-multiplegt-dyn/)
[![Downloads](https://static.pepy.tech/personalized-badge/py-did-multiplegt-dyn?period=total&units=international_system&left_color=blue&right_color=grey&left_text=Downloads)](https://pepy.tech/project/py-did-multiplegt-dyn)
[![Last commit](https://img.shields.io/github/last-commit/anzonyquispe/py_did_multiplegt_dyn.svg)](https://github.com/anzonyquispe/py_did_multiplegt_dyn/commits/main)
[![GitHub stars](https://img.shields.io/github/stars/anzonyquispe/py_did_multiplegt_dyn.svg?style=social)](https://github.com/anzonyquispe/py_did_multiplegt_dyn/stargazers)
[![GitHub issues](https://img.shields.io/github/issues/anzonyquispe/py_did_multiplegt_dyn.svg)](https://github.com/anzonyquispe/py_did_multiplegt_dyn/issues)
[![License](https://img.shields.io/github/license/d2cml-ai/csdid.svg)](https://github.com/anzonyquispe/py_did_multiplegt_dyn/blob/main/LICENSE)



🔊🔊 **Check out [this](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=5337463) new document with examples on how to implement the did_multiplegt_dyn on various, complex designs!** We provide detailed guidelines on how to address designs with (1) a binary treatment that can turn on an off, (2) a continuous absorbing treatment, (3) a discrete multivalued treatment that can increase or decrease multiple times over time, and (4) two, binary and absorbing treatments, where the second treatment always happens after the first. You can find the code and data for replicating this examples in the "Examples did_multiplegt_dyn on various designs" folder. 🔊🔊



## DidMultiplegtDyn
Estimation in Difference-in-Difference (DID) designs with multiple groups and periods.

[Short description](#Short-description) | [Setup](#Setup) | [Syntax](#Syntax) | [Description](#Description)

[Options](#Options) | [Example](#Example) | [FAQ](#FAQ) | [References](#References) | [Authors](#Authors)

## Short description

Estimation of event-study Difference-in-Difference (DID) estimators in designs with multiple groups and periods, and with a potentially non-binary treatment that may increase or decrease multiple times.  



## Setup

### Python
```s
!pip install py-did-multiplegt-dyn
```


### Syntax
**DidMultiplegtDyn** <- def(**df** = *polars*, **outcome** = *string*, **group** = *string*, **time** = *string*, **treatment** = *string*, **effects** = 1,**normalized** = False, **effects_equal** = False, **placebo** = 0, **controls** = None, **trends_nonparam** = None, **trends_lin** = False, **continuous** = 0, **weight** = None, **cluster** = None, **same_switchers** = False, **same_switchers_pl** = False, **switchers** = "", **ci_level** = 95, **graph_off** = False, **save_results** = None, **less_conservative_se** = False, **dont_drop_larger_lower** = False, **drop_if_d_miss_before_first_switch** = False)


## Description

**did_multiplegt_dyn** estimates the effect of a treatment on an outcome, using group-(e.g. county- or state-) level panel data. The command computes the DID event-study estimators introduced in de Chaisemartin and D'Haultfoeuille (2024). Like other recently proposed DID estimation commands (**csdid**, **didimputation**,...), **did_multiplegt_dyn** can be used with a binary and staggered (absorbing) treatment. But unlike those other commands, **did_multiplegt_dyn** can also be used with a non-binary treatment (discrete or continuous) that can increase or decrease multiple times. Lagged treatments may affect the outcome, and the current and lagged treatments may have heterogeneous effects, across space and/or over time.  The event-study estimators computed by the command rely on a no-anticipation and parallel trends assumptions. The panel may be unbalanced:  not all groups have to be observed at every period.  The data may also be at a more disaggregated level than the group level (e.g. individual-level wage data to measure the effect of a regional-level minimum-wage on individuals' wages).

For all "switchers", namely groups that experience a change of their treatment over the study period, let $F_g$ denote the first time period when g's treatment changes. The command computes the non-normalized event-study estimators $DID_\ell$. $DID_1$ is the average, across all switchers, of DID estimators comparing the $F_{g-1}$ to $F_g$ outcome evolution of $g$ to that of groups with the same period-one treatment as $g$ but whose treatment has not changed yet at $F_g$. More generally, $DID_1$ is the average, across all switchers, of DID estimators comparing the $F_{g-1}$ to $F_{g-1+\ell}$ outcome evolution of $g$ to that of groups with the same period-one treatment as $g$ but whose treatment has not changed yet at $F_{g-1+\ell}$. Non-normalized event-study effects are average effects of having been exposed to a weakly higher treatment dose for $\ell$ periods, where the magnitude and timing of the incremental treatment doses can vary across groups. The command also computes the normalized event-study estimators $DID^n_\ell$, that normalize $DID_\ell$ by the average total incremental treatment dose received by switchers from $F_{g-1}$ to $F_{g-1+\ell}$ with respect to their period-one treatment. This normalization ensures that $DID^n_\ell$ estimates a weighted average of the effects of the current treatment and of its $\ell-1$ first lags on the outcome. The command also computes an estimated average total effect per unit of treatment, where "total effect" refers to the sum of the effects of a treatment increment, at the time when it takes place and at later periods, see Section 3.3 of de Chaisemartin and D'Haultfoeuille (2024) for further details. Finally, the command also computes placebo estimators, that average DIDs comparing the outcome evolution of switcher $g$ and of its control groups, from $F_{g-1}$ to $F_{g-1-\ell}$, namely before g's treatment changes for the first time. Those placebos can be used to test the parallel trends and no-anticipation assumptions under which the estimators computed by **did_multiplegt_dyn** are unbiased.

**Y** ( **outcome**) is the outcome variable.

**G** ( **group**) is the group variable.

**T** ( **time**) is the time period variable.  The command assumes that the time variable is evenly spaced
    (e.g.: the panel is at the yearly level, and no year is missing for all groups).  When it is not
    (e.g.: the panel is at the yearly level, but three consecutive years are missing for all groups),
    the command can still be used, though it requires a bit of tweaking, see FAQ section below.

**D** ( **treatment**) is the treatment variable.
    

## Options   
**effects(**#**)**: gives the number of event-study effects to be estimated. By default, the command estimates non-normalized event-study effects. Non-normalized event-study effects are averages, across all switchers, of the effect of having received their actual rather than their period-one treatment dose, for $\ell$ periods. While non-normalized event-study effects can be interpreted as average effects of being exposed to a weakly higher treatment dose for $\ell$ periods, the magnitude and timing of the incremental treatment doses can vary across groups.


**normalized**: when this option is specified, the command estimates normalized event-study effects, that are equal to a weighted average of the effects of the current treatment and of its $\ell-1$ first lags on the outcome. See Sections 3.1 and 3.2 of de Chaisemartin and D'Haultfoeuille (2020a) for further details.

**normalized_weights**: the command reports the weights that normalized effect $\ell$ puts on the effect of the current treatment, on the effect of the first treatment lag, etc.

**effects_equal**: when this option is specified and the user requests that at least two effects be estimated, the command performs an F-test that all effects are equal. When the **normalized** option is specified, this test can be useful to assess if the current and lagged treatments all have the same effect on the outcome or if their effects differ, see Lemma 3 of de Chaisemartin and D'Haultfoeuille (2020a).

**placebo(**#**)**: gives the number of placebo estimators to be computed. Placebos compare the outcome evolution of switchers and of their controls, before switchers' treatment changes for the first time. Under the parallel trends and no-anticipation assumptions underlying the event-study estimators computed by did_multiplegt_dyn, the expectation of the placebos is equal to zero. Thus, placebos can be used to test those assumptions, by testing the None that all placebos are equal to zero. If the user requests that at least two placebos be estimated, the command computes the p-value of a joint test of that None hypothesis. The number of placebos requested can be at most equal to the number of time periods in the data minus 2, though most often only a smaller number of placebos can be computed. Also, the number of placebos requested cannot be larger than the number of effects requested.

**controls(***list***)**: gives the names of the control variables to be included in the estimation. Estimators with controls are similar to those without controls, except that the first-difference of the outcome is replaced by residuals from regressions of the first-difference of the outcome on the first-differences of the controls and time fixed effects. Those regressions are estimated in the sample of control $(g,t)$s: $(g,t)$s such that group $g$'s treatment has not changed yet at $t$. Those regressions are also estimated separately for each value of the baseline treatment. Estimators with controls are unbiased even if groups experience differential trends, provided such differential trends can be fully explained by a linear model in covariates changes. To control for time-invariant covariates, one needs to interact them with the time variable **T**, or with time fixed effects. See Section 1.2 of the Web Appendix of de Chaisemartin and D'Haultfoeuille (2020a) for further details.

**trends_lin**: when this option is specified, the estimation of the treatment effects allows for group-specific linear trends. Estimators with linear trends start by computing event-study effects on the outcome's first-difference, rather than on the outcome itself, thus allowing for group-specific linear trends. Then, to recover event-study effect $\ell$ on the outcome, event-study effects on the outcome's first-difference are summed from 1 to $\ell$. See Section 1.3 of the Web Appendix of de Chaisemartin and D'Haultfoeuille (2024) for further details. When this option is specified, the estimated average total effect per unit of treatment is not computed.

**trends_nonparam(***list***)**: when this option is specified, the DID estimators computed by the command only compare switchers to controls whose treatment has not changed yet, with the same baseline treatment, and with the same value of the varlist. Estimators with the **trends_nonparam** option are unbiased even if groups experience differential trends, provided all groups with the same value of the varlist experience parallel trends. The vector can only include time-invariant variables, and the interaction of those variables has to be coarser than the group variable. For instance, if one works with a county $\times$ year data set and one wants to allow for state-specific trends, then one should write *trends_nonparam(state)*, where state is the state identifier. See Section 1.4 of the Web Appendix of de Chaisemartin and D'Haultfoeuille (2024) for further details.

**continuous(**#**)**: allows to use the command even when groups' period-one treatment is continuous, meaning that all groups have a different period-one treatment value. With a discrete period-one treatment, the command compares the outcome evolution of switchers and non-switchers with the same period-one treatment.  But with a truly continuous period-one treatment, there will be no two groups with the same period-one treatment. The command assumes that group's status-quo outcome evolution is a polynomial in their period-one treatment. The user's chosen polynomial order is the option's argument. See Section 1.10 of the Web Appendix of de Chaisemartin and D'Haultfoeuille (2024) for further details. Unlike the other variance estimators computed by the command, those computed when the continuous option is specified are not backed by a proven asymptotic normality result. Preliminary simulation evidence indicates that when the option is used with a correctly-specified polynomial order, those variance estimators are conservative. On the other hand, when the specified polynomial order is strictly larger than needed, those variance estimators can become liberal. Thus, when this option is specified, we recommend using the bootstrap for inference, by manually bootstrapping the command. At least, one should perform a robustness check where one compares the analytic variance computed by the command to a bootstrapped variance.

**weight(***varlist***)**: gives the name of a variable to be used to weight the data. For instance, if one works with a district $\times$ year data set and one wants to weight the estimation by each district $\times$ year's population, one should write weight(population), where population is the population of each district $\times$ year. If the data set is at a more disaggregated level than group $\times$ time, the command aggregates it at the group $\times$ time level internally, and weights the estimation by the number of observations in each group $\times$ time cell if the weight option is not specified, or by the sum of the weights of the observations in each group $\times$ time cell if the weight option is specified.

**cluster(***varname***)**: can be used to cluster the estimators' standard errors. Only one clustering variable is allowed. A common practice in DID analysis is to cluster standard errors at the group level. Such clustering is implemented by default by the command. Standard errors can be clustered at a more aggregated level than the group level, but they cannot be clustered at a more disaggregated level.


**same_switchers**: if this option is specified and the user requests that at least two effects be estimated, the command will restrict the estimation of the event-study effects to switchers for which all effects can be estimated, to avoid compositional changes.

**same_switchers_pl**: the command restricts the estimation of event-study and placebo effects to switchers for which all event-study and placebos effects can be estimated. 

**switchers(***string***)**: one may be interested in estimating separately the treatment effect of switchers-in, whose average treatment after they switch is larger than their baseline treatment, and of switchers-out, whose average treatment after they switch is lower than their baseline treatment. In that case, one should run the command first with the switchers = "in" option, and then with the switchers = "out" option.

**only_never_switchers**: if this option is specified, the command estimates the event-study effects using only never-switchers as control units.

**ci_level**: with this option you can change the level of the confidence intervals displayed in the output tables and the graphs. The default value is fixed at 95, yielding a 95% coverage.

**less_conservative_se**: when groups' treatment can change multiple times, the standard errors reported by default by the command may be conservative. Then, less conservative standard errors can be obtained by specifying this option. See de Chaisemartin et al. (2024) for further details.

**bootstrap(***reps*, *seed***)**: when this option is specified, bootstraped instead of analytical standard errors are reported.  The number of bootstrap replications is the option's first argument, the *seed* is the option's second argument. The two arguments need to be separated by a comma. You always need to specify the comma, even if you leave either or both arguments blank.  In this case, the default values of both arguments are 50 replications and not setting a seed. If the **cluster** option is also requested, the bootstrap is clustered at the level requested in the cluster option.

**dont_drop_larger_lower**: by default, the command drops all the $(g,t)$ cells such that at $t$, group $g$ has experienced both a strictly larger and a strictly lower treatment than its baseline treatment. de Chaisemartin and D'Haultfoeuille (2020a) recommend this procedure, if you are interested in more details you can see their Section 3.1. The option **dont_drop_larger_lower** allows to overwrite this procedure and keeps $(g,t)$ cells such that at $t$, group $g$ has experienced both a strictly larger and a strictly lower treatment than its baseline treatment in the estimation sample.

**drop_if_d_miss_before_first_switch**: This option is relevant when the treatment of some groups is missing at some time periods. Then, the command imputes some of those missing treatments. Those imputations are detailed in Appendix A of de Chaisemartin et al (2024). In designs where groups' treatments can change at most once, all those imputations are justified by the design. In other designs, some of those imputations may be liberal. drop_if_d_miss_before_first_switch can be used to overrule the potentially liberal imputations that are not innocuous for the non-normalized event-study estimators. See Appendix A of de Chaisemartin et al (2024) for further details.

**<ins>Technical note on Python output:</ins>** 

The standard output of did_multiplegt_dyn in Python is a dictionary of objects with *DID* class. This allows for customized *summary* method dispatch and plot suitable for event studies. The basic output list of the program includes: (i) a list with all the results from the estimation (*results*), (ii) a matplotlib object for the event-study graph (*plot*). Additional options enrich the output list and normally add up to other items in the *results* sublist. You can inspect the attribute result of the *DID* class which is a dicctionaty with all the intermediate results, final dataset, joint placebo and effects tests.



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

favara_imbs = pl.read_csv('https://raw.githubusercontent.com/anzonyquispe/py_did_multiplegt_dyn/main/data/favara_imbs_did_multiplegt_dyn.csv', 
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

