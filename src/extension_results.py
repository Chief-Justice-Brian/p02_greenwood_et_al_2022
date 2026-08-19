"""Run the fragility, missed-crisis, autoregressive, and 2023 extensions.

``run_fragility_regressions`` interacts the R-zones with the JST bank
fragility flags on subsample-matched specifications;
``run_continuous_fragility_regressions`` is the proposal's continuous
robustness check, with the standardized ratio level replacing each flag;
``run_fragility_threshold_sweep`` re-estimates the noncore specification at
every candidate cutoff in the proposal's grid, with the q=80 baseline frozen
ex ante; ``run_fragility_form_checks`` runs two precommitted functional-form
checks on the standalone noncore channel (level vs 3-year change, and a
quadratic curvature term); ``fragility_correlations`` reports the pairwise
correlations that motivate keeping the three ratios separate;
``run_rzone_exit_regressions`` tests whether risk stays elevated after a
country leaves the R-zone;
``missed_crisis_diagnostics`` profiles the crises both R-zones missed;
``run_dynamic_comparison`` estimates the footnote-10 autoregressive
specification against the static baseline and reports in-sample AUC for
both; ``case_study_2023`` extracts the US 2019-2023 predictor slice
(``usa_2023_case_study.csv``).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from sklearn.metrics import roc_auc_score

from baseline_regressions import DK_BANDWIDTH
from build_analysis_panel import load_analysis_panel
from rzone_features import add_zone_exit_buckets, indicator_above, quantile_cutoffs
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))

# The proposal's cutoff grid for the noncore fragility flag, with q = 0.8
# fixed ex ante as the baseline; the sweep is reported in full and the
# baseline is never revised toward the best-performing gridpoint.
FRAGILITY_SWEEP_QUANTILES = [0.5, 2 / 3, 0.75, 0.8, 0.9]


def _fit(data, outcome, predictors, bandwidth):
    """Fit an entity-effects PanelOLS with Driscoll-Kraay standard errors.

    :param data: frame holding country_iso3, year, the outcome, and the
        predictors.
    :param outcome: name of the 0/1 crisis outcome column.
    :param predictors: list of indicator column names to regress on.
    :param bandwidth: Driscoll-Kraay lag bandwidth for the horizon.
    :returns: the fitted result and the complete-case estimation sample indexed
        by (country_iso3, year).
    """
    estimation = data[["country_iso3", "year", outcome, *predictors]].dropna().copy()
    estimation = estimation.set_index(["country_iso3", "year"])
    model = PanelOLS(
        estimation[outcome].astype(float),
        estimation[predictors].astype(float),
        entity_effects=True,
        drop_absorbed=True,
        check_rank=True,
    )
    result = model.fit(cov_type="driscoll-kraay", bandwidth=bandwidth)
    return result, estimation


def run_fragility_regressions(panel):
    """Interact each sector R-zone with the JST bank-fragility flags.

    For every sector x fragility measure x horizon, fits the GHSS baseline and
    the fragility-extended specification on the same complete-case sample.

    :param panel: country-year analysis panel from load_analysis_panel.
    :returns: (coefficient rows, model-fit rows) as DataFrames.
    """
    coefficients = []
    models = []
    for sector in ["business", "household"]:
        debt = f"high_{sector}_debt_growth"
        price = f"high_{sector}_price_growth"
        rzone = f"rzone_{sector}"
        base_predictors = [debt, price, rzone]
        for fragility in ["noncore", "ltd", "lev"]:
            high = f"high_{fragility}"
            interaction = f"rzone_{sector}_x_high_{fragility}"
            # Both models use the extension's smaller complete-case sample.
            sample = panel[
                panel[f"in_{sector}_sample"]
                & panel[high].notna()
                & panel[interaction].notna()
            ].copy()
            specifications = {
                "ghss_same_sample": base_predictors,
                "fragility_extension": [*base_predictors, high, interaction],
            }
            for horizon in [1, 2, 3, 4]:
                outcome = f"crisis_next_{horizon}y"
                for specification, predictors in specifications.items():
                    result, _ = _fit(sample, outcome, predictors, DK_BANDWIDTH[horizon])
                    for variable in predictors:
                        coefficients.append(
                            {
                                "sector": sector,
                                "fragility_measure": fragility,
                                "horizon": horizon,
                                "specification": specification,
                                "variable": variable,
                                "coefficient_pp": 100.0 * result.params[variable],
                                "std_error_pp": 100.0 * result.std_errors[variable],
                                "t_stat": result.tstats[variable],
                                "p_value": result.pvalues[variable],
                            }
                        )
                    models.append(
                        {
                            "sector": sector,
                            "fragility_measure": fragility,
                            "horizon": horizon,
                            "specification": specification,
                            "n": int(result.nobs),
                            "within_r2_pct": 100.0 * result.rsquared_within,
                        }
                    )
    return pd.DataFrame(coefficients), pd.DataFrame(models)


def run_continuous_fragility_regressions(panel):
    """Re-fit the fragility extension with standardized ratio levels.

    The proposal's continuous robustness check: the 0/1 flag in the extended
    specification is replaced by the standardized level of each ratio,
    z(noncore), z(ltd), or -z(lev), so a bigger value always means more
    fragile and coefficients read as percentage points of crisis probability
    per one standard deviation. Means and standard deviations come from the
    in-paper-sample rows, the same reference sample that sets the flag
    cutoffs, so post-2016 data never moves the scale.

    :param panel: country-year analysis panel from load_analysis_panel.
    :returns: (coefficient rows, model-fit rows) as DataFrames.
    """
    coefficients = []
    models = []
    for sector in ["business", "household"]:
        debt = f"high_{sector}_debt_growth"
        price = f"high_{sector}_price_growth"
        rzone = f"rzone_{sector}"
        base_predictors = [debt, price, rzone]
        for fragility in ["noncore", "ltd", "lev"]:
            reference = panel["in_paper_sample"] & panel[fragility].notna()
            center = panel.loc[reference, fragility].mean()
            spread = panel.loc[reference, fragility].std(ddof=1)
            fragility_z = f"{fragility}_fragility_z"
            interaction = f"{rzone}_x_{fragility_z}"
            # Same complete-case sample as the flag version of this measure.
            sample = panel[
                panel[f"in_{sector}_sample"] & panel[fragility].notna()
            ].copy()
            sample[fragility_z] = (sample[fragility] - center) / spread
            if fragility == "lev":
                # lev is a capital ratio, so its standardized level enters
                # with a minus sign: bigger must always mean more fragile.
                sample[fragility_z] = -sample[fragility_z]
            sample[interaction] = sample[rzone] * sample[fragility_z]
            specifications = {
                "ghss_same_sample": base_predictors,
                "continuous_extension": [*base_predictors, fragility_z, interaction],
            }
            for horizon in [1, 2, 3, 4]:
                outcome = f"crisis_next_{horizon}y"
                for specification, predictors in specifications.items():
                    result, _ = _fit(sample, outcome, predictors, DK_BANDWIDTH[horizon])
                    for variable in predictors:
                        coefficients.append(
                            {
                                "sector": sector,
                                "fragility_measure": fragility,
                                "horizon": horizon,
                                "specification": specification,
                                "variable": variable,
                                "coefficient_pp": 100.0 * result.params[variable],
                                "std_error_pp": 100.0 * result.std_errors[variable],
                                "t_stat": result.tstats[variable],
                                "p_value": result.pvalues[variable],
                            }
                        )
                    models.append(
                        {
                            "sector": sector,
                            "fragility_measure": fragility,
                            "horizon": horizon,
                            "specification": specification,
                            "n": int(result.nobs),
                            "within_r2_pct": 100.0 * result.rsquared_within,
                        }
                    )
    return pd.DataFrame(coefficients), pd.DataFrame(models)


def run_fragility_threshold_sweep(panel):
    """Sweep the noncore fragility cutoff over the proposal's quantile grid.

    Re-estimates the noncore-extended specification at every candidate
    cutoff on a fixed complete-case sample per sector, reporting the
    interaction estimate with its 95% confidence band and the joint R-zone x
    high-fragility cell count so thin cells are visible.

    :param panel: country-year analysis panel from load_analysis_panel.
    :returns: one row per sector x candidate quantile x horizon.
    """
    reference = panel["in_paper_sample"] & panel["noncore"].notna()
    cutoffs = quantile_cutoffs(
        panel.loc[reference, "noncore"], FRAGILITY_SWEEP_QUANTILES
    )
    rows = []
    for sector in ["business", "household"]:
        debt = f"high_{sector}_debt_growth"
        price = f"high_{sector}_price_growth"
        rzone = f"rzone_{sector}"
        # The sample never changes across gridpoints, so every row of the
        # sweep is comparable: only the cutoff moves.
        sample = panel[panel[f"in_{sector}_sample"] & panel["noncore"].notna()].copy()
        for quantile in FRAGILITY_SWEEP_QUANTILES:
            high = "high_noncore_at_q"
            interaction = f"{rzone}_x_{high}"
            sample[high] = indicator_above(sample["noncore"], cutoffs[quantile])
            sample[interaction] = (sample[rzone] * sample[high]).astype("Int64")
            predictors = [debt, price, rzone, high, interaction]
            for horizon in [1, 2, 3, 4]:
                outcome = f"crisis_next_{horizon}y"
                result, estimation = _fit(
                    sample, outcome, predictors, DK_BANDWIDTH[horizon]
                )
                confidence = result.conf_int().loc[interaction]
                joint_cell_count = int(
                    (estimation[rzone].eq(1) & estimation[high].eq(1)).sum()
                )
                rows.append(
                    {
                        "sector": sector,
                        "quantile_pct": round(100 * quantile, 1),
                        "noncore_cutoff": cutoffs[quantile],
                        "horizon": horizon,
                        "n": int(result.nobs),
                        "joint_cell_count": joint_cell_count,
                        "interaction_pp": 100.0 * result.params[interaction],
                        "interaction_se_pp": 100.0 * result.std_errors[interaction],
                        "interaction_ci_low_pp": 100.0 * confidence["lower"],
                        "interaction_ci_high_pp": 100.0 * confidence["upper"],
                        "interaction_p_value": result.pvalues[interaction],
                        "high_fragility_pp": 100.0 * result.params[high],
                        "high_fragility_p_value": result.pvalues[high],
                    }
                )
    return pd.DataFrame(rows)


def run_fragility_form_checks(panel):
    """Run two precommitted functional-form checks on the noncore channel.

    The stock-flow check races the standardized noncore level against its
    standardized 3-year change (level_only, change_only, level_and_change)
    on a shared complete-case sample; the Diamond-Dybvig reading, stated
    before estimation, expects the level to carry the signal. The curvature
    check adds a squared term to the continuous specification on that
    specification's own sample; the no-threshold reading of the flat sweep,
    also stated before estimation, expects a near-zero quadratic.

    Interactions stay out of both checks: they probe the standalone
    channel's form, and the amplification interaction was already rejected.

    :param panel: country-year analysis panel from load_analysis_panel.
    :returns: (coefficient rows, model-fit rows) as DataFrames, labeled by
        check and specification.
    """
    working = panel.sort_values(["country_iso3", "year"]).copy()
    # The base grid is a complete year grid per country, so diff(3) is a
    # change over exactly three years.
    working["delta3_noncore"] = working.groupby("country_iso3", sort=False)[
        "noncore"
    ].diff(3)

    level_z = "noncore_fragility_z"
    change_z = "delta3_noncore_fragility_z"
    level_z_sq = "noncore_fragility_z_sq"
    for source, z_column in [("noncore", level_z), ("delta3_noncore", change_z)]:
        reference = working["in_paper_sample"] & working[source].notna()
        center = working.loc[reference, source].mean()
        spread = working.loc[reference, source].std(ddof=1)
        working[z_column] = (working[source] - center) / spread
    working[level_z_sq] = working[level_z] ** 2

    coefficients = []
    models = []
    for sector in ["business", "household"]:
        debt = f"high_{sector}_debt_growth"
        price = f"high_{sector}_price_growth"
        rzone = f"rzone_{sector}"
        base_predictors = [debt, price, rzone]
        checks = {
            # Both stock-flow samples require the 3-year change, so all
            # three specifications share one complete-case sample.
            "stock_flow": (
                working[
                    working[f"in_{sector}_sample"]
                    & working["noncore"].notna()
                    & working["delta3_noncore"].notna()
                ],
                {
                    "level_only": [*base_predictors, level_z],
                    "change_only": [*base_predictors, change_z],
                    "level_and_change": [*base_predictors, level_z, change_z],
                },
            ),
            # The curvature check runs on the continuous specification's
            # exact sample so its verdict speaks to the same regression.
            "curvature": (
                working[working[f"in_{sector}_sample"] & working["noncore"].notna()],
                {
                    "level_reference": [*base_predictors, level_z],
                    "level_quadratic": [*base_predictors, level_z, level_z_sq],
                },
            ),
        }
        for check, (sample, specifications) in checks.items():
            for horizon in [1, 2, 3, 4]:
                outcome = f"crisis_next_{horizon}y"
                for specification, predictors in specifications.items():
                    result, _ = _fit(sample, outcome, predictors, DK_BANDWIDTH[horizon])
                    for variable in predictors:
                        coefficients.append(
                            {
                                "check": check,
                                "sector": sector,
                                "horizon": horizon,
                                "specification": specification,
                                "variable": variable,
                                "coefficient_pp": 100.0 * result.params[variable],
                                "std_error_pp": 100.0 * result.std_errors[variable],
                                "t_stat": result.tstats[variable],
                                "p_value": result.pvalues[variable],
                            }
                        )
                    models.append(
                        {
                            "check": check,
                            "sector": sector,
                            "horizon": horizon,
                            "specification": specification,
                            "n": int(result.nobs),
                            "within_r2_pct": 100.0 * result.rsquared_within,
                        }
                    )
    return pd.DataFrame(coefficients), pd.DataFrame(models)


def fragility_correlations(panel):
    """Compute pairwise correlations of the fragility ratios.

    Uses the in-paper-sample rows, the same reference sample that sets the
    fragility cutoffs. The pairwise complete-case counts come along so thin
    overlaps are visible.

    :param panel: country-year analysis panel from load_analysis_panel.
    :returns: one row per unordered pair of noncore, ltd, and lev.
    """
    historical = panel[panel["in_paper_sample"]]
    variables = ["noncore", "ltd", "lev"]
    rows = []
    for position, first in enumerate(variables):
        for second in variables[position + 1 :]:
            pair = historical[[first, second]].dropna()
            rows.append(
                {
                    "variable_a": first,
                    "variable_b": second,
                    "correlation": float(pair[first].corr(pair[second])),
                    "n_pairs": len(pair),
                }
            )
    return pd.DataFrame(rows)


def _add_group_lags(panel, columns):
    """Add a lag1_{column} one-year within-country lag for each given column.

    :param panel: country-year frame with country_iso3 and year columns.
    :param columns: names of the columns to lag.
    """
    result = panel.sort_values(["country_iso3", "year"]).copy()
    grouped = result.groupby("country_iso3", sort=False)
    for column in columns:
        result[f"lag1_{column}"] = grouped[column].shift(1)
    return result


def run_dynamic_comparison(panel):
    """Race the static baseline against the footnote-10 autoregressive model.

    Both specifications are fit per sector and horizon on the dynamic model's
    complete-case sample; in-sample AUC comes from the fitted values.

    :param panel: country-year analysis panel from load_analysis_panel.
    :returns: (coefficient rows, model-fit rows) as DataFrames.
    """
    coefficient_rows = []
    model_rows = []
    lag_columns = ["crisis_bvx"]
    for sector in ["business", "household"]:
        lag_columns.extend(
            [
                f"high_{sector}_debt_growth",
                f"high_{sector}_price_growth",
                f"rzone_{sector}",
            ]
        )
    working = _add_group_lags(panel, list(dict.fromkeys(lag_columns)))

    for sector in ["business", "household"]:
        debt = f"high_{sector}_debt_growth"
        price = f"high_{sector}_price_growth"
        rzone = f"rzone_{sector}"
        baseline = [debt, price, rzone]
        dynamic = [
            *baseline,
            "crisis_bvx",
            "lag1_crisis_bvx",
            f"lag1_{debt}",
            f"lag1_{price}",
            f"lag1_{rzone}",
        ]
        complete = working[working[f"in_{sector}_sample"]].dropna(subset=dynamic)
        for horizon in [1, 2, 3, 4]:
            outcome = f"crisis_next_{horizon}y"
            for specification, predictors in {
                "ghss_same_sample": baseline,
                "autoregressive": dynamic,
            }.items():
                result, estimation = _fit(
                    complete, outcome, predictors, DK_BANDWIDTH[horizon]
                )
                fitted = result.fitted_values.iloc[:, 0].reindex(estimation.index)
                observed = estimation[outcome].astype(float)
                auc = (
                    roc_auc_score(observed, fitted)
                    if observed.nunique() == 2
                    else np.nan
                )
                for variable in predictors:
                    coefficient_rows.append(
                        {
                            "sector": sector,
                            "horizon": horizon,
                            "specification": specification,
                            "variable": variable,
                            "coefficient_pp": 100.0 * result.params[variable],
                            "std_error_pp": 100.0 * result.std_errors[variable],
                            "t_stat": result.tstats[variable],
                            "p_value": result.pvalues[variable],
                        }
                    )
                model_rows.append(
                    {
                        "sector": sector,
                        "horizon": horizon,
                        "specification": specification,
                        "n": int(result.nobs),
                        "within_r2_pct": 100.0 * result.rsquared_within,
                        "in_sample_auc": auc,
                    }
                )
    return pd.DataFrame(coefficient_rows), pd.DataFrame(model_rows)


def run_rzone_exit_regressions(panel):
    """Test whether crisis risk stays elevated after leaving the R-zone.

    The zone's price condition switches off as a boom stalls, which is often
    the prelude to the bust, so the expectation stated before estimation is
    that recently-exited country-years carry elevated risk that decays with
    time since exit. Two specifications per sector share one complete-case
    sample beside the same-sample GHSS baseline: a pooled recent-exit
    indicator (in the zone within the past three years, out ever since) and
    its mutually exclusive one-, two-, and three-years-since-exit buckets.

    Every specification runs on two samples: all origins, and origins with
    no BVX crisis in the current or three prior years. A crisis itself can
    crush prices and eject a country from the zone, and BVX code some
    multi-year episodes as two onsets, so only the no-recent-crisis sample
    reads the exit effect as advance warning rather than episode overlap.

    :param panel: country-year analysis panel from load_analysis_panel.
    :returns: (coefficient rows, model-fit rows) as DataFrames, labeled by
        sample.
    """
    coefficients = []
    models = []
    for sector in ["business", "household"]:
        zone = f"rzone_{sector}"
        working = add_zone_exit_buckets(panel, zone)
        bucket_columns = [f"{zone}_exited_{k}y" for k in [1, 2, 3]]
        recent_exit = f"{zone}_recent_exit"
        # The buckets are mutually exclusive, so their sum is the pooled
        # indicator; NA in any bucket keeps the row out of both specs.
        working[recent_exit] = (
            working[bucket_columns[0]]
            + working[bucket_columns[1]]
            + working[bucket_columns[2]]
        ).astype("Int64")
        # Kleene logic again: one observed crisis year is enough to exclude
        # an origin; unknown-history origins stay out of the clean sample.
        grouped_crisis = working.groupby("country_iso3", sort=False)["crisis_bvx"]
        had_recent_crisis = working["crisis_bvx"].eq(1)
        for lag in [1, 2, 3]:
            had_recent_crisis = had_recent_crisis | grouped_crisis.shift(lag).eq(1)
        no_recent_crisis = (~had_recent_crisis).fillna(False)
        debt = f"high_{sector}_debt_growth"
        price = f"high_{sector}_price_growth"
        base_predictors = [debt, price, zone]
        in_scope = working[f"in_{sector}_sample"] & working[recent_exit].notna()
        samples = {
            "all_origins": working[in_scope],
            "no_recent_crisis": working[in_scope & no_recent_crisis],
        }
        specifications = {
            "ghss_same_sample": base_predictors,
            "recent_exit": [*base_predictors, recent_exit],
            "exit_by_year": [*base_predictors, *bucket_columns],
        }
        for sample_label, sample in samples.items():
            for horizon in [1, 2, 3, 4]:
                outcome = f"crisis_next_{horizon}y"
                for specification, predictors in specifications.items():
                    result, _ = _fit(sample, outcome, predictors, DK_BANDWIDTH[horizon])
                    for variable in predictors:
                        coefficients.append(
                            {
                                "sample": sample_label,
                                "sector": sector,
                                "horizon": horizon,
                                "specification": specification,
                                "variable": variable,
                                "coefficient_pp": 100.0 * result.params[variable],
                                "std_error_pp": 100.0 * result.std_errors[variable],
                                "t_stat": result.tstats[variable],
                                "p_value": result.pvalues[variable],
                            }
                        )
                    models.append(
                        {
                            "sample": sample_label,
                            "sector": sector,
                            "horizon": horizon,
                            "specification": specification,
                            "n": int(result.nobs),
                            "within_r2_pct": 100.0 * result.rsquared_within,
                        }
                    )
    return pd.DataFrame(coefficients), pd.DataFrame(models)


def missed_crisis_diagnostics(panel):
    """Profile each BVX crisis by whether an R-zone fired in the prior 3 years.

    Each fragility measure contributes its most fragile prior value (maximum
    noncore and ltd, minimum lev since fragility means thin capital) and its
    percentile measured from the fragile end, so 100 always means maximally
    fragile.

    :param panel: country-year analysis panel from load_analysis_panel.
    :returns: one row per crisis inside the replication sample with the worst
        prior bank-fragility values, their fragility percentiles, and a
        missed_by_rzone flag for crises neither sector's R-zone preceded.
    """
    historical = panel[panel["is_historical_replication"]].copy()
    fragility_reference = {
        variable: historical[variable].dropna()
        for variable in ["noncore", "ltd", "lev"]
    }
    rows = []
    for _, crisis in historical[historical["crisis_bvx"].eq(1)].iterrows():
        country = crisis["country_iso3"]
        year = int(crisis["year"])
        prior = historical[
            historical["country_iso3"].eq(country)
            & historical["year"].between(year - 3, year - 1)
        ].sort_values("year")
        # Keep crises that could have been forecast from at least one common
        # sample origin; older/uncovered events are outside the replication.
        if prior.empty or not prior["in_paper_sample"].any():
            continue
        record = {
            "country_iso3": country,
            "country": crisis["country"],
            "crisis_year": year,
            "preceded_by_business_rzone": bool(prior["rzone_business"].eq(1).any()),
            "preceded_by_household_rzone": bool(prior["rzone_household"].eq(1).any()),
        }
        record["preceded_by_either_rzone"] = (
            record["preceded_by_business_rzone"]
            or record["preceded_by_household_rzone"]
        )
        for variable, reference in fragility_reference.items():
            available = prior.dropna(subset=[variable])
            if available.empty:
                record[f"prior_{variable}_worst"] = np.nan
                record[f"prior_{variable}_fragility_pct"] = np.nan
                continue
            if variable == "lev":
                # Thin capital is the fragile direction, so the worst prior
                # value is the minimum and its percentile counts from below.
                value = float(available[variable].min())
                fragility_pct = 100.0 * float(reference.ge(value).mean())
            else:
                value = float(available[variable].max())
                fragility_pct = 100.0 * float(reference.le(value).mean())
            record[f"prior_{variable}_worst"] = value
            record[f"prior_{variable}_fragility_pct"] = fragility_pct
        rows.append(record)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["missed_by_rzone"] = ~result["preceded_by_either_rzone"]
    return result


def case_study_2023(panel):
    """Extract the US 2019-2023 predictor, R-zone, and fragility columns.

    :param panel: country-year analysis panel from load_analysis_panel.
    """
    columns = [
        "country_iso3",
        "country",
        "year",
        "delta3_business_debt_gdp",
        "delta3_log_real_equity",
        "rzone_business",
        "delta3_household_debt_gdp",
        "delta3_log_real_house_price",
        "rzone_household",
        "noncore",
        "ltd",
        "lev",
    ]
    return panel[panel["country_iso3"].eq("USA") & panel["year"].between(2019, 2023)][
        columns
    ]


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_analysis_panel(DATA_DIR)

    fragility_coefficients, fragility_models = run_fragility_regressions(panel)
    fragility_coefficients.to_csv(
        OUTPUT_DIR / "fragility_regression_coefficients.csv", index=False
    )
    fragility_models.to_csv(OUTPUT_DIR / "fragility_regression_models.csv", index=False)

    continuous_coefficients, continuous_models = run_continuous_fragility_regressions(
        panel
    )
    continuous_coefficients.to_csv(
        OUTPUT_DIR / "fragility_continuous_coefficients.csv", index=False
    )
    continuous_models.to_csv(
        OUTPUT_DIR / "fragility_continuous_models.csv", index=False
    )

    run_fragility_threshold_sweep(panel).to_csv(
        OUTPUT_DIR / "fragility_threshold_sweep.csv", index=False
    )

    form_coefficients, form_models = run_fragility_form_checks(panel)
    form_coefficients.to_csv(
        OUTPUT_DIR / "fragility_form_checks_coefficients.csv", index=False
    )
    form_models.to_csv(OUTPUT_DIR / "fragility_form_checks_models.csv", index=False)

    exit_coefficients, exit_models = run_rzone_exit_regressions(panel)
    exit_coefficients.to_csv(OUTPUT_DIR / "rzone_exit_coefficients.csv", index=False)
    exit_models.to_csv(OUTPUT_DIR / "rzone_exit_models.csv", index=False)
    fragility_correlations(panel).to_csv(
        OUTPUT_DIR / "fragility_correlations.csv", index=False
    )

    dynamic_coefficients, dynamic_models = run_dynamic_comparison(panel)
    dynamic_coefficients.to_csv(
        OUTPUT_DIR / "dynamic_regression_coefficients.csv", index=False
    )
    dynamic_models.to_csv(OUTPUT_DIR / "dynamic_regression_models.csv", index=False)

    missed_crisis_diagnostics(panel).to_csv(
        OUTPUT_DIR / "missed_crisis_fragility.csv", index=False
    )
    case_study_2023(panel).to_csv(OUTPUT_DIR / "usa_2023_case_study.csv", index=False)
    print(
        "Saved fragility (with threshold sweep and correlations), missed-crisis, "
        "dynamic, and 2023 extension results"
    )
