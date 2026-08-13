"""Run fragility, missed-crisis, and autoregressive extensions."""

from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from sklearn.metrics import roc_auc_score

from baseline_regressions import DK_BANDWIDTH
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))


def _fit(data, outcome, predictors, bandwidth):
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


def _add_group_lags(panel, columns):
    result = panel.sort_values(["country_iso3", "year"]).copy()
    grouped = result.groupby("country_iso3", sort=False)
    for column in columns:
        result[f"lag1_{column}"] = grouped[column].shift(1)
    return result


def run_dynamic_comparison(panel):
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


def missed_crisis_diagnostics(panel):
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
                record[f"max_prior_{variable}"] = np.nan
                record[f"max_prior_{variable}_percentile"] = np.nan
                continue
            value = float(available[variable].max())
            record[f"max_prior_{variable}"] = value
            record[f"max_prior_{variable}_percentile"] = 100.0 * float(
                reference.le(value).mean()
            )
        rows.append(record)
    result = pd.DataFrame(rows)
    if not result.empty:
        result["missed_by_rzone"] = ~result["preceded_by_either_rzone"]
    return result


def case_study_2023(panel):
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
    panel = pd.read_parquet(DATA_DIR / "rzone_analysis_panel.parquet")

    fragility_coefficients, fragility_models = run_fragility_regressions(panel)
    fragility_coefficients.to_csv(
        OUTPUT_DIR / "fragility_regression_coefficients.csv", index=False
    )
    fragility_models.to_csv(OUTPUT_DIR / "fragility_regression_models.csv", index=False)

    dynamic_coefficients, dynamic_models = run_dynamic_comparison(panel)
    dynamic_coefficients.to_csv(
        OUTPUT_DIR / "dynamic_regression_coefficients.csv", index=False
    )
    dynamic_models.to_csv(OUTPUT_DIR / "dynamic_regression_models.csv", index=False)

    missed_crisis_diagnostics(panel).to_csv(
        OUTPUT_DIR / "missed_crisis_fragility.csv", index=False
    )
    case_study_2023(panel).to_csv(OUTPUT_DIR / "usa_2023_case_study.csv", index=False)
    print("Saved fragility, missed-crisis, dynamic, and 2023 extension results")
