"""Estimate the primary GHSS country-fixed-effects probability models."""

from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy.stats import t as student_t

from build_analysis_panel import load_analysis_panel
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))

# Table 4 uses Driscoll--Kraay lags 0, 3, 5, and 6 at horizons 1--4.
DK_BANDWIDTH = {1: 0, 2: 3, 3: 5, 4: 6}


def _fit(data, outcome_column, predictor_columns, bandwidth):
    """Fit a country-fixed-effects linear probability model.

    :param data: country-year panel slice containing the outcome and predictors.
    :param outcome_column: name of the 0/1 crisis outcome column.
    :param predictor_columns: list of indicator column names to regress on.
    :param bandwidth: Driscoll--Kraay lag length for the standard errors.
    """
    estimation = data[
        ["country_iso3", "year", outcome_column, *predictor_columns]
    ].dropna()
    estimation = estimation.set_index(["country_iso3", "year"])

    model = PanelOLS(
        estimation[outcome_column].astype(float),
        estimation[predictor_columns].astype(float),
        entity_effects=True,
        drop_absorbed=True,
        check_rank=True,
    )
    return model.fit(cov_type="driscoll-kraay", bandwidth=bandwidth)


def _t_test(effect, standard_error, degrees_of_freedom):
    """Run a two-sided t-test of an effect against zero.

    :param effect: estimated effect size.
    :param standard_error: standard error of the effect; nonpositive or
        missing values yield (nan, nan).
    :param degrees_of_freedom: residual degrees of freedom for the t
        reference distribution.
    :returns: (t_statistic, p_value).
    """
    if not standard_error > 0:
        return np.nan, np.nan
    t_statistic = effect / standard_error
    p_value = 2 * student_t.sf(abs(t_statistic), df=degrees_of_freedom)
    return t_statistic, p_value


def _model_row(result, sector, horizon, specification_name, bandwidth):
    """Build the model-level result row, including the sum-of-coefficients test.

    :param result: fitted PanelOLS result for one specification.
    :param sector: "business" or "household".
    :param horizon: forecast horizon in years.
    :param specification_name: key naming the predictor set.
    :param bandwidth: Driscoll--Kraay lag length used in the fit.
    :returns: dict of model-level statistics; effects are scaled to pp.
    """
    weights = np.ones(len(result.params))
    combined_effect = 100 * result.params.sum()
    combined_variance = weights @ result.cov.to_numpy() @ weights
    combined_standard_error = 100 * np.sqrt(max(combined_variance, 0))
    combined_t_statistic, combined_p_value = _t_test(
        combined_effect,
        combined_standard_error,
        float(result.df_resid),
    )

    return {
        "sector": sector,
        "horizon": horizon,
        "specification": specification_name,
        "n": int(result.nobs),
        "within_r2_pct": 100 * result.rsquared_within,
        "combined_effect_pp": combined_effect,
        "combined_std_error_pp": combined_standard_error,
        "combined_t_stat": combined_t_statistic,
        "combined_p_value": combined_p_value,
        "dk_bandwidth": bandwidth,
    }


def run_baseline_regressions(panel):
    """Estimate every Table 4 baseline specification for both sectors.

    Runs each predictor set (debt only, price only, full, R-zone only) at
    horizons 1-4 with the horizon-specific Driscoll--Kraay bandwidth.

    :param panel: full country-year analysis panel from load_analysis_panel.
    :returns: (coefficient-level frame, model-level frame); effects are scaled
        to pp.
    """
    coefficient_rows = []
    model_rows = []

    for sector in ["business", "household"]:
        sample = panel[panel[f"in_{sector}_sample"]]

        debt_column = f"high_{sector}_debt_growth"
        price_column = f"high_{sector}_price_growth"
        rzone_column = f"rzone_{sector}"

        specifications = {
            "high_debt_only": [debt_column],
            "high_price_only": [price_column],
            "full": [debt_column, price_column, rzone_column],
            "rzone_only": [rzone_column],
        }

        for horizon, bandwidth in DK_BANDWIDTH.items():
            outcome_column = f"crisis_next_{horizon}y"

            for specification_name, predictor_columns in specifications.items():
                result = _fit(sample, outcome_column, predictor_columns, bandwidth,)

                for predictor_column in predictor_columns:
                    coefficient_rows.append(
                        {
                            "sector": sector,
                            "horizon": horizon,
                            "specification": specification_name,
                            "variable": predictor_column,
                            "coefficient_pp": (100 * result.params[predictor_column]),
                            "std_error_pp": (100 * result.std_errors[predictor_column]),
                            "t_stat": result.tstats[predictor_column],
                            "p_value": result.pvalues[predictor_column],
                        }
                    )

                model_rows.append(_model_row(result, sector, horizon, specification_name, bandwidth,)
                )

    return pd.DataFrame(coefficient_rows), pd.DataFrame(model_rows)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    panel = load_analysis_panel(DATA_DIR)
    coefficients, models = run_baseline_regressions(panel)

    coefficients.to_csv(
        OUTPUT_DIR / "table4_baseline_coefficients.csv",
        index=False,
    )
    models.to_csv(
        OUTPUT_DIR / "table4_baseline_models.csv",
        index=False,
    )

    print("Saved country-FE Driscoll-Kraay baseline regression results")
