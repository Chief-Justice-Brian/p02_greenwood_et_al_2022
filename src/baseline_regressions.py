"""Estimate the primary GHSS country-fixed-effects probability models."""

from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PanelOLS
from scipy.stats import t as student_t

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))

# Table 4 uses Driscoll--Kraay lags 0, 3, 5, and 6 at horizons 1--4.
DK_BANDWIDTH = {1: 0, 2: 3, 3: 5, 4: 6}


def _fit(data, outcome, predictors, bandwidth):
    estimation = data[["country_iso3", "year", outcome, *predictors]].dropna().copy()
    estimation = estimation.set_index(["country_iso3", "year"])
    dependent = estimation[outcome].astype(float)
    regressors = estimation[predictors].astype(float)
    model = PanelOLS(
        dependent,
        regressors,
        entity_effects=True,
        drop_absorbed=True,
        check_rank=True,
    )
    return model.fit(cov_type="driscoll-kraay", bandwidth=bandwidth)


def run_baseline_regressions(panel):
    coefficient_rows = []
    model_rows = []
    for sector in ["business", "household"]:
        sample = panel[panel[f"in_{sector}_sample"]].copy()
        debt = f"high_{sector}_debt_growth"
        price = f"high_{sector}_price_growth"
        rzone = f"rzone_{sector}"
        specifications = {
            "high_debt_only": [debt],
            "high_price_only": [price],
            "full": [debt, price, rzone],
            "rzone_only": [rzone],
        }
        for horizon in [1, 2, 3, 4]:
            outcome = f"crisis_next_{horizon}y"
            for specification, predictors in specifications.items():
                result = _fit(sample, outcome, predictors, DK_BANDWIDTH[horizon])
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
                weights = np.ones(len(result.params))
                combined = 100.0 * result.params.sum()
                combined_variance = float(weights @ result.cov.to_numpy() @ weights)
                combined_se = 100.0 * np.sqrt(max(combined_variance, 0.0))
                combined_t = combined / combined_se if combined_se > 0 else np.nan
                combined_p = 2.0 * student_t.sf(
                    abs(combined_t), df=float(result.df_resid)
                )
                model_rows.append(
                    {
                        "sector": sector,
                        "horizon": horizon,
                        "specification": specification,
                        "n": int(result.nobs),
                        "within_r2_pct": 100.0 * result.rsquared_within,
                        "combined_effect_pp": combined,
                        "combined_std_error_pp": combined_se,
                        "combined_t_stat": combined_t,
                        "combined_p_value": combined_p,
                        "dk_bandwidth": DK_BANDWIDTH[horizon],
                    }
                )
    return pd.DataFrame(coefficient_rows), pd.DataFrame(model_rows)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = pd.read_parquet(DATA_DIR / "rzone_analysis_panel.parquet")
    coefficients, models = run_baseline_regressions(panel)
    coefficients.to_csv(OUTPUT_DIR / "table4_baseline_coefficients.csv", index=False)
    models.to_csv(OUTPUT_DIR / "table4_baseline_models.csv", index=False)
    print("Saved country-FE Driscoll-Kraay baseline regression results")
