"""Build post-publication versions of the assigned GHSS exhibits.

Historical replication files are never overwritten. Updated predictor and
R-Zone series run through 2025. Crisis outcomes combine BVX through 2016 with
the IMF Laeven--Valencia (2026) systemic-crisis chronology through 2025, so
the latest valid forecast origins are 2024, 2023, 2022, and 2021 at horizons
one through four.
"""

from pathlib import Path
from textwrap import fill

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from baseline_regressions import DK_BANDWIDTH, _fit
from descriptive_results import (
    combine_event_history_panels,
    plot_event_history,
)
from figure3_global_rzone import annual_rzone_fraction, plot_figure3
from settings import config
from table3_results import (
    SECTOR_COLUMNS,
    _difference_from_median,
)
from table3_results import (
    build_latex as build_table3_latex,
)
from table4_results import build_latex as build_table4_latex
from updated_crisis_chronology import (
    UPDATED_CRISIS_SOURCE,
    add_updated_crisis_series,
)

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
LATEST_PREDICTOR_YEAR = 2025

TABLE1_LABELS = {
    "crisis_updated": "Systemic crisis onset (BVX/IMF) (\\%)",
    "crisis_jst": "Schularick and Taylor (2012) (\\%)",
    "crisis_rr": "Reinhart and Rogoff (2009) (\\%)",
    "bank_equity_crash": "Bank Equity Crash (\\%)",
    "bank_failures": "Bank Failures (\\%)",
    "banking_panic": "Panics (\\%)",
    "real_gdp_growth": "$\\Delta_1$ log real GDP (\\%)",
    "delta3_business_debt_gdp": "$\\Delta_3$ Business Debt / GDP (\\%)",
    "delta3_household_debt_gdp": "$\\Delta_3$ Household Debt / GDP (\\%)",
    "delta3_log_real_private_debt": "$\\Delta_3$ log real Debt (\\%)",
    "delta3_log_real_equity": "$\\Delta_3$ log real Equity Index (\\%)",
    "delta3_log_real_house_price": "$\\Delta_3$ log real House Price Index (\\%)",
}

PERCENT_ROWS = {
    "crisis_updated",
    "crisis_jst",
    "crisis_rr",
    "bank_equity_crash",
    "bank_failures",
    "banking_panic",
}


def _pair_mask(panel: pd.DataFrame, sector: str) -> pd.Series:
    price = (
        "delta3_log_real_equity"
        if sector == "business"
        else "delta3_log_real_house_price"
    )
    return panel[f"delta3_{sector}_debt_gdp"].notna() & panel[price].notna()


def updated_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    specifications = {
        "business_debt_growth": "delta3_business_debt_gdp",
        "real_equity_growth": "delta3_log_real_equity",
        "household_debt_growth": "delta3_household_debt_gdp",
        "real_house_price_growth": "delta3_log_real_house_price",
        "business_rzone": "rzone_business",
        "household_rzone": "rzone_household",
        "updated_systemic_crisis_onset": "crisis_updated",
    }
    rows = []
    for series, column in specifications.items():
        observed = panel.loc[panel[column].notna(), ["country_iso3", "year"]]
        latest = int(observed["year"].max())
        rows.append(
            {
                "series": series,
                "column": column,
                "start_year": int(observed["year"].min()),
                "end_year": latest,
                "observations": len(observed),
                "countries_in_latest_year": int(
                    observed.loc[observed["year"].eq(latest), "country_iso3"].nunique()
                ),
            }
        )
    for sector in ["business", "household"]:
        observed = panel.loc[
            _pair_mask(panel, sector) & panel["year"].le(LATEST_PREDICTOR_YEAR),
            ["country_iso3", "year"],
        ]
        latest = int(observed["year"].max())
        rows.append(
            {
                "series": f"{sector}_predictor_pair",
                "column": "complete debt-price pair",
                "start_year": int(observed["year"].min()),
                "end_year": latest,
                "observations": len(observed),
                "countries_in_latest_year": int(
                    observed.loc[observed["year"].eq(latest), "country_iso3"].nunique()
                ),
            }
        )
    for horizon in range(1, 5):
        column = f"crisis_updated_next_{horizon}y"
        observed = panel.loc[panel[column].notna(), ["country_iso3", "year"]]
        rows.append(
            {
                "series": f"future_crisis_within_{horizon}y",
                "column": column,
                "start_year": int(observed["year"].min()),
                "end_year": int(observed["year"].max()),
                "observations": len(observed),
                "countries_in_latest_year": int(
                    observed.loc[
                        observed["year"].eq(observed["year"].max()), "country_iso3"
                    ].nunique()
                ),
            }
        )
    return pd.DataFrame(rows)


def updated_table1(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recalculate the full Table 1 layout at each series' latest coverage."""
    variable_specs = [
        (None, "crisis_updated"),
        (None, "crisis_jst"),
        (None, "crisis_rr"),
        (None, "bank_equity_crash"),
        (None, "bank_failures"),
        (None, "banking_panic"),
        (None, "real_gdp_growth"),
        ("business", "delta3_business_debt_gdp"),
        ("household", "delta3_household_debt_gdp"),
        (None, "delta3_log_real_private_debt"),
        ("business", "delta3_log_real_equity"),
        ("household", "delta3_log_real_house_price"),
    ]
    rows = []
    for sector, variable in variable_specs:
        mask = panel["year"].le(LATEST_PREDICTOR_YEAR)
        if sector:
            mask &= _pair_mask(panel, sector)
        sample = panel.loc[mask, ["year", variable]].dropna()
        scale = 100.0 if variable in PERCENT_ROWS else 1.0
        rows.append(
            {
                "variable": variable,
                "label": TABLE1_LABELS[variable],
                "n": len(sample),
                "mean": scale * sample[variable].mean(),
                "sd": scale * sample[variable].std(ddof=1),
                "start_year": int(sample["year"].min()),
                "end_year": int(sample["year"].max()),
            }
        )

    quantile_rows = []
    quantile_specs = [
        ("business", "delta3_business_debt_gdp", [0.2, 0.4, 0.6, 0.8]),
        ("household", "delta3_household_debt_gdp", [0.2, 0.4, 0.6, 0.8]),
        (None, "delta3_log_real_private_debt", [0.2, 0.4, 0.6, 0.8]),
        ("business", "delta3_log_real_equity", [1 / 3, 2 / 3]),
        ("household", "delta3_log_real_house_price", [1 / 3, 2 / 3]),
    ]
    for sector, variable, quantiles in quantile_specs:
        mask = panel["year"].le(LATEST_PREDICTOR_YEAR)
        if sector:
            mask &= _pair_mask(panel, sector)
        sample = panel.loc[mask, variable].dropna()
        is_debt_gate = sector is not None and variable.endswith("debt_gdp")
        is_price_gate = sector is not None and (
            "price" in variable or variable.endswith("equity")
        )
        frozen_column = None
        if is_debt_gate:
            frozen_column = f"{sector}_debt_q80_cutoff"
        elif is_price_gate:
            frozen_column = f"{sector}_price_t667_cutoff"
        frozen = (
            float(panel[frozen_column].dropna().iloc[0]) if frozen_column else np.nan
        )
        for quantile in quantiles:
            used_for_rzone = bool(
                (is_debt_gate and np.isclose(quantile, 0.8))
                or (is_price_gate and np.isclose(quantile, 2 / 3))
            )
            quantile_rows.append(
                {
                    "sector": sector,
                    "variable": variable,
                    "label": TABLE1_LABELS[variable],
                    "quantile": quantile,
                    "expanded_sample_quantile": sample.quantile(quantile),
                    "frozen_historical_rzone_cutoff": (
                        frozen if used_for_rzone else np.nan
                    ),
                    "used_for_updated_rzone": used_for_rzone,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(quantile_rows)


def updated_table3(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sector, (price_col, debt_col) in SECTOR_COLUMNS.items():
        predictor_sample = panel.loc[
            _pair_mask(panel, sector) & panel["year"].le(LATEST_PREDICTOR_YEAR)
        ].copy()
        total = len(predictor_sample)
        distribution = (
            predictor_sample.groupby([price_col, debt_col], observed=True)
            .size()
            .to_dict()
        )
        for horizon in range(1, 5):
            outcome = f"crisis_updated_next_{horizon}y"
            sample = predictor_sample.loc[predictor_sample[outcome].notna()].copy()
            for price in range(1, 4):
                for debt in range(1, 6):
                    cell = sample.loc[
                        sample[price_col].eq(price) & sample[debt_col].eq(debt)
                    ]
                    difference, t_stat, p_value = _difference_from_median(
                        sample,
                        outcome,
                        price_col,
                        debt_col,
                        price,
                        debt,
                        DK_BANDWIDTH[horizon],
                    )
                    rows.append(
                        {
                            "sector": sector,
                            "horizon": horizon,
                            "price_tercile": price,
                            "debt_quintile": debt,
                            "n": len(cell),
                            "distribution_pct": 100.0
                            * distribution.get((price, debt), 0)
                            / total,
                            "crisis_frequency_pct": 100.0 * cell[outcome].mean(),
                            "difference_from_median_pct": difference,
                            "difference_t_stat": t_stat,
                            "difference_p_value": p_value,
                            "forecast_end_year": int(sample["year"].max()),
                            "crisis_definition": UPDATED_CRISIS_SOURCE,
                        }
                    )
    return pd.DataFrame(rows)


def updated_table4(panel: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    coefficients = []
    models = []
    for sector in ["business", "household"]:
        debt = f"high_{sector}_debt_growth"
        price = f"high_{sector}_price_growth"
        rzone = f"rzone_{sector}"
        specifications = {
            "high_debt_only": [debt],
            "high_price_only": [price],
            "full": [debt, price, rzone],
            "rzone_only": [rzone],
        }
        pair = _pair_mask(panel, sector)
        for horizon in range(1, 5):
            outcome = f"crisis_updated_next_{horizon}y"
            sample = panel.loc[pair & panel[outcome].notna()].copy()
            for specification, predictors in specifications.items():
                result = _fit(sample, outcome, predictors, DK_BANDWIDTH[horizon])
                for variable in predictors:
                    coefficients.append(
                        {
                            "sector": sector,
                            "horizon": horizon,
                            "specification": specification,
                            "variable": variable,
                            "coefficient_pp": 100.0 * result.params[variable],
                            "std_error_pp": 100.0 * result.std_errors[variable],
                            "t_stat": result.tstats[variable],
                            "p_value": result.pvalues[variable],
                            "forecast_end_year": int(sample["year"].max()),
                            "crisis_definition": UPDATED_CRISIS_SOURCE,
                        }
                    )
                combined = 100.0 * result.params.sum()
                weights = pd.Series(1.0, index=result.params.index)
                variance = float(weights @ result.cov @ weights)
                combined_se = 100.0 * max(variance, 0.0) ** 0.5
                combined_t = combined / combined_se if combined_se > 0 else np.nan
                combined_p = 2.0 * student_t.sf(
                    abs(combined_t), df=float(result.df_resid)
                )
                models.append(
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
                        "forecast_end_year": int(sample["year"].max()),
                        "crisis_definition": UPDATED_CRISIS_SOURCE,
                    }
                )
    return pd.DataFrame(coefficients), pd.DataFrame(models)


def _table1_latex(stats: pd.DataFrame, quantiles: pd.DataFrame) -> str:
    """Render the updated statistics in the paper's grouped Table 1 style."""
    summary = stats.set_index("variable").to_dict(orient="index")
    quantile_lookup = {
        (row.variable, row.quantile): row.expanded_sample_quantile
        for row in quantiles.itertuples(index=False)
    }

    def number(value: float) -> str:
        return "" if pd.isna(value) else f"{value:.2f}"

    def row(variable: str, qs: tuple[float | None, ...] = ()) -> str:
        values = [
            TABLE1_LABELS[variable],
            f"{int(summary[variable]['n'])}",
            number(summary[variable]["mean"]),
            number(summary[variable]["sd"]),
        ]
        values.extend(
            "" if quantile is None else number(quantile_lookup[(variable, quantile)])
            for quantile in qs
        )
        values.extend([""] * (8 - len(values)))
        return " & ".join(values) + r" \\"

    lines = [
        r"\begin{tabular}{rrrr@{\hspace{1.4em}}rrrr}",
        r"\toprule",
        r" & $N$ & Mean & SD & & & & \\",
        r"\cmidrule(lr){2-4}",
        r"\multicolumn{1}{r}{\underline{Financial Crisis Indicators:}} & & & & & & & \\",
        row("crisis_updated"),
        row("crisis_jst"),
        row("crisis_rr"),
        r"\addlinespace",
        r"\multicolumn{1}{r}{\underline{Crashes, Failures and Panics:}} & & & & & & & \\",
        row("bank_equity_crash"),
        row("bank_failures"),
        row("banking_panic"),
        r"\addlinespace",
        r"\multicolumn{1}{r}{\underline{GDP:}} & & & & & & & \\",
        row("real_gdp_growth"),
        r"\addlinespace",
        r" & & & & \multicolumn{4}{c}{Quantiles} \\",
        r"\multicolumn{1}{r}{\underline{Debt Growth:}} & & & & Q20 & Q40 & Q60 & Q80 \\",
        r"\cmidrule(lr){5-8}",
        row("delta3_business_debt_gdp", (0.2, 0.4, 0.6, 0.8)),
        row("delta3_household_debt_gdp", (0.2, 0.4, 0.6, 0.8)),
        row("delta3_log_real_private_debt", (0.2, 0.4, 0.6, 0.8)),
        r"\addlinespace",
        r"\multicolumn{1}{r}{\underline{Price Growth:}} & & & & & Q33.3 & Q66.7 & \\",
        r"\cmidrule(lr){5-8}",
        row("delta3_log_real_equity", (None, 1 / 3, 2 / 3, None)),
        row("delta3_log_real_house_price", (None, 1 / 3, 2 / 3, None)),
        r"\bottomrule",
        r"\end{tabular}",
    ]
    return "\n".join(lines) + "\n"


def build_updated_figures(panel: pd.DataFrame) -> None:
    business = OUTPUT_DIR / "updated_business_rzone_timeline.png"
    household = OUTPUT_DIR / "updated_household_rzone_timeline.png"
    crisis_label = "Systemic crisis onset (BVX/IMF)"
    plot_event_history(
        panel,
        "business",
        business,
        end_year=LATEST_PREDICTOR_YEAR,
        crisis_column="crisis_updated",
        crisis_label=crisis_label,
    )
    plot_event_history(
        panel,
        "household",
        household,
        end_year=LATEST_PREDICTOR_YEAR,
        crisis_column="crisis_updated",
        crisis_label=crisis_label,
    )
    caption = fill(
        "R-Zone markers and predictor coverage extend through 2025. Crisis "
        "onsets use BVX through 2016 and the IMF Laeven-Valencia (2026) "
        "systemic-crisis chronology from 2017 through 2025. Historical "
        "R-Zone thresholds are frozen and are not recalibrated.",
        width=155,
    )
    combine_event_history_panels(
        business,
        household,
        [
            OUTPUT_DIR / "figure1_event_history_updated.png",
            OUTPUT_DIR / "figure1_event_history_updated.pdf",
        ],
        title="Updated Figure 1: Event History through 2025",
        caption=caption,
    )

    annual = annual_rzone_fraction(
        panel,
        start_year=1950,
        end_year=LATEST_PREDICTOR_YEAR,
        historical_sample=False,
    )
    annual.to_csv(OUTPUT_DIR / "figure3_global_rzone_updated.csv", index=False)
    plot_figure3(
        annual,
        OUTPUT_DIR / "figure3_fraction_countries_rzone_updated.png",
        OUTPUT_DIR / "figure3_fraction_countries_rzone_updated.pdf",
        OUTPUT_DIR / "figure3_fraction_countries_rzone_updated.jpg",
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = add_updated_crisis_series(
        pd.read_parquet(DATA_DIR / "rzone_analysis_panel.parquet")
    )
    coverage = updated_coverage(panel)
    coverage.to_csv(OUTPUT_DIR / "updated_data_coverage.csv", index=False)

    table1, table1_quantiles = updated_table1(panel)
    table1.to_csv(OUTPUT_DIR / "table1_updated_stats.csv", index=False)
    table1_quantiles.to_csv(OUTPUT_DIR / "table1_updated_quantiles.csv", index=False)
    (OUTPUT_DIR / "table1_updated.tex").write_text(
        _table1_latex(table1, table1_quantiles), encoding="utf-8"
    )

    table3 = updated_table3(panel)
    table3.to_csv(OUTPUT_DIR / "table3_updated_crisis_probabilities.csv", index=False)
    (OUTPUT_DIR / "table3_updated.tex").write_text(
        build_table3_latex(table3), encoding="utf-8"
    )

    table4_coefficients, table4_models = updated_table4(panel)
    table4_coefficients.to_csv(
        OUTPUT_DIR / "table4_updated_coefficients.csv", index=False
    )
    table4_models.to_csv(OUTPUT_DIR / "table4_updated_models.csv", index=False)
    (OUTPUT_DIR / "table4_updated.tex").write_text(
        build_table4_latex(table4_coefficients, table4_models), encoding="utf-8"
    )

    build_updated_figures(panel)
    print(
        "Saved updated Tables 1/3/4, Figures 1/3, and coverage metadata "
        "without modifying the historical replication outputs"
    )


if __name__ == "__main__":
    main()
