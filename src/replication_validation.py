"""Build a machine-readable validation report for all assigned exhibits."""

from pathlib import Path

import numpy as np
import pandas as pd

import exhibit_benchmarks as exhibit
import paper_benchmarks as table1_bench
from build_analysis_panel import load_analysis_panel
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
SPECIFICATIONS = ["high_debt_only", "high_price_only", "full", "rzone_only"]


def _comparison(
    exhibit_name: str,
    statistic: str,
    key: str,
    replicated: float,
    published: float,
    tolerance: float,
) -> dict[str, float | str | bool]:
    """Build one benchmark-comparison record with a within_tolerance flag.

    :param exhibit_name: exhibit the check belongs to (e.g. "Table 1").
    :param statistic: name of the statistic being compared.
    :param key: identifier for the specific cell or series being checked.
    :param replicated: value produced by our pipeline.
    :param published: value printed in the paper.
    :param tolerance: maximum absolute difference that still counts as a match.
    """
    difference = float(replicated) - float(published)
    return {
        "exhibit": exhibit_name,
        "statistic": statistic,
        "key": key,
        "replicated": float(replicated),
        "published": float(published),
        "tolerance": float(tolerance),
        "difference": difference,
        "absolute_difference": abs(difference),
        "within_tolerance": abs(difference) <= float(tolerance) + 1e-12,
    }


def table1_mean_tolerance(variable: str) -> float:
    """Return 10% of the standard deviation printed for a Table 1 row."""
    published_sd = table1_bench.TABLE1_PUBLISHED_ROWS[variable][2]
    return exhibit.TABLE1_MEAN_SD_FRACTION * published_sd


def table1_sd_tolerance(variable: str) -> float:
    """Return 10% of the standard deviation printed for a Table 1 row."""
    published_sd = table1_bench.TABLE1_PUBLISHED_ROWS[variable][2]
    return exhibit.TABLE1_SD_FRACTION * published_sd


def table1_quantile_tolerance(variable: str) -> float:
    """Return 10% of the relevant published central quantile span."""
    published_quantiles = {
        "delta3_business_debt_gdp": table1_bench.DEBT_GROWTH_QUANTILES["business"],
        "delta3_household_debt_gdp": table1_bench.DEBT_GROWTH_QUANTILES["household"],
        "delta3_log_real_private_debt": table1_bench.PRIVATE_DEBT_LOG_QUANTILES,
        "delta3_log_real_equity": table1_bench.PRICE_GROWTH_TERCILES["equity"],
        "delta3_log_real_house_price": table1_bench.PRICE_GROWTH_TERCILES[
            "house_price"
        ],
    }[variable]
    span = max(published_quantiles.values()) - min(published_quantiles.values())
    return exhibit.TABLE1_QUANTILE_SPAN_FRACTION * span


def _wilson_half_width(percent: float, n: float) -> float:
    """Return a 95% Wilson interval half-width in percentage points."""
    if n <= 0:
        raise ValueError("Wilson interval requires a positive sample size")
    probability = percent / 100.0
    z = exhibit.NORMAL_95_Z
    denominator = 1.0 + z**2 / n
    half_width = (
        z
        / denominator
        * np.sqrt(probability * (1.0 - probability) / n + z**2 / (4.0 * n**2))
    )
    return 100.0 * half_width


def _normalized_rmse_rows(
    cell_rows: pd.DataFrame, exhibit_name: str, statistics: list[str], key: str
) -> list[dict]:
    """Summarize errors relative to their paper-derived cell tolerances."""
    rows = []
    for statistic in statistics:
        selected = cell_rows.loc[cell_rows["statistic"].eq(statistic)]
        normalized = selected["difference"] / selected["tolerance"]
        rows.append(
            _comparison(
                exhibit_name,
                f"{statistic}_normalized_rmse",
                key,
                np.sqrt(np.mean(np.square(normalized))),
                0.0,
                exhibit.NORMALIZED_RMSE_TOLERANCE,
            )
        )
    return rows


def validate_table1(
    stats: pd.DataFrame, cutoffs: pd.DataFrame | None = None
) -> list[dict]:
    """Compare Table 1 summary rows, and optionally quantile cutoffs, to the paper.

    :param stats: calculate_table1 output (table1_stats.csv).
    :param cutoffs: calculate_quantile_comparison output; skipped when None.
    """
    rows: list[dict] = []
    indexed = stats.set_index("variable")
    for variable, (
        paper_n,
        paper_mean,
        paper_sd,
    ) in table1_bench.TABLE1_PUBLISHED_ROWS.items():
        actual = indexed.loc[variable]
        rows.extend(
            [
                _comparison(
                    "Table 1",
                    "N",
                    variable,
                    actual["replicated_n"],
                    paper_n,
                    paper_n * exhibit.TABLE1_N_TOLERANCE_FRACTION,
                ),
                _comparison(
                    "Table 1",
                    "mean",
                    variable,
                    actual["replicated_mean"],
                    paper_mean,
                    table1_mean_tolerance(variable),
                ),
                _comparison(
                    "Table 1",
                    "standard_deviation",
                    variable,
                    actual["replicated_sd"],
                    paper_sd,
                    table1_sd_tolerance(variable),
                ),
            ]
        )
    if cutoffs is not None:
        for cutoff in cutoffs.itertuples(index=False):
            rows.append(
                _comparison(
                    "Table 1",
                    "quantile_cutoff",
                    f"{cutoff.variable}:{cutoff.quantile_label}",
                    cutoff.replicated,
                    cutoff.paper,
                    table1_quantile_tolerance(cutoff.variable),
                )
            )
    return rows


def validate_table3(results: pd.DataFrame) -> list[dict]:
    """Compare every Table 3 cell to the published panels, plus per-statistic RMSEs.

    :param results: calculate_table3 output (table3_crisis_probabilities.csv).
    """
    rows: list[dict] = []
    indexed = results.set_index(["sector", "horizon", "price_tercile", "debt_quintile"])
    for sector in ["business", "household"]:
        for price in range(1, 4):
            for debt in range(1, 6):
                key = f"{sector}:price={price}:debt={debt}"
                actual = indexed.loc[(sector, 1, price, debt)]
                published = exhibit.TABLE3_DISTRIBUTIONS[sector][price - 1][debt - 1]
                distribution_tolerance = _wilson_half_width(
                    published, exhibit.TABLE4_N[sector]
                )
                rows.append(
                    _comparison(
                        "Table 3",
                        "distribution_pct",
                        key,
                        actual["distribution_pct"],
                        published,
                        distribution_tolerance,
                    )
                )
        for horizon in range(1, 5):
            for price in range(1, 4):
                for debt in range(1, 6):
                    key = f"{sector}:h={horizon}:price={price}:debt={debt}"
                    actual = indexed.loc[(sector, horizon, price, debt)]
                    distribution_pct = exhibit.TABLE3_DISTRIBUTIONS[sector][price - 1][
                        debt - 1
                    ]
                    cell_n = exhibit.TABLE4_N[sector] * distribution_pct / 100.0
                    published_frequency = exhibit.TABLE3_FREQUENCIES[sector][horizon][
                        price - 1
                    ][debt - 1]
                    probability_tolerance = _wilson_half_width(
                        published_frequency, cell_n
                    )
                    median_distribution_pct = exhibit.TABLE3_DISTRIBUTIONS[sector][1][2]
                    median_n = (
                        exhibit.TABLE4_N[sector] * median_distribution_pct / 100.0
                    )
                    median_frequency = exhibit.TABLE3_FREQUENCIES[sector][horizon][1][2]
                    difference_tolerance = np.hypot(
                        probability_tolerance,
                        _wilson_half_width(median_frequency, median_n),
                    )
                    rows.extend(
                        [
                            _comparison(
                                "Table 3",
                                "crisis_frequency_pct",
                                key,
                                actual["crisis_frequency_pct"],
                                published_frequency,
                                probability_tolerance,
                            ),
                            _comparison(
                                "Table 3",
                                "difference_from_median_pct",
                                key,
                                actual["difference_from_median_pct"],
                                exhibit.TABLE3_DIFFERENCES[sector][horizon][price - 1][
                                    debt - 1
                                ],
                                difference_tolerance,
                            ),
                        ]
                    )
    cell_rows = pd.DataFrame(rows)
    rows.extend(
        _normalized_rmse_rows(
            cell_rows,
            "Table 3",
            ["distribution_pct", "crisis_frequency_pct", "difference_from_median_pct"],
            "all_cells",
        )
    )
    return rows


def _coefficient_specs(variable: str) -> list[str]:
    """Return the two specifications whose published columns report this variable.

    :param variable: predictor column name (high-debt, high-price, or rzone
        indicator).
    :returns: the two matching specification names.
    """
    if variable.startswith("rzone_"):
        return ["full", "rzone_only"]
    if "debt_growth" in variable:
        return ["high_debt_only", "full"]
    return ["high_price_only", "full"]


def _table4_typical_standard_error() -> float:
    """Infer the median identified standard error from published Table 4."""
    standard_errors = []
    for sector, variables in exhibit.TABLE4_COEFFICIENTS.items():
        for variable, horizon_values in variables.items():
            for coefficients, t_stats in zip(
                horizon_values, exhibit.TABLE4_TSTATS[sector][variable], strict=True
            ):
                for coefficient, t_stat in zip(coefficients, t_stats, strict=True):
                    if abs(coefficient) > 0 and abs(t_stat) >= 0.1:
                        standard_errors.append(abs(coefficient / t_stat))
    return float(np.median(standard_errors))


TABLE4_TYPICAL_STANDARD_ERROR = _table4_typical_standard_error()


def _coefficient_tolerance(coefficient: float, t_stat: float) -> float:
    """Allow 1.5 published standard errors for a Table 4 coefficient."""
    standard_error = (
        abs(coefficient / t_stat)
        if abs(coefficient) > 0 and abs(t_stat) >= 0.1
        else TABLE4_TYPICAL_STANDARD_ERROR
    )
    return exhibit.TABLE4_COEFFICIENT_SE_MULTIPLIER * standard_error


def _t_stat_tolerance(t_stat: float) -> float:
    """Scale t-statistic tolerance to 25% of its published magnitude."""
    return max(
        exhibit.TABLE4_TSTAT_ROUNDING_FLOOR,
        exhibit.TABLE4_TSTAT_RELATIVE_TOLERANCE * abs(t_stat),
    )


def _r2_tolerance(within_r2: float) -> float:
    """Scale within-R2 tolerance to 25% of its published magnitude."""
    return max(
        exhibit.TABLE4_R2_ROUNDING_FLOOR_PP,
        exhibit.TABLE4_R2_RELATIVE_TOLERANCE * abs(within_r2),
    )


def _combined_t_stat(sector: str, horizon: int, specification: str) -> float:
    """Return the published t-statistic matching a combined-effect column."""
    if specification == "full":
        return exhibit.TABLE4_COMBINED_TSTATS[sector][horizon - 1]
    if specification == "high_debt_only":
        variable = f"high_{sector}_debt_growth"
        position = 0
    elif specification == "high_price_only":
        variable = f"high_{sector}_price_growth"
        position = 0
    else:
        variable = f"rzone_{sector}"
        position = 1
    return exhibit.TABLE4_TSTATS[sector][variable][horizon - 1][position]


def validate_table4(coefficients: pd.DataFrame, models: pd.DataFrame) -> list[dict]:
    """Compare Table 4 coefficients, t-stats, and model statistics to the paper.

    Also appends per-statistic RMSE checks across all cells.

    :param coefficients: the coefficient-level frame from
        run_baseline_regressions.
    :param models: the model-level frame from run_baseline_regressions.
    """
    rows: list[dict] = []
    coef_index = coefficients.set_index(
        ["sector", "horizon", "specification", "variable"]
    )
    model_index = models.set_index(["sector", "horizon", "specification"])

    for sector, variables in exhibit.TABLE4_COEFFICIENTS.items():
        for variable, horizon_values in variables.items():
            specifications = _coefficient_specs(variable)
            for horizon, published_values in enumerate(horizon_values, start=1):
                for specification, published in zip(
                    specifications, published_values, strict=True
                ):
                    actual = coef_index.loc[(sector, horizon, specification, variable)]
                    key = f"{sector}:h={horizon}:{specification}:{variable}"
                    published_t_stat = exhibit.TABLE4_TSTATS[sector][variable][
                        horizon - 1
                    ][specifications.index(specification)]
                    rows.extend(
                        [
                            _comparison(
                                "Table 4",
                                "coefficient_pp",
                                key,
                                actual["coefficient_pp"],
                                published,
                                _coefficient_tolerance(published, published_t_stat),
                            ),
                            _comparison(
                                "Table 4",
                                "t_stat",
                                key,
                                actual["t_stat"],
                                published_t_stat,
                                _t_stat_tolerance(published_t_stat),
                            ),
                        ]
                    )

    for sector in ["business", "household"]:
        for horizon in range(1, 5):
            for specification_index, specification in enumerate(SPECIFICATIONS):
                actual = model_index.loc[(sector, horizon, specification)]
                key = f"{sector}:h={horizon}:{specification}"
                published_effect = exhibit.TABLE4_COMBINED_EFFECTS[sector][horizon - 1][
                    specification_index
                ]
                published_combined_t = _combined_t_stat(sector, horizon, specification)
                published_r2 = exhibit.TABLE4_WITHIN_R2[sector][horizon - 1][
                    specification_index
                ]
                rows.extend(
                    [
                        _comparison(
                            "Table 4",
                            "combined_effect_pp",
                            key,
                            actual["combined_effect_pp"],
                            published_effect,
                            _coefficient_tolerance(
                                published_effect, published_combined_t
                            ),
                        ),
                        _comparison(
                            "Table 4",
                            "within_r2_pct",
                            key,
                            actual["within_r2_pct"],
                            published_r2,
                            _r2_tolerance(published_r2),
                        ),
                        _comparison(
                            "Table 4",
                            "N",
                            key,
                            actual["n"],
                            exhibit.TABLE4_N[sector],
                            exhibit.TABLE4_N[sector]
                            * exhibit.TABLE4_N_TOLERANCE_FRACTION,
                        ),
                    ]
                )
            full = model_index.loc[(sector, horizon, "full")]
            rows.append(
                _comparison(
                    "Table 4",
                    "combined_t_stat",
                    f"{sector}:h={horizon}:full",
                    full["combined_t_stat"],
                    exhibit.TABLE4_COMBINED_TSTATS[sector][horizon - 1],
                    _t_stat_tolerance(
                        exhibit.TABLE4_COMBINED_TSTATS[sector][horizon - 1]
                    ),
                )
            )
    cell_rows = pd.DataFrame(rows)
    rows.extend(
        _normalized_rmse_rows(
            cell_rows,
            "Table 4",
            [
                "coefficient_pp",
                "combined_effect_pp",
                "t_stat",
                "combined_t_stat",
                "within_r2_pct",
            ],
            "all_models",
        )
    )
    return rows


def _crises_preceded_by_rzone(sample: pd.DataFrame, sector: str) -> int:
    """Count BVX crisis years within three years after an R-Zone country-year.

    :param sample: one sector's in-sample slice of the analysis panel.
    :param sector: "business" or "household".
    """
    rzone_origins = sample.loc[
        sample[f"rzone_{sector}"].eq(1) & sample["crisis_next_3y"].eq(1),
        ["country_iso3", "year"],
    ]
    covered_crises = {
        (origin.country_iso3, year)
        for origin in rzone_origins.itertuples()
        for year in range(int(origin.year) + 1, int(origin.year) + 4)
    }
    crises = sample.loc[sample["crisis_bvx"].eq(1), ["country_iso3", "year"]]
    return sum(
        (crisis.country_iso3, int(crisis.year)) in covered_crises
        for crisis in crises.itertuples()
    )


def validate_figure1(panel: pd.DataFrame) -> list[dict]:
    """Compare Figure 1 sector aggregates and key crisis-marker years to the paper.

    :param panel: full country-year analysis panel from load_analysis_panel.
    """
    rows: list[dict] = []
    for sector, published in exhibit.FIGURE1_AGGREGATES.items():
        sample = panel.loc[panel[f"in_{sector}_sample"]]
        rzone = sample[f"rzone_{sector}"].eq(1)
        actual = {
            "rzone_events": int(rzone.sum()),
            "rzone_followed_by_crisis_3y": int(
                (rzone & sample["crisis_next_3y"].eq(1)).sum()
            ),
            "crisis_count": int(sample["crisis_bvx"].eq(1).sum()),
            "crises_preceded_by_rzone_3y": _crises_preceded_by_rzone(sample, sector),
            "ppv_pct": 100.0 * sample.loc[rzone, "crisis_next_3y"].mean(),
        }
        tolerances = {
            statistic: max(
                1,
                np.ceil(
                    exhibit.FIGURE1_COUNT_TOLERANCE_FRACTION * published[statistic]
                ),
            )
            for statistic in [
                "rzone_events",
                "rzone_followed_by_crisis_3y",
                "crisis_count",
                "crises_preceded_by_rzone_3y",
            ]
        }
        tolerances["ppv_pct"] = _wilson_half_width(
            published["ppv_pct"], published["rzone_events"]
        )
        for statistic, published_value in published.items():
            rows.append(
                _comparison(
                    "Figure 1",
                    statistic,
                    sector,
                    actual[statistic],
                    published_value,
                    tolerances[statistic],
                )
            )

    historical = panel.loc[panel["year"].between(1950, 2016)]
    for country, published_years in exhibit.FIGURE1_KEY_CRISIS_YEARS.items():
        actual_years = sorted(
            historical.loc[
                historical["country_iso3"].eq(country) & historical["crisis_bvx"].eq(1),
                "year",
            ].astype(int)
        )
        rows.append(
            _comparison(
                "Figure 1",
                "key_crisis_marker_count",
                country,
                len(actual_years),
                len(published_years),
                0,
            )
        )
        for position, published_year in enumerate(published_years):
            actual_year = actual_years[position] if position < len(actual_years) else -1
            rows.append(
                _comparison(
                    "Figure 1",
                    "key_crisis_marker_year",
                    f"{country}:{position + 1}",
                    actual_year,
                    published_year,
                    0,
                )
            )
    return rows


def validate_figure3(annual: pd.DataFrame) -> list[dict]:
    """Compare Figure 3 peaks, era maxima, and year range to published benchmarks.

    :param annual: annual_rzone_fraction output (figure3_global_rzone.csv).
    """
    rows: list[dict] = []
    business_peak = annual.loc[annual["business_pct"].idxmax()]
    household_peak = annual.loc[annual["household_pct"].idxmax()]
    values = {
        "business_peak_pct": business_peak["business_pct"],
        "business_peak_year": business_peak["year"],
        "household_peak_pct": household_peak["household_pct"],
        "household_peak_year": household_peak["year"],
        "business_1950s_max_pct": annual.loc[
            annual["year"].between(1950, 1959), "business_pct"
        ].max(),
        "household_1970s_max_pct": annual.loc[
            annual["year"].between(1970, 1979), "household_pct"
        ].max(),
        "household_2002_2008_max_pct": annual.loc[
            annual["year"].between(2002, 2008), "household_pct"
        ].max(),
    }
    for statistic, published in exhibit.FIGURE3_BENCHMARKS.items():
        tolerance = (
            exhibit.FIGURE3_PEAK_YEAR_TOLERANCE
            if statistic.endswith("year")
            else 100.0
            * exhibit.FIGURE3_COUNTRY_MISMATCH_TOLERANCE
            / exhibit.FIGURE3_SAMPLE_COUNTRIES
        )
        rows.append(
            _comparison(
                "Figure 3",
                statistic,
                "historical_series",
                values[statistic],
                published,
                tolerance,
            )
        )
    rows.extend(
        [
            _comparison(
                "Figure 3",
                "first_year",
                "historical_series",
                annual.year.min(),
                1950,
                0,
            ),
            _comparison(
                "Figure 3", "last_year", "historical_series", annual.year.max(), 2012, 0
            ),
        ]
    )
    return rows


def build_validation_report() -> pd.DataFrame:
    """Assemble every exhibit's benchmark comparisons from the saved outputs."""
    rows: list[dict] = []
    rows.extend(
        validate_table1(
            pd.read_csv(OUTPUT_DIR / "table1_stats.csv"),
            pd.read_csv(OUTPUT_DIR / "table1_cutoffs.csv"),
        )
    )
    rows.extend(
        validate_table3(pd.read_csv(OUTPUT_DIR / "table3_crisis_probabilities.csv"))
    )
    rows.extend(
        validate_table4(
            pd.read_csv(OUTPUT_DIR / "table4_baseline_coefficients.csv"),
            pd.read_csv(OUTPUT_DIR / "table4_baseline_models.csv"),
        )
    )
    panel = load_analysis_panel(DATA_DIR)
    rows.extend(validate_figure1(panel))
    rows.extend(validate_figure3(pd.read_csv(OUTPUT_DIR / "figure3_global_rzone.csv")))
    return pd.DataFrame(rows)


def main() -> None:
    """Write the validation CSV and report any out-of-tolerance comparisons."""
    report = build_validation_report()
    path = OUTPUT_DIR / "replication_validation.csv"
    report.to_csv(path, index=False)
    summary = report.groupby("exhibit")["within_tolerance"].agg(["sum", "count"])
    print(summary.to_string())
    print(f"Saved {len(report):,} benchmark comparisons to {path}")
    if not report["within_tolerance"].all():
        failures = report.loc[
            ~report["within_tolerance"],
            ["exhibit", "statistic", "key", "replicated", "published", "tolerance"],
        ]
        print(f"Out-of-tolerance comparisons:\n{failures.to_string(index=False)}")


if __name__ == "__main__":
    main()
