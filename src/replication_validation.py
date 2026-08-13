"""Build a machine-readable validation report for all assigned exhibits."""

from pathlib import Path

import numpy as np
import pandas as pd

import exhibit_benchmarks as exhibit
import paper_benchmarks as table1_bench
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


def validate_table1(
    stats: pd.DataFrame, cutoffs: pd.DataFrame | None = None
) -> list[dict]:
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
                    exhibit.TABLE1_MEAN_TOLERANCE_PP,
                ),
                _comparison(
                    "Table 1",
                    "standard_deviation",
                    variable,
                    actual["replicated_sd"],
                    paper_sd,
                    exhibit.TABLE1_SD_TOLERANCE_PP,
                ),
            ]
        )
    if cutoffs is not None:
        cutoff_tolerances = {
            "delta3_business_debt_gdp": table1_bench.TOLERANCE_DEBT_CUTOFF_PP,
            "delta3_household_debt_gdp": table1_bench.TOLERANCE_DEBT_CUTOFF_PP,
            "delta3_log_real_private_debt": (
                table1_bench.TOLERANCE_PRIVATE_DEBT_QUANTILE_PP
            ),
            "delta3_log_real_equity": table1_bench.TOLERANCE_PRICE_CUTOFF_PP,
            "delta3_log_real_house_price": table1_bench.TOLERANCE_PRICE_CUTOFF_PP,
        }
        for cutoff in cutoffs.itertuples(index=False):
            rows.append(
                _comparison(
                    "Table 1",
                    "quantile_cutoff",
                    f"{cutoff.variable}:{cutoff.quantile_label}",
                    cutoff.replicated,
                    cutoff.paper,
                    cutoff_tolerances[cutoff.variable],
                )
            )
    return rows


def validate_table3(results: pd.DataFrame) -> list[dict]:
    rows: list[dict] = []
    indexed = results.set_index(["sector", "horizon", "price_tercile", "debt_quintile"])
    for sector in ["business", "household"]:
        for price in range(1, 4):
            for debt in range(1, 6):
                key = f"{sector}:price={price}:debt={debt}"
                actual = indexed.loc[(sector, 1, price, debt)]
                published = exhibit.TABLE3_DISTRIBUTIONS[sector][price - 1][debt - 1]
                rows.append(
                    _comparison(
                        "Table 3",
                        "distribution_pct",
                        key,
                        actual["distribution_pct"],
                        published,
                        exhibit.TABLE3_DISTRIBUTION_TOLERANCE_PP,
                    )
                )
        for horizon in range(1, 5):
            for price in range(1, 4):
                for debt in range(1, 6):
                    key = f"{sector}:h={horizon}:price={price}:debt={debt}"
                    actual = indexed.loc[(sector, horizon, price, debt)]
                    rows.extend(
                        [
                            _comparison(
                                "Table 3",
                                "crisis_frequency_pct",
                                key,
                                actual["crisis_frequency_pct"],
                                exhibit.TABLE3_FREQUENCIES[sector][horizon][price - 1][
                                    debt - 1
                                ],
                                exhibit.TABLE3_PROBABILITY_TOLERANCE_PP,
                            ),
                            _comparison(
                                "Table 3",
                                "difference_from_median_pct",
                                key,
                                actual["difference_from_median_pct"],
                                exhibit.TABLE3_DIFFERENCES[sector][horizon][price - 1][
                                    debt - 1
                                ],
                                exhibit.TABLE3_PROBABILITY_TOLERANCE_PP,
                            ),
                        ]
                    )
    cell_rows = pd.DataFrame(rows)
    rmse_tolerances = {
        "distribution_pct": exhibit.TABLE3_DISTRIBUTION_RMSE_TOLERANCE_PP,
        "crisis_frequency_pct": exhibit.TABLE3_PROBABILITY_RMSE_TOLERANCE_PP,
        "difference_from_median_pct": (exhibit.TABLE3_PROBABILITY_RMSE_TOLERANCE_PP),
    }
    for statistic, tolerance in rmse_tolerances.items():
        differences = cell_rows.loc[cell_rows["statistic"].eq(statistic), "difference"]
        rows.append(
            _comparison(
                "Table 3",
                f"{statistic}_rmse",
                "all_cells",
                np.sqrt(np.mean(np.square(differences))),
                0.0,
                tolerance,
            )
        )
    return rows


def _coefficient_specs(variable: str) -> list[str]:
    if variable.startswith("rzone_"):
        return ["full", "rzone_only"]
    if "debt_growth" in variable:
        return ["high_debt_only", "full"]
    return ["high_price_only", "full"]


def validate_table4(coefficients: pd.DataFrame, models: pd.DataFrame) -> list[dict]:
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
                    rows.extend(
                        [
                            _comparison(
                                "Table 4",
                                "coefficient_pp",
                                key,
                                actual["coefficient_pp"],
                                published,
                                exhibit.TABLE4_COEFFICIENT_TOLERANCE_PP,
                            ),
                            _comparison(
                                "Table 4",
                                "t_stat",
                                key,
                                actual["t_stat"],
                                exhibit.TABLE4_TSTATS[sector][variable][horizon - 1][
                                    specifications.index(specification)
                                ],
                                exhibit.TABLE4_TSTAT_TOLERANCE,
                            ),
                        ]
                    )

    for sector in ["business", "household"]:
        for horizon in range(1, 5):
            for specification_index, specification in enumerate(SPECIFICATIONS):
                actual = model_index.loc[(sector, horizon, specification)]
                key = f"{sector}:h={horizon}:{specification}"
                rows.extend(
                    [
                        _comparison(
                            "Table 4",
                            "combined_effect_pp",
                            key,
                            actual["combined_effect_pp"],
                            exhibit.TABLE4_COMBINED_EFFECTS[sector][horizon - 1][
                                specification_index
                            ],
                            exhibit.TABLE4_COEFFICIENT_TOLERANCE_PP,
                        ),
                        _comparison(
                            "Table 4",
                            "within_r2_pct",
                            key,
                            actual["within_r2_pct"],
                            exhibit.TABLE4_WITHIN_R2[sector][horizon - 1][
                                specification_index
                            ],
                            exhibit.TABLE4_R2_TOLERANCE_PP,
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
                    exhibit.TABLE4_TSTAT_TOLERANCE,
                )
            )
    cell_rows = pd.DataFrame(rows)
    rmse_tolerances = {
        "coefficient_pp": exhibit.TABLE4_COEFFICIENT_RMSE_TOLERANCE_PP,
        "combined_effect_pp": exhibit.TABLE4_COEFFICIENT_RMSE_TOLERANCE_PP,
        "t_stat": exhibit.TABLE4_TSTAT_RMSE_TOLERANCE,
        "combined_t_stat": exhibit.TABLE4_TSTAT_RMSE_TOLERANCE,
        "within_r2_pct": exhibit.TABLE4_R2_RMSE_TOLERANCE_PP,
    }
    for statistic, tolerance in rmse_tolerances.items():
        differences = cell_rows.loc[cell_rows["statistic"].eq(statistic), "difference"]
        rows.append(
            _comparison(
                "Table 4",
                f"{statistic}_rmse",
                "all_models",
                np.sqrt(np.mean(np.square(differences))),
                0.0,
                tolerance,
            )
        )
    return rows


def _crises_preceded_by_rzone(sample: pd.DataFrame, sector: str) -> int:
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
            "rzone_events": exhibit.FIGURE1_RZONE_COUNT_TOLERANCE,
            "rzone_followed_by_crisis_3y": (exhibit.FIGURE1_FOLLOWED_COUNT_TOLERANCE),
            "crisis_count": exhibit.FIGURE1_CRISIS_COUNT_TOLERANCE,
            "crises_preceded_by_rzone_3y": (exhibit.FIGURE1_PRECEDED_COUNT_TOLERANCE),
            "ppv_pct": exhibit.FIGURE1_PPV_TOLERANCE_PP,
        }
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
            else exhibit.FIGURE3_SHARE_TOLERANCE_PP
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
    panel = pd.read_parquet(DATA_DIR / "rzone_analysis_panel.parquet")
    rows.extend(validate_figure1(panel))
    rows.extend(validate_figure3(pd.read_csv(OUTPUT_DIR / "figure3_global_rzone.csv")))
    return pd.DataFrame(rows)


def main() -> None:
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
        raise RuntimeError(
            f"Replication validation failed:\n{failures.to_string(index=False)}"
        )


if __name__ == "__main__":
    main()
