"""Checks for the separately labelled post-publication exhibit update."""

from pathlib import Path

import pandas as pd
import pytest

from build_analysis_panel import analysis_panel_path, load_analysis_panel
from data_overview import calculate_data_overview
from figure3_global_rzone import annual_rzone_fraction
from settings import config
from updated_crisis_chronology import (
    UPDATED_CRISIS_END_YEAR,
    UPDATED_CRISIS_SOURCE,
    add_updated_crisis_series,
)
from updated_results import LATEST_PREDICTOR_YEAR, updated_coverage, updated_table1

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
PANEL_PATH = analysis_panel_path(DATA_DIR)
FINAL_REPORT_SOURCE = (
    Path(__file__).resolve().parents[1] / "reports" / "final_report.tex"
)

pytestmark = pytest.mark.skipif(
    not PANEL_PATH.exists(), reason="analysis panel not built yet -- run `doit` first"
)


@pytest.fixture(scope="module")
def updated_panel():
    return add_updated_crisis_series(load_analysis_panel(DATA_DIR))


def test_updated_chronology_preserves_bvx_and_has_explicit_extension(updated_panel):
    historical = updated_panel["year"].le(2016)
    pd.testing.assert_series_equal(
        updated_panel.loc[historical, "crisis_updated"],
        updated_panel.loc[historical, "crisis_bvx"],
        check_names=False,
        check_dtype=False,
    )
    extension = updated_panel["year"].between(2017, UPDATED_CRISIS_END_YEAR)
    assert updated_panel.loc[extension, "crisis_updated"].notna().all()
    # The confirmed post-2016 IMF onsets occur outside the paper's 42 countries.
    assert updated_panel.loc[extension, "crisis_updated"].sum() == 0
    assert updated_panel["updated_crisis_source"].eq(UPDATED_CRISIS_SOURCE).all()


def test_updated_future_outcomes_have_horizon_specific_endpoints(updated_panel):
    for horizon in range(1, 5):
        observed = updated_panel.loc[
            updated_panel[f"crisis_updated_next_{horizon}y"].notna(), "year"
        ]
        assert observed.max() == UPDATED_CRISIS_END_YEAR - horizon


def test_coverage_reports_latest_usable_predictor_pairs(updated_panel):
    coverage = updated_coverage(updated_panel).set_index("series")
    assert coverage.loc["business_predictor_pair", "end_year"] == 2025
    assert coverage.loc["household_predictor_pair", "end_year"] == 2025
    for horizon in range(1, 5):
        assert (
            coverage.loc[f"future_crisis_within_{horizon}y", "end_year"]
            == UPDATED_CRISIS_END_YEAR - horizon
        )


def test_updated_table1_recalculates_expanded_statistics_but_freezes_gates(
    updated_panel,
):
    stats, quantiles = updated_table1(updated_panel)
    stats = stats.set_index("variable")
    assert stats.loc["delta3_business_debt_gdp", "end_year"] == 2025
    assert stats.loc["delta3_log_real_equity", "end_year"] == 2025
    assert stats.loc["delta3_household_debt_gdp", "end_year"] == 2025
    assert stats.loc["delta3_log_real_house_price", "end_year"] == 2025
    gates = quantiles.loc[quantiles["used_for_updated_rzone"]]
    assert len(gates) == 4
    expected = {
        "delta3_business_debt_gdp": "business_debt_q80_cutoff",
        "delta3_log_real_equity": "business_price_t667_cutoff",
        "delta3_household_debt_gdp": "household_debt_q80_cutoff",
        "delta3_log_real_house_price": "household_price_t667_cutoff",
    }
    for row in gates.itertuples(index=False):
        frozen = updated_panel[expected[row.variable]].dropna().iloc[0]
        assert row.frozen_historical_rzone_cutoff == pytest.approx(frozen)


def test_historical_and_updated_figure3_windows_are_kept_separate(updated_panel):
    historical = annual_rzone_fraction(updated_panel)
    updated = annual_rzone_fraction(
        updated_panel,
        end_year=LATEST_PREDICTOR_YEAR,
        historical_sample=False,
    )
    assert (historical["year"].min(), historical["year"].max()) == (1950, 2012)
    assert (updated["year"].min(), updated["year"].max()) == (1950, 2025)
    for sector in ["business", "household"]:
        observed = updated[f"{sector}_n"].notna()
        assert updated.loc[observed, f"{sector}_pct"].between(0, 100).all()


def test_generated_updated_tables_encode_valid_forecast_endpoints():
    required = [
        OUTPUT_DIR / "table3_updated_crisis_probabilities.csv",
        OUTPUT_DIR / "table4_updated_models.csv",
    ]
    if not all(path.exists() for path in required):
        pytest.skip("updated exhibits not built -- run `doit analysis:updated`")
    table3 = pd.read_csv(required[0])
    table4 = pd.read_csv(required[1])
    expected = {horizon: 2025 - horizon for horizon in range(1, 5)}
    for frame in [table3, table4]:
        assert frame.groupby("horizon")["forecast_end_year"].nunique().eq(1).all()
        assert (
            frame.groupby("horizon")["forecast_end_year"].first().to_dict() == expected
        )
    assert table3.shape[0] == 2 * 4 * 3 * 5
    assert table4.groupby(["sector", "horizon"])["n"].nunique().eq(1).all()


def test_original_data_overview_uses_complete_pairs_and_valid_percentages(
    updated_panel,
):
    summary = calculate_data_overview(updated_panel)
    assert len(summary) == 4
    assert set(summary["sector"]) == {"business", "household"}
    assert set(zip(summary["start_year"], summary["end_year"], strict=True)) == {
        (1950, 2012),
        (2013, 2025),
    }
    assert summary["coverage_pct"].between(0, 100).all()
    assert summary["rzone_share_pct"].between(0, 100).all()
    for row in summary.itertuples(index=False):
        price = (
            "delta3_log_real_equity"
            if row.sector == "business"
            else "delta3_log_real_house_price"
        )
        expected = updated_panel.loc[
            updated_panel["year"].between(row.start_year, row.end_year)
            & updated_panel[f"delta3_{row.sector}_debt_gdp"].notna()
            & updated_panel[price].notna()
        ].shape[0]
        assert row.country_years == expected


def test_final_report_includes_all_main_exhibits_and_no_code_listings():
    source = FINAL_REPORT_SOURCE.read_text(encoding="utf-8")
    required_inputs = [
        "data_overview_summary.tex",
        "table1_summary_stats.tex",
        "table3_replication.tex",
        "table4_replication.tex",
        "table1_updated.tex",
        "table3_updated.tex",
        "table4_updated.tex",
        "report_fragility_summary.tex",
        "report_missed_crises.tex",
        "report_dynamic_summary.tex",
        "report_tracker_summary.tex",
        "report_usa_case_study.tex",
    ]
    required_figures = [
        "data_overview_figure.jpg",
        "historical_business_rzone_timeline.png",
        "historical_household_rzone_timeline.png",
        "figure3_fraction_countries_rzone.jpg",
        "updated_business_rzone_timeline.png",
        "updated_household_rzone_timeline.png",
        "figure3_fraction_countries_rzone_updated.jpg",
    ]
    for artifact in [*required_inputs, *required_figures]:
        assert artifact in source
    for forbidden in ["lstlisting", r"\begin{verbatim}", r"\begin{minted}"]:
        assert forbidden not in source
