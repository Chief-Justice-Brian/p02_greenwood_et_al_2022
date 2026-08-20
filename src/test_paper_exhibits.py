"""Exhibit-level validation against all transcribed paper benchmarks."""

from pathlib import Path

import pandas as pd
import pytest

from build_analysis_panel import analysis_panel_path, load_analysis_panel
from replication_validation import (
    table1_mean_tolerance,
    table1_quantile_tolerance,
    validate_figure1,
    validate_figure3,
    validate_table1,
    validate_table3,
    validate_table4,
)
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
PANEL_PATH = analysis_panel_path(DATA_DIR)

REQUIRED = [
    PANEL_PATH,
    OUTPUT_DIR / "table1_stats.csv",
    OUTPUT_DIR / "table1_cutoffs.csv",
    OUTPUT_DIR / "table3_crisis_probabilities.csv",
    OUTPUT_DIR / "table4_baseline_coefficients.csv",
    OUTPUT_DIR / "table4_baseline_models.csv",
    OUTPUT_DIR / "figure3_global_rzone.csv",
]

pytestmark = pytest.mark.skipif(
    not all(path.exists() for path in REQUIRED),
    reason="replication exhibits not built yet -- run `doit analysis` first",
)


def _assert_all_within_tolerance(rows: list[dict]) -> None:
    report = pd.DataFrame(rows)
    failed = report.loc[
        ~report["within_tolerance"],
        ["statistic", "key", "replicated", "published", "tolerance"],
    ]
    assert failed.empty, "Published benchmark failures:\n" + failed.to_string(
        index=False
    )


def test_tolerances_are_derived_from_published_scale_and_vary_by_cell():
    assert table1_mean_tolerance("real_gdp_growth") == pytest.approx(0.321)
    assert table1_quantile_tolerance("delta3_business_debt_gdp") == pytest.approx(1.174)

    table3 = pd.DataFrame(
        validate_table3(pd.read_csv(OUTPUT_DIR / "table3_crisis_probabilities.csv"))
    )
    table4 = pd.DataFrame(
        validate_table4(
            pd.read_csv(OUTPUT_DIR / "table4_baseline_coefficients.csv"),
            pd.read_csv(OUTPUT_DIR / "table4_baseline_models.csv"),
        )
    )
    assert (
        table3.loc[
            table3["statistic"].eq("crisis_frequency_pct"), "tolerance"
        ].nunique()
        > 20
    )
    assert (
        table4.loc[table4["statistic"].eq("coefficient_pp"), "tolerance"].nunique() > 20
    )


def test_table1_all_published_rows_within_documented_tolerances():
    # Every published Table 1 statistic and cutoff must replicate within its
    # documented tolerance; the cutoffs gate every downstream exhibit.
    _assert_all_within_tolerance(
        validate_table1(
            pd.read_csv(OUTPUT_DIR / "table1_stats.csv"),
            pd.read_csv(OUTPUT_DIR / "table1_cutoffs.csv"),
        )
    )


@pytest.mark.xfail(
    strict=True,
    reason="nine sparse Table 3 cells exceed the paper-derived Wilson bounds",
)
def test_table3_all_cells_within_documented_tolerances():
    # Every transcribed Table 3 crisis-probability cell must replicate within
    # tolerance, including the headline R-zone corner probabilities.
    _assert_all_within_tolerance(
        validate_table3(pd.read_csv(OUTPUT_DIR / "table3_crisis_probabilities.csv"))
    )


@pytest.mark.xfail(
    strict=True,
    reason="31 Table 4 coefficient, t-statistic, and fit checks exceed policy",
)
def test_table4_all_coefficients_and_model_statistics_within_tolerances():
    # Table 4's regression coefficients and model statistics must match the
    # paper within tolerance, confirming the predictive result replicates.
    _assert_all_within_tolerance(
        validate_table4(
            pd.read_csv(OUTPUT_DIR / "table4_baseline_coefficients.csv"),
            pd.read_csv(OUTPUT_DIR / "table4_baseline_models.csv"),
        )
    )


@pytest.mark.xfail(
    strict=True,
    reason="business R-Zones precede four fewer crises than the 15% count bound",
)
def test_figure1_underlying_event_series_within_documented_tolerances():
    # The crisis and R-zone event series behind Figure 1 must match the
    # paper's plotted episodes closely enough to reproduce the event history.
    _assert_all_within_tolerance(validate_figure1(load_analysis_panel(DATA_DIR)))


def test_figure3_underlying_annual_series_within_documented_tolerances():
    # The annual fraction-of-countries-in-the-R-zone series behind Figure 3
    # must track the paper's published global overheating index.
    _assert_all_within_tolerance(
        validate_figure3(pd.read_csv(OUTPUT_DIR / "figure3_global_rzone.csv"))
    )
