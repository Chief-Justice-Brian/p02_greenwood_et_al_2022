"""Exhibit-level validation against all transcribed paper benchmarks."""

from pathlib import Path

import pandas as pd
import pytest

from build_analysis_panel import analysis_panel_path, load_analysis_panel
from replication_validation import (
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


def test_table1_all_published_rows_within_documented_tolerances():
    _assert_all_within_tolerance(
        validate_table1(
            pd.read_csv(OUTPUT_DIR / "table1_stats.csv"),
            pd.read_csv(OUTPUT_DIR / "table1_cutoffs.csv"),
        )
    )


def test_table3_all_cells_within_documented_tolerances():
    _assert_all_within_tolerance(
        validate_table3(pd.read_csv(OUTPUT_DIR / "table3_crisis_probabilities.csv"))
    )


def test_table4_all_coefficients_and_model_statistics_within_tolerances():
    _assert_all_within_tolerance(
        validate_table4(
            pd.read_csv(OUTPUT_DIR / "table4_baseline_coefficients.csv"),
            pd.read_csv(OUTPUT_DIR / "table4_baseline_models.csv"),
        )
    )


def test_figure1_underlying_event_series_within_documented_tolerances():
    _assert_all_within_tolerance(validate_figure1(load_analysis_panel(DATA_DIR)))


def test_figure3_underlying_annual_series_within_documented_tolerances():
    _assert_all_within_tolerance(
        validate_figure3(pd.read_csv(OUTPUT_DIR / "figure3_global_rzone.csv"))
    )
