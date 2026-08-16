"""Integration checks for artifacts created by the full ``doit`` pipeline."""

from pathlib import Path

import pandas as pd
import pytest

from build_analysis_panel import analysis_panel_path, load_analysis_panel
from settings import config
from table1_summary_stats import calculate_quantile_comparison, calculate_table1

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
PANEL_PATH = analysis_panel_path(DATA_DIR)

pytestmark = pytest.mark.skipif(
    not PANEL_PATH.exists(), reason="analysis panel not built yet -- run `doit` first"
)


@pytest.fixture(scope="module")
def analysis_panel():
    return load_analysis_panel(DATA_DIR)


def test_analysis_panel_key_and_common_horizon(analysis_panel):
    assert not analysis_panel.duplicated(["country_iso3", "year"]).any()
    common = analysis_panel[analysis_panel["in_paper_sample"]]
    assert common["year"].max() == 2012
    assert common[[f"crisis_next_{h}y" for h in range(1, 5)]].notna().all().all()


@pytest.mark.parametrize(
    "sector,published_frequency", [("business", 6.0), ("household", 10.3)]
)
def test_rzone_frequency(analysis_panel, sector, published_frequency):
    sample = analysis_panel[analysis_panel[f"in_{sector}_sample"]]
    frequency = 100.0 * sample[f"rzone_{sector}"].mean()
    assert frequency == pytest.approx(published_frequency, abs=1.5)


def test_baseline_regressions_keep_a_common_horizon_sample(analysis_panel):
    path = OUTPUT_DIR / "table4_baseline_models.csv"
    if not path.exists():
        pytest.skip("baseline regressions not built yet -- run `doit analysis`")
    models = pd.read_csv(path)
    for sector in ["business", "household"]:
        expected = int(analysis_panel[f"in_{sector}_sample"].sum())
        sector_models = models[models["sector"].eq(sector)]
        assert sector_models["n"].nunique() == 1
        assert sector_models["n"].iloc[0] == expected


def test_extension_comparisons_use_identical_samples():
    for filename, group_columns in [
        ("fragility_regression_models.csv", ["sector", "fragility_measure", "horizon"]),
        ("dynamic_regression_models.csv", ["sector", "horizon"]),
    ]:
        path = OUTPUT_DIR / filename
        if not path.exists():
            pytest.skip(f"{filename} not built yet -- run `doit analysis`")
        models = pd.read_csv(path)
        assert models.groupby(group_columns)["n"].nunique().eq(1).all()


def test_table1_uses_complete_sector_pair_samples(analysis_panel):
    summary = calculate_table1(analysis_panel).set_index("variable")
    assert (
        summary.loc["delta3_business_debt_gdp", "replicated_n"]
        == analysis_panel["in_business_sample"].sum()
    )
    assert (
        summary.loc["delta3_log_real_equity", "replicated_n"]
        == analysis_panel["in_business_sample"].sum()
    )
    assert (
        summary.loc["delta3_household_debt_gdp", "replicated_n"]
        == analysis_panel["in_household_sample"].sum()
    )
    assert (
        summary.loc["delta3_log_real_house_price", "replicated_n"]
        == analysis_panel["in_household_sample"].sum()
    )


def test_table1_includes_all_published_quantiles(analysis_panel):
    quantiles = calculate_quantile_comparison(analysis_panel)
    counts = quantiles.groupby("variable")["quantile"].count().to_dict()
    assert counts == {
        "delta3_business_debt_gdp": 4,
        "delta3_household_debt_gdp": 4,
        "delta3_log_real_equity": 2,
        "delta3_log_real_house_price": 2,
        "delta3_log_real_private_debt": 4,
    }
