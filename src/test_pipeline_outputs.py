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
    # The (country, year) key must be unique, and every paper-sample origin
    # must carry all four horizon labels so regressions share a common sample.
    assert not analysis_panel.duplicated(["country_iso3", "year"]).any()
    common = analysis_panel[analysis_panel["in_paper_sample"]]
    assert common["year"].max() == 2012
    assert common[[f"crisis_next_{h}y" for h in range(1, 5)]].notna().all().all()


@pytest.mark.parametrize(
    "sector,published_frequency", [("business", 6.0), ("household", 10.3)]
)
def test_rzone_frequency(analysis_panel, sector, published_frequency):
    # The share of country-years flagged as R-zone in each sector must sit
    # near the paper's published frequency, or our gates are miscalibrated.
    sample = analysis_panel[analysis_panel[f"in_{sector}_sample"]]
    frequency = 100.0 * sample[f"rzone_{sector}"].mean()
    assert frequency == pytest.approx(published_frequency, abs=1.5)


def test_baseline_regressions_keep_a_common_horizon_sample(analysis_panel):
    # All four horizons of a sector's baseline regression must run on the same
    # n, so coefficients are comparable across horizons as in the paper.
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
    # Each extension horse race must compare models on identical samples;
    # otherwise fit differences could just be sample differences.
    for filename, group_columns in [
        ("fragility_regression_models.csv", ["sector", "fragility_measure", "horizon"]),
        ("fragility_continuous_models.csv", ["sector", "fragility_measure", "horizon"]),
        ("fragility_form_checks_models.csv", ["check", "sector", "horizon"]),
        ("rzone_exit_models.csv", ["sample", "sector", "horizon"]),
        ("dynamic_regression_models.csv", ["sector", "horizon"]),
    ]:
        path = OUTPUT_DIR / filename
        if not path.exists():
            pytest.skip(f"{filename} not built yet -- run `doit analysis`")
        models = pd.read_csv(path)
        assert models.groupby(group_columns)["n"].nunique().eq(1).all()


def test_continuous_and_flag_fragility_runs_share_samples():
    # The continuous robustness check must re-fit the extension on exactly the
    # rows the flag version used; a sample difference would make the two
    # verdicts on the amplification claim incomparable.
    flag_path = OUTPUT_DIR / "fragility_regression_models.csv"
    continuous_path = OUTPUT_DIR / "fragility_continuous_models.csv"
    if not flag_path.exists() or not continuous_path.exists():
        pytest.skip("fragility outputs not built yet -- run `doit analysis`")
    keys = ["sector", "fragility_measure", "horizon"]
    flag_n = pd.read_csv(flag_path).groupby(keys)["n"].first()
    continuous_n = pd.read_csv(continuous_path).groupby(keys)["n"].first()
    assert flag_n.eq(continuous_n).all()


def test_curvature_check_runs_on_the_continuous_sample():
    # The quadratic verdict must speak to the continuous specification's
    # exact regression, so the curvature check's sample must match it.
    form_path = OUTPUT_DIR / "fragility_form_checks_models.csv"
    continuous_path = OUTPUT_DIR / "fragility_continuous_models.csv"
    if not form_path.exists() or not continuous_path.exists():
        pytest.skip("fragility form checks not built yet -- run `doit analysis`")
    keys = ["sector", "horizon"]
    form = pd.read_csv(form_path)
    curvature_n = form.loc[form["check"].eq("curvature")].groupby(keys)["n"].first()
    continuous = pd.read_csv(continuous_path)
    continuous_n = (
        continuous.loc[continuous["fragility_measure"].eq("noncore")]
        .groupby(keys)["n"]
        .first()
    )
    assert curvature_n.eq(continuous_n).all()


def test_lev_fragility_flag_marks_thin_capital(analysis_panel):
    # high_lev must implement the proposal's flipped inequality: fragility is
    # thin capital, so every flagged country-year has less capital than every
    # unflagged one and roughly a fifth of the reference sample is flagged.
    historical = analysis_panel[analysis_panel["in_paper_sample"]]
    flagged = historical.loc[historical["high_lev"].eq(1), "lev"]
    unflagged = historical.loc[historical["high_lev"].eq(0), "lev"]
    assert flagged.max() < unflagged.min()
    assert 0.15 < historical["high_lev"].mean() < 0.25


def test_fragility_sweep_is_anchored_at_the_frozen_baseline():
    # The q=80 gridpoint must reproduce the horse-race noncore interaction
    # exactly, and joint cells must shrink as the cutoff rises, so the sweep
    # extends the frozen baseline instead of quietly re-estimating it.
    sweep_path = OUTPUT_DIR / "fragility_threshold_sweep.csv"
    coefficients_path = OUTPUT_DIR / "fragility_regression_coefficients.csv"
    if not sweep_path.exists() or not coefficients_path.exists():
        pytest.skip("fragility sweep not built yet -- run `doit analysis`")
    sweep = pd.read_csv(sweep_path)
    coefficients = pd.read_csv(coefficients_path)
    assert len(sweep) == 2 * 5 * 4
    for (sector, horizon), rows in sweep.groupby(["sector", "horizon"]):
        ordered = rows.sort_values("quantile_pct")
        assert ordered["joint_cell_count"].is_monotonic_decreasing
        baseline = coefficients.loc[
            coefficients["sector"].eq(sector)
            & coefficients["fragility_measure"].eq("noncore")
            & coefficients["horizon"].eq(horizon)
            & coefficients["specification"].eq("fragility_extension")
            & coefficients["variable"].eq(f"rzone_{sector}_x_high_noncore")
        ].iloc[0]
        anchored = ordered.loc[ordered["quantile_pct"].eq(80.0)].iloc[0]
        assert anchored["interaction_pp"] == pytest.approx(
            baseline["coefficient_pp"], rel=1e-9
        )


def test_table1_uses_complete_sector_pair_samples(analysis_panel):
    # Table 1 rows must be computed on complete debt+price pairs, matching
    # the paper's sample rule for each sector.
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
    # The cutoff comparison must cover every quantile the paper publishes per
    # variable, so no R-zone gate is silently dropped from the report.
    quantiles = calculate_quantile_comparison(analysis_panel)
    counts = quantiles.groupby("variable")["quantile"].count().to_dict()
    assert counts == {
        "delta3_business_debt_gdp": 4,
        "delta3_household_debt_gdp": 4,
        "delta3_log_real_equity": 2,
        "delta3_log_real_house_price": 2,
        "delta3_log_real_private_debt": 4,
    }
