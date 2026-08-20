"""Validate Table 1 using paper-scaled tolerances from the audit module."""

from pathlib import Path

import pytest

import exhibit_benchmarks as exhibit
import paper_benchmarks as bench
from build_analysis_panel import analysis_panel_path, load_analysis_panel
from replication_validation import table1_mean_tolerance, table1_quantile_tolerance
from settings import config

DATA_DIR = config("DATA_DIR")
PANEL_PATH = analysis_panel_path(Path(DATA_DIR))

pytestmark = pytest.mark.skipif(
    not PANEL_PATH.exists(),
    reason="analysis panel not built yet -- run `doit` first",
)


@pytest.fixture(scope="module")
def paper_sample():
    """The country-years satisfying the paper's baseline sample rule."""
    panel = load_analysis_panel(Path(DATA_DIR))
    return panel[panel["in_paper_sample"]]


@pytest.fixture(scope="module")
def business_debt_growth(paper_sample):
    """Business debt growth where the business pair (debt+equity) exists."""
    has_equity = paper_sample["delta3_log_real_equity"].notna()
    return paper_sample.loc[has_equity, "delta3_business_debt_gdp"].dropna()


@pytest.fixture(scope="module")
def household_debt_growth(paper_sample):
    """Household debt growth where the household pair (debt+house) exists."""
    has_house = paper_sample["delta3_log_real_house_price"].notna()
    return paper_sample.loc[has_house, "delta3_household_debt_gdp"].dropna()


@pytest.fixture(scope="module")
def equity_growth(paper_sample):
    """Equity growth where the business pair (debt+equity) exists."""
    has_debt = paper_sample["delta3_business_debt_gdp"].notna()
    return paper_sample.loc[has_debt, "delta3_log_real_equity"].dropna()


@pytest.fixture(scope="module")
def house_price_growth(paper_sample):
    """House price growth where the household pair (debt+house) exists."""
    has_debt = paper_sample["delta3_household_debt_gdp"].notna()
    return paper_sample.loc[has_debt, "delta3_log_real_house_price"].dropna()


@pytest.mark.parametrize(
    "quantile,published",
    list(bench.DEBT_GROWTH_QUANTILES["business"].items()),
)
def test_business_quantiles(business_debt_growth, quantile, published):
    # Business debt quintile cutoffs must match Table 1; Q80 is the gate
    # that decides which country-years enter the business R-zone.
    recomputed = business_debt_growth.quantile(quantile)
    assert recomputed == pytest.approx(
        published, abs=table1_quantile_tolerance("delta3_business_debt_gdp")
    ), f"business debt Q{int(quantile * 100)}: {recomputed:.2f} vs paper {published}"


@pytest.mark.parametrize(
    "quantile,published",
    list(bench.DEBT_GROWTH_QUANTILES["household"].items()),
)
def test_household_quantiles(household_debt_growth, quantile, published):
    # Household debt quintile cutoffs must match Table 1; Q80 is the gate
    # that decides which country-years enter the household R-zone.
    recomputed = household_debt_growth.quantile(quantile)
    assert recomputed == pytest.approx(
        published, abs=table1_quantile_tolerance("delta3_household_debt_gdp")
    ), f"household debt Q{int(quantile * 100)}: {recomputed:.2f} vs paper {published}"


@pytest.mark.parametrize(
    "quantile,published",
    list(bench.PRICE_GROWTH_TERCILES["equity"].items()),
)
def test_equity_terciles(equity_growth, quantile, published):
    # Equity tercile cutoffs must match Table 1; T66.7 gates the business
    # R-zone, and our IFS/JST equity splice is the likeliest divergence.
    recomputed = equity_growth.quantile(quantile)
    assert recomputed == pytest.approx(
        published, abs=table1_quantile_tolerance("delta3_log_real_equity")
    ), f"equity T{quantile * 100:.1f}: {recomputed:.2f} vs paper {published}"


@pytest.mark.parametrize(
    "quantile,published",
    list(bench.PRICE_GROWTH_TERCILES["house_price"].items()),
)
def test_house_price_terciles(house_price_growth, quantile, published):
    # House price tercile cutoffs must match Table 1; T66.7 gates the
    # household R-zone.
    recomputed = house_price_growth.quantile(quantile)
    assert recomputed == pytest.approx(
        published, abs=table1_quantile_tolerance("delta3_log_real_house_price")
    ), f"house T{quantile * 100:.1f}: {recomputed:.2f} vs paper {published}"


def test_country_year_count(paper_sample):
    # The paper-sample country-year count must match the published N, so
    # we rebuilt the paper's sample rather than a similar-looking one.
    published = bench.SAMPLE_SIZES["bvx_crisis_indicator"]
    allowed = published * exhibit.TABLE1_N_TOLERANCE_FRACTION
    assert len(paper_sample) == pytest.approx(published, abs=allowed)


def test_business_pair_count(business_debt_growth):
    # The business debt+equity pair count must match the published N
    # within the allowed fraction.
    published = bench.SAMPLE_SIZES["business_pairs"]
    allowed = published * exhibit.TABLE1_N_TOLERANCE_FRACTION
    assert len(business_debt_growth) == pytest.approx(published, abs=allowed)


def test_household_pair_count(household_debt_growth):
    # The household debt+house pair count must match the published N
    # within the allowed fraction.
    published = bench.SAMPLE_SIZES["household_pairs"]
    allowed = published * exhibit.TABLE1_N_TOLERANCE_FRACTION
    assert len(household_debt_growth) == pytest.approx(published, abs=allowed)


def test_country_count(paper_sample):
    # Argentina (CPI gaps -- the paper used central-bank data) is the
    # single known, documented loss
    n_countries = paper_sample["country_iso3"].nunique()
    assert n_countries >= bench.N_SAMPLE_COUNTRIES - 1


def test_bvx_crisis_frequency(paper_sample):
    # The unconditional BVX crisis rate must match the paper's published
    # mean; it is the ~7% baseline every R-zone probability is judged against.
    recomputed_pct = 100.0 * paper_sample["crisis_bvx"].mean()
    published_pct = bench.CRISIS_INDICATOR_MEANS["crisis_bvx"]
    assert recomputed_pct == pytest.approx(
        published_pct, abs=table1_mean_tolerance("crisis_bvx")
    )


@pytest.mark.parametrize(
    "column", ["bank_equity_crash", "bank_failures", "banking_panic"]
)
def test_crash_failure_panic_rates(paper_sample, column):
    # Crash, failure, and panic rates must match Table 1's descriptive
    # rows; a miss flags panel drift, not a classification change.
    recomputed_pct = 100.0 * paper_sample[column].mean()
    _, published_mean, _ = bench.TABLE1_PUBLISHED_ROWS[column]
    assert recomputed_pct == pytest.approx(
        published_mean, abs=table1_mean_tolerance(column)
    )


def test_real_gdp_growth_mean(paper_sample):
    # Mean real GDP growth must match Table 1's descriptive row, checking
    # the macro side of the panel against the paper's sample.
    recomputed = paper_sample["real_gdp_growth"].mean()
    _, published_mean, _ = bench.TABLE1_PUBLISHED_ROWS["real_gdp_growth"]
    assert recomputed == pytest.approx(
        published_mean, abs=table1_mean_tolerance("real_gdp_growth")
    )


@pytest.mark.parametrize(
    "quantile,published",
    list(bench.PRIVATE_DEBT_LOG_QUANTILES.items()),
)
def test_private_debt_log_quantiles(paper_sample, quantile, published):
    # Total private debt growth quantiles must match Table 1's context
    # rows; these are descriptive and define no R-zone boundary.
    values = paper_sample["delta3_log_real_private_debt"].dropna()
    recomputed = values.quantile(quantile)
    assert recomputed == pytest.approx(
        published, abs=table1_quantile_tolerance("delta3_log_real_private_debt")
    )
