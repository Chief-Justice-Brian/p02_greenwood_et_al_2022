"""Validation Gate: our replicated panel must match the paper's Table 1.

Each test recomputes one published statistic from our spliced panel and
compares it against Greenwood, Hanson, Shleifer and Sorensen (WP 20-130,
Table 1). "Match" means within that statistic's tolerance, and every published
value and every tolerance comes from ``paper_benchmarks.py`` -- nothing from
the paper is hard-coded here. Exact agreement is impossible: our sources are
revised vintages of the paper's, and our equity series substitutes IFS/JST/OECD
for the paywalled GFD.

What each class pins:

``TestDebtGrowthCutoffs``, ``TestPriceGrowthCutoffs``
    The quantile cutoffs. Four are the R-zone gates themselves -- business and
    household debt Q80, equity and house price T66.7 -- so a failure there
    changes which country-years count as overheated and invalidates every
    downstream exhibit. The other eight are context.
``TestSampleComposition``
    That we rebuilt the paper's sample rather than a similar one: country-year
    counts, country count, and the unconditional BVX crisis rate. The country
    count deliberately allows 41 of 42 -- Argentina is a known, documented
    loss, since the paper hand-sourced its CPI from the central bank.
``TestTable1ContextRows``
    The descriptive rows: crashes, failures, panics, GDP growth, total debt
    quantiles. A failure here is informative, not structural.

The tests need the built panel, so ``doit`` must have run the pull and tidy
stages first. Without it they skip with a clear message rather than fail, so a
bare ``pytest`` on a fresh clone reports no false alarms.
"""

from pathlib import Path

import pytest

import paper_benchmarks as bench
from build_analysis_panel import analysis_panel_path, load_analysis_panel
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
    has_debt = paper_sample["delta3_business_debt_gdp"].notna()
    return paper_sample.loc[has_debt, "delta3_log_real_equity"].dropna()


@pytest.fixture(scope="module")
def house_price_growth(paper_sample):
    has_debt = paper_sample["delta3_household_debt_gdp"].notna()
    return paper_sample.loc[has_debt, "delta3_log_real_house_price"].dropna()


class TestDebtGrowthCutoffs:
    """Debt-growth quintile cutoffs vs Table 1 (esp. the Q80 R-zone gates)."""

    @pytest.mark.parametrize(
        "quantile,published",
        list(bench.DEBT_GROWTH_QUANTILES["business"].items()),
    )
    def test_business_quantiles(self, business_debt_growth, quantile, published):
        recomputed = business_debt_growth.quantile(quantile)
        assert recomputed == pytest.approx(
            published, abs=bench.TOLERANCE_DEBT_CUTOFF_PP
        ), f"business debt Q{int(quantile * 100)}: {recomputed:.2f} vs paper {published}"

    @pytest.mark.parametrize(
        "quantile,published",
        list(bench.DEBT_GROWTH_QUANTILES["household"].items()),
    )
    def test_household_quantiles(self, household_debt_growth, quantile, published):
        recomputed = household_debt_growth.quantile(quantile)
        assert recomputed == pytest.approx(
            published, abs=bench.TOLERANCE_DEBT_CUTOFF_PP
        ), f"household debt Q{int(quantile * 100)}: {recomputed:.2f} vs paper {published}"


class TestPriceGrowthCutoffs:
    """Price-growth tercile cutoffs vs Table 1 (the R-zone price gates)."""

    @pytest.mark.parametrize(
        "quantile,published",
        list(bench.PRICE_GROWTH_TERCILES["equity"].items()),
    )
    def test_equity_terciles(self, equity_growth, quantile, published):
        recomputed = equity_growth.quantile(quantile)
        assert recomputed == pytest.approx(
            published, abs=bench.TOLERANCE_PRICE_CUTOFF_PP
        ), f"equity T{quantile * 100:.1f}: {recomputed:.2f} vs paper {published}"

    @pytest.mark.parametrize(
        "quantile,published",
        list(bench.PRICE_GROWTH_TERCILES["house_price"].items()),
    )
    def test_house_price_terciles(self, house_price_growth, quantile, published):
        recomputed = house_price_growth.quantile(quantile)
        assert recomputed == pytest.approx(
            published, abs=bench.TOLERANCE_PRICE_CUTOFF_PP
        ), f"house T{quantile * 100:.1f}: {recomputed:.2f} vs paper {published}"


class TestSampleComposition:
    """Panel dimensions vs the paper's reported sample."""

    def test_country_year_count(self, paper_sample):
        published = bench.SAMPLE_SIZES["bvx_crisis_indicator"]
        allowed = published * bench.TOLERANCE_SAMPLE_SIZE_FRACTION
        assert len(paper_sample) == pytest.approx(published, abs=allowed)

    def test_business_pair_count(self, business_debt_growth):
        published = bench.SAMPLE_SIZES["business_pairs"]
        allowed = published * bench.TOLERANCE_SAMPLE_SIZE_FRACTION
        assert len(business_debt_growth) == pytest.approx(published, abs=allowed)

    def test_household_pair_count(self, household_debt_growth):
        published = bench.SAMPLE_SIZES["household_pairs"]
        allowed = published * bench.TOLERANCE_SAMPLE_SIZE_FRACTION
        assert len(household_debt_growth) == pytest.approx(published, abs=allowed)

    def test_country_count(self, paper_sample):
        # Argentina (CPI gaps -- the paper used central-bank data) is the
        # single known, documented loss
        n_countries = paper_sample["country_iso3"].nunique()
        assert n_countries >= bench.N_SAMPLE_COUNTRIES - 1

    def test_bvx_crisis_frequency(self, paper_sample):
        recomputed_pct = 100.0 * paper_sample["crisis_bvx"].mean()
        published_pct = bench.CRISIS_INDICATOR_MEANS["crisis_bvx"]
        assert recomputed_pct == pytest.approx(
            published_pct, abs=bench.TOLERANCE_CRISIS_RATE_PP
        )


class TestTable1ContextRows:
    """The remaining Table 1 rows: crashes/failures/panics, GDP, total debt.

    These are descriptive rather than structural -- none of them defines an
    R-zone boundary -- so a failure here says the panel drifted, not that the
    classification changed. Tolerances live in ``paper_benchmarks.py``.
    """

    @pytest.mark.parametrize(
        "column", ["bank_equity_crash", "bank_failures", "banking_panic"]
    )
    def test_crash_failure_panic_rates(self, paper_sample, column):
        recomputed_pct = 100.0 * paper_sample[column].mean()
        _, published_mean, _ = bench.TABLE1_PUBLISHED_ROWS[column]
        assert recomputed_pct == pytest.approx(
            published_mean, abs=bench.TOLERANCE_INDICATOR_RATE_PP
        )

    def test_real_gdp_growth_mean(self, paper_sample):
        recomputed = paper_sample["real_gdp_growth"].mean()
        _, published_mean, _ = bench.TABLE1_PUBLISHED_ROWS["real_gdp_growth"]
        assert recomputed == pytest.approx(
            published_mean, abs=bench.TOLERANCE_GDP_GROWTH_PP
        )

    @pytest.mark.parametrize(
        "quantile,published",
        list(bench.PRIVATE_DEBT_LOG_QUANTILES.items()),
    )
    def test_private_debt_log_quantiles(self, paper_sample, quantile, published):
        values = paper_sample["delta3_log_real_private_debt"].dropna()
        recomputed = values.quantile(quantile)
        assert recomputed == pytest.approx(
            published, abs=bench.TOLERANCE_PRIVATE_DEBT_QUANTILE_PP
        )
