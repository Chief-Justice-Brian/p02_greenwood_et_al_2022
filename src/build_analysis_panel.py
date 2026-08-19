"""Build the unified historical and post-2016 R-zone analysis panel.

Merges the crisis, credit, equity, and house-price panels onto a full
country-year grid and adds the analysis variables: forward crisis windows,
sample flags, quantile assignments, R-zone indicators, and the JST bank
fragility features (noncore, ltd, lev) with their high-fragility flags (Q80
for noncore and ltd, below-Q20 for the lev capital ratio) and R-zone
interactions. Two conventions drive every downstream exhibit: quantile
cutoffs are estimated only on ``in_paper_sample`` rows and stored as frozen
columns, so post-2016 data can never move the historical thresholds; and a
forecast origin joins the paper sample only when its full four-year outcome
window is observed, which ends the common window at
``PAPER_SAMPLE_END_YEAR - 4``.
Output: ``rzone_analysis_panel.parquet``, one row per (country, year).
"""

from pathlib import Path

import pandas as pd

from clean_credit_panel import load_credit_panel
from clean_crisis_chronologies import load_crisis_panel
from clean_equity_panel import load_equity_panel
from clean_house_price_panel import load_house_price_panel
from country_sample import GHSS_COUNTRIES
from pull_jst_macrohistory import load_jst_macrohistory
from rzone_features import (
    add_future_event_windows,
    assign_bucket,
    indicator_above,
    indicator_below,
    quantile_cutoffs,
)
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
START_YEAR = config("START_DATE").year
END_YEAR = config("END_DATE").year
PAPER_SAMPLE_START_YEAR = config("PAPER_SAMPLE_START_YEAR")
PAPER_SAMPLE_END_YEAR = config("PAPER_SAMPLE_END_YEAR")
LAST_COMMON_FORECAST_YEAR = PAPER_SAMPLE_END_YEAR - 4
ANALYSIS_PANEL_FILENAME = "rzone_analysis_panel.parquet"


def latest_predictor_year(panel):
    """Last year with a complete debt/price predictor pair in either sector.

    Deliberately behind END_YEAR: END_DATE says how far the pulls ask (kept
    ahead so new source data lands without config edits), while this measures
    what the vintage actually delivers, which labels every exhibit window.
    Derived from the panel so labels can never disagree with the data; the
    one pinned year in the test suite is the tripwire that fails loudly when
    a new vintage moves this.

    :param panel: country-year analysis panel from load_analysis_panel.
    :returns: the year, as a plain int.
    """
    business_pair = (
        panel["delta3_business_debt_gdp"].notna()
        & panel["delta3_log_real_equity"].notna()
    )
    household_pair = (
        panel["delta3_household_debt_gdp"].notna()
        & panel["delta3_log_real_house_price"].notna()
    )
    return int(panel.loc[business_pair | household_pair, "year"].max())


def _base_grid():
    """Create the full country-year grid (START_YEAR..END_YEAR) with names."""
    index = pd.MultiIndex.from_product(
        [GHSS_COUNTRIES, range(START_YEAR, END_YEAR + 1)],
        names=["country_iso3", "year"],
    )
    panel = index.to_frame(index=False)
    panel["country"] = panel["country_iso3"].map(GHSS_COUNTRIES)
    return panel


def _merge_one_to_one(panel, other):
    """Left-join ``other`` on (country_iso3, year), enforcing unique keys.

    :param panel: accumulating country-year analysis panel (the left side).
    :param other: tidy country-year panel whose columns are joined on.
    :returns: the joined panel.
    """
    return panel.merge(
        other, on=["country_iso3", "year"], how="left", validate="one_to_one"
    )


def _add_sector_features(panel, sector, debt_column, price_column):
    """Add one sector's quantile buckets, high-growth flags, and R-zone flag.

    Cutoffs come only from in_paper_sample rows where both inputs are observed
    and are stored as frozen ``*_cutoff`` columns.

    :param panel: analysis panel carrying the growth columns and the
        in_paper_sample flag; returned with the new columns added.
    :param sector: column prefix, "business" or "household".
    :param debt_column: the sector's 3-year debt-growth column.
    :param price_column: the sector's 3-year price-growth column.
    :returns: the panel with the sector's new columns added.
    """
    pair = panel[debt_column].notna() & panel[price_column].notna()
    reference = panel["in_paper_sample"] & pair
    debt_cutoffs = quantile_cutoffs(
        panel.loc[reference, debt_column], [0.2, 0.4, 0.6, 0.8]
    )
    price_cutoffs = quantile_cutoffs(panel.loc[reference, price_column], [1 / 3, 2 / 3])

    panel[f"{sector}_debt_quintile"] = assign_bucket(panel[debt_column], debt_cutoffs)
    panel[f"{sector}_price_tercile"] = assign_bucket(panel[price_column], price_cutoffs)
    panel[f"high_{sector}_debt_growth"] = indicator_above(
        panel[debt_column], debt_cutoffs[0.8]
    )
    panel[f"high_{sector}_price_growth"] = indicator_above(
        panel[price_column], price_cutoffs[2 / 3]
    )
    panel[f"rzone_{sector}"] = (
        panel[f"high_{sector}_debt_growth"] * panel[f"high_{sector}_price_growth"]
    ).astype("Int64")
    panel[f"{sector}_debt_q80_cutoff"] = debt_cutoffs[0.8]
    panel[f"{sector}_price_t667_cutoff"] = price_cutoffs[2 / 3]
    return panel


def _add_fragility(panel, jst):
    """Merge JST fragility variables and add high-fragility flags and R-zone
    interactions.

    Adds noncore, ltd, and lev with high_* flags at frozen in-paper-sample
    cutoffs, each interacted with both sectors' R-zone indicators. Noncore and
    ltd flag values above their Q80; lev is a capital ratio, so its fragility
    flag marks THIN capital, values below the Q20 (the extension proposal's
    flipped inequality).

    :param panel: analysis panel carrying in_paper_sample and the
        rzone_business / rzone_household indicators.
    :param jst: JST macrohistory table supplying noncore, ltd, and lev.
    :returns: the panel with the fragility columns added.
    """
    fragility = jst[jst["iso"].isin(GHSS_COUNTRIES)][
        ["iso", "year", "noncore", "ltd", "lev"]
    ].rename(columns={"iso": "country_iso3"})
    fragility["year"] = fragility["year"].astype(int)
    panel = _merge_one_to_one(panel, fragility)
    for variable in ["noncore", "ltd", "lev"]:
        reference = panel["in_paper_sample"] & panel[variable].notna()
        if variable == "lev":
            cutoff = quantile_cutoffs(panel.loc[reference, variable], [0.2])[0.2]
            panel[f"high_{variable}"] = indicator_below(panel[variable], cutoff)
            panel[f"{variable}_q20_cutoff"] = cutoff
        else:
            cutoff = quantile_cutoffs(panel.loc[reference, variable], [0.8])[0.8]
            panel[f"high_{variable}"] = indicator_above(panel[variable], cutoff)
            panel[f"{variable}_q80_cutoff"] = cutoff
        for sector in ["business", "household"]:
            panel[f"rzone_{sector}_x_high_{variable}"] = (
                panel[f"rzone_{sector}"] * panel[f"high_{variable}"]
            ).astype("Int64")
    return panel


def build_analysis_panel(crisis, credit, equity, house, jst):
    """Assemble the unified analysis panel from the four tidy panels plus JST.

    :param crisis: tidy crisis panel from the clean_crisis_chronologies
        module.
    :param credit: tidy credit panel from the clean_credit_panel module.
    :param equity: tidy equity panel from the clean_equity_panel module.
    :param house: tidy house-price panel from the clean_house_price_panel
        module.
    :param jst: JST macrohistory table supplying the fragility variables.
    :returns: one row per (country, year) with forward outcomes, sample
        flags, quantile features, R-zone indicators, and fragility
        interactions.
    :raises ValueError: if the panel contains duplicate country-years.
    """
    panel = _base_grid()
    for source_panel in [crisis, credit, equity, house]:
        panel = _merge_one_to_one(panel, source_panel)

    panel = add_future_event_windows(panel)
    business_pair = (
        panel["delta3_business_debt_gdp"].notna()
        & panel["delta3_log_real_equity"].notna()
    )
    household_pair = (
        panel["delta3_household_debt_gdp"].notna()
        & panel["delta3_log_real_house_price"].notna()
    )
    common_window = panel["year"].between(
        PAPER_SAMPLE_START_YEAR, LAST_COMMON_FORECAST_YEAR
    )
    panel["in_paper_sample"] = (
        common_window
        & panel["crisis_next_4y"].notna()
        & (business_pair | household_pair)
    )
    panel["in_business_sample"] = panel["in_paper_sample"] & business_pair
    panel["in_household_sample"] = panel["in_paper_sample"] & household_pair
    panel["is_historical_replication"] = panel["year"].between(
        PAPER_SAMPLE_START_YEAR, PAPER_SAMPLE_END_YEAR
    )
    panel["is_post_2016"] = panel["year"].gt(PAPER_SAMPLE_END_YEAR)

    panel = _add_sector_features(
        panel,
        "business",
        "delta3_business_debt_gdp",
        "delta3_log_real_equity",
    )
    panel = _add_sector_features(
        panel,
        "household",
        "delta3_household_debt_gdp",
        "delta3_log_real_house_price",
    )
    panel["rzone_either"] = panel[["rzone_business", "rzone_household"]].max(
        axis=1, skipna=True
    )
    panel["rzone_both"] = panel["rzone_business"] * panel["rzone_household"]
    panel = _add_fragility(panel, jst)

    if panel.duplicated(["country_iso3", "year"]).any():
        raise ValueError("analysis panel contains duplicate country-years")
    return panel.sort_values(["country_iso3", "year"]).reset_index(drop=True)


def analysis_panel_path(data_dir=DATA_DIR):
    """Path of the combined analysis panel parquet.

    :param data_dir: directory holding the project's parquet files (defaults
        to the configured DATA_DIR).
    :returns: the parquet file path under ``data_dir``.
    """
    return Path(data_dir) / ANALYSIS_PANEL_FILENAME


def load_analysis_panel(data_dir=DATA_DIR):
    """Load the combined R-zone analysis panel: one row per (country, year).

    :param data_dir: directory holding the project's parquet files (defaults
        to the configured DATA_DIR).
    :returns: the loaded panel, one row per (country, year).
    """
    return pd.read_parquet(analysis_panel_path(data_dir))


if __name__ == "__main__":
    panel = build_analysis_panel(
        load_crisis_panel(DATA_DIR),
        load_credit_panel(DATA_DIR),
        load_equity_panel(DATA_DIR),
        load_house_price_panel(DATA_DIR),
        load_jst_macrohistory(DATA_DIR),
    )
    panel.to_parquet(analysis_panel_path(DATA_DIR), index=False)
    print(
        "Saved rzone_analysis_panel.parquet: "
        f"{panel.shape}; common sample={panel.in_paper_sample.sum()}, "
        f"business={panel.in_business_sample.sum()}, "
        f"household={panel.in_household_sample.sum()}"
    )
