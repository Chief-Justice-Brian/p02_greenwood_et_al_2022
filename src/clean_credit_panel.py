"""Construct spliced three-year credit-growth measures.

Three-year changes are calculated inside each provider before splicing, as
required by GHSS footnote 6. Sources in splice priority: IMF GDD, then BIS
(which also carries the extension forward), then JST supplying early history
for the subset of sample countries it covers.

Four measures are produced: the business, household, and total private
debt-to-GDP three-year changes (percentage points), and the three-year log
change in real total private debt (x100). The real-debt measure deflates
each provider's debt level with the macro CPI panel, except JST, whose own
CPI is used so its native scale stays consistent. Output:
``credit_panel.parquet``, one row per (country, year), with a source label
beside every measure.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from clean_macro_deflators import load_macro_deflators
from country_sample import BIS_ISO2_TO_ISO3, GHSS_COUNTRIES
from panel_utils import (
    annual_average_from_quarterly,
    delta3_within_source,
    log_positive,
    splice_by_priority,
)
from pull_bis_total_credit import load_bis_total_credit
from pull_imf_gdd import load_imf_gdd
from pull_jst_macrohistory import load_jst_macrohistory
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
CREDIT_PANEL_FILENAME = "credit_panel.parquet"


def _gdd_levels(gdd, indicator):
    """Extract one IMF GDD indicator as country_iso3 | year | value levels.

    :param gdd: long IMF Global Debt Database pull, one row per
        country-indicator-year.
    :param indicator: GDD indicator code to extract (e.g. "NFC_LS").
    :returns: country_iso3 | year | value levels for the indicator.
    """
    return (
        gdd[gdd["indicator"].eq(indicator)][["country_iso3", "year", "value"]]
        .drop_duplicates(["country_iso3", "year"])
        .reset_index(drop=True)
    )


def _bis_levels(bis, borrower):
    """Annualize BIS credit-to-GDP levels for one borrower sector.

    Keeps all-lender, market-value, break-adjusted, percent-of-GDP series.

    :param bis: quarterly BIS total credit pull, one row per series-quarter
        with SDMX dimension columns.
    :param borrower: BIS tc_borrowers code ("N" non-financial corporations,
        "H" households, "P" private non-financial sector).
    :returns: annual country_iso3 | year | value levels for the sector.
    """
    # Dimension codes are documented in pull_bis_total_credit's docstring.
    selected = bis[
        bis["tc_borrowers"].eq(borrower)  # the requested sector
        & bis["tc_lenders"].eq("A")  # credit from all lending sectors
        & bis["valuation"].eq("M")  # market value
        & bis["unit_type"].eq("770")  # percent of GDP
        & bis["tc_adjust"].eq("A")  # adjusted for breaks
    ].copy()
    selected["country_iso3"] = selected["borrowers_country"].map(BIS_ISO2_TO_ISO3)
    selected = selected.dropna(subset=["country_iso3"])
    return annual_average_from_quarterly(selected[["country_iso3", "period", "value"]])


def _jst_ratio_levels(jst, debt_column):
    """Compute one JST debt column as a percent-of-GDP level series.

    :param jst: JST macrohistory table with iso, year, gdp, and the nominal
        debt level columns.
    :param debt_column: JST nominal debt column to convert (e.g. "tbus").
    :returns: country_iso3 | year | value percent-of-GDP levels.
    """
    selected = jst[jst["iso"].isin(GHSS_COUNTRIES)][
        ["iso", "year", debt_column, "gdp"]
    ].copy()
    selected["value"] = 100.0 * selected[debt_column] / selected["gdp"]
    selected.loc[~np.isfinite(selected["value"]), "value"] = np.nan
    return selected.rename(columns={"iso": "country_iso3"})[
        ["country_iso3", "year", "value"]
    ]


def _three_year_change(levels):
    """Take 3-year simple differences of the value column (percentage points).

    :param levels: country_iso3 | year | value ratio levels from one provider.
    :returns: country_iso3 | year | delta3 changes in percentage points.
    """
    return delta3_within_source(levels, "value", is_log=False)


def _splice_ratio(gdd, bis, jst, gdd_indicator, bis_borrower, jst_column):
    """Splice one sector's 3-year debt-to-GDP change: IMF GDD, then BIS, then JST.

    :param gdd: long, indicator-coded IMF Global Debt Database pull.
    :param bis: quarterly BIS total credit pull.
    :param jst: JST macrohistory table.
    :param gdd_indicator: the sector's series code in IMF GDD (e.g. "NFC_LS"
        for business).
    :param bis_borrower: the sector's series code in BIS (e.g. "N" for
        business).
    :param jst_column: the sector's series code in JST (e.g. "tbus" for
        business).
    :returns: spliced country_iso3 | year | delta3 | source rows.
    """
    return splice_by_priority(
        [
            ("IMF_GDD", _three_year_change(_gdd_levels(gdd, gdd_indicator))),
            ("BIS", _three_year_change(_bis_levels(bis, bis_borrower))),
            ("JST", _three_year_change(_jst_ratio_levels(jst, jst_column))),
        ]
    )


def _real_debt_change(levels, macro, debt_is_ratio):
    """Compute the 3-year log change (x100) of real debt from one source.

    :param levels: country_iso3 | year | value levels for one provider.
    :param macro: deflator panel supplying cpi and nominal_gdp.
    :param debt_is_ratio: if True, ``value`` is percent of GDP and is
        converted to a currency level via nominal GDP before deflating by
        CPI.
    :returns: country_iso3 | year | delta3 log changes (x100).
    """
    merged = levels.merge(
        macro[["country_iso3", "year", "cpi", "nominal_gdp"]],
        on=["country_iso3", "year"],
        how="left",
        validate="many_to_one",
    )
    if debt_is_ratio:
        real_debt = merged["value"] / 100.0 * merged["nominal_gdp"] / merged["cpi"]
    else:
        real_debt = merged["value"] / merged["cpi"]
    merged["log_real_debt"] = log_positive(real_debt)
    return delta3_within_source(merged, "log_real_debt", is_log=True)


def build_credit_panel(gdd, bis, jst, macro):
    """Assemble the spliced 3-year credit-growth panel across providers.

    :param gdd: raw IMF Global Debt Database pull (long, indicator-coded).
    :param bis: raw BIS total credit pull (quarterly).
    :param jst: JST macrohistory table.
    :param macro: CPI / nominal-GDP deflator panel, for real debt levels.
    :returns: one row per (country, year); every measure carries a source
        label.
    :raises ValueError: if the panel contains duplicate country-years.
    """
    sample_codes = set(GHSS_COUNTRIES)
    gdd = gdd[gdd["country_iso3"].isin(sample_codes)].copy()

    business = _splice_ratio(gdd, bis, jst, "NFC_LS", "N", "tbus").rename(
        columns={
            "delta3": "delta3_business_debt_gdp",
            "source": "business_debt_source",
        }
    )
    household = _splice_ratio(gdd, bis, jst, "HH_LS", "H", "thh").rename(
        columns={
            "delta3": "delta3_household_debt_gdp",
            "source": "household_debt_source",
        }
    )
    private_ratio = _splice_ratio(gdd, bis, jst, "PVD_LS", "P", "tloans").rename(
        columns={
            "delta3": "delta3_private_debt_gdp",
            "source": "private_debt_ratio_source",
        }
    )

    gdd_real = _real_debt_change(_gdd_levels(gdd, "PVD_LS"), macro, True)
    bis_real = _real_debt_change(_bis_levels(bis, "P"), macro, True)
    jst_total = (
        jst[jst["iso"].isin(sample_codes)][["iso", "year", "tloans", "cpi"]]
        .rename(columns={"iso": "country_iso3", "tloans": "value"})
        .copy()
    )
    # JST's loan and CPI levels share a consistent native scale.  Use JST CPI
    # rather than a chained external level to avoid currency/index-base jumps.
    jst_total["log_real_debt"] = log_positive(jst_total["value"] / jst_total["cpi"])
    jst_real = delta3_within_source(jst_total, "log_real_debt", is_log=True)
    private_real = splice_by_priority(
        [("IMF_GDD", gdd_real), ("BIS", bis_real), ("JST", jst_real)]
    ).rename(
        columns={
            "delta3": "delta3_log_real_private_debt",
            "source": "private_real_debt_source",
        }
    )

    result = business.merge(
        household, on=["country_iso3", "year"], how="outer", validate="one_to_one"
    )
    result = result.merge(
        private_ratio,
        on=["country_iso3", "year"],
        how="outer",
        validate="one_to_one",
    )
    result = result.merge(
        private_real,
        on=["country_iso3", "year"],
        how="outer",
        validate="one_to_one",
    )
    if result.duplicated(["country_iso3", "year"]).any():
        raise ValueError("credit panel contains duplicate country-years")
    return result.sort_values(["country_iso3", "year"]).reset_index(drop=True)


def load_credit_panel(data_dir=DATA_DIR):
    """Load the spliced credit-growth panel: one row per (country, year).

    :param data_dir: directory holding the project's parquet files (defaults
        to the configured DATA_DIR).
    :returns: the loaded panel, one row per (country, year).
    """
    return pd.read_parquet(Path(data_dir) / CREDIT_PANEL_FILENAME)


if __name__ == "__main__":
    panel = build_credit_panel(
        load_imf_gdd(DATA_DIR),
        load_bis_total_credit(DATA_DIR),
        load_jst_macrohistory(DATA_DIR),
        load_macro_deflators(DATA_DIR),
    )
    panel.to_parquet(DATA_DIR / CREDIT_PANEL_FILENAME, index=False)
    print(f"Saved credit_panel.parquet: {panel.shape}")
