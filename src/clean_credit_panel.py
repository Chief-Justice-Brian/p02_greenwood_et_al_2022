"""Construct spliced three-year credit-growth measures.

Three-year changes are calculated inside each provider before splicing, as
required by GHSS footnote 5.  IMF GDD is primary, BIS fills gaps (and carries
the extension forward), and JST supplies early history for its 18 countries.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from country_sample import BIS_ISO2_TO_ISO3, GHSS_COUNTRIES
from panel_utils import (
    annual_average_from_quarterly,
    delta3_within_source,
    log_positive,
    splice_by_priority,
)
from settings import config

DATA_DIR = Path(config("DATA_DIR"))


def _gdd_levels(gdd, indicator):
    return (
        gdd[gdd["indicator"].eq(indicator)][["country_iso3", "year", "value"]]
        .drop_duplicates(["country_iso3", "year"])
        .reset_index(drop=True)
    )


def _bis_levels(bis, borrower):
    selected = bis[
        bis["tc_borrowers"].eq(borrower)
        & bis["tc_lenders"].eq("A")
        & bis["valuation"].eq("M")
        & bis["unit_type"].eq("770")
        & bis["tc_adjust"].eq("A")
    ].copy()
    selected["country_iso3"] = selected["borrowers_country"].map(BIS_ISO2_TO_ISO3)
    selected = selected.dropna(subset=["country_iso3"])
    return annual_average_from_quarterly(selected[["country_iso3", "period", "value"]])


def _jst_ratio_levels(jst, debt_column):
    selected = jst[jst["iso"].isin(GHSS_COUNTRIES)][
        ["iso", "year", debt_column, "gdp"]
    ].copy()
    selected["value"] = 100.0 * selected[debt_column] / selected["gdp"]
    selected.loc[~np.isfinite(selected["value"]), "value"] = np.nan
    return selected.rename(columns={"iso": "country_iso3"})[
        ["country_iso3", "year", "value"]
    ]


def _three_year_change(levels):
    return delta3_within_source(levels, "value", is_log=False)


def _splice_ratio(gdd, bis, jst, gdd_indicator, bis_borrower, jst_column):
    return splice_by_priority(
        [
            ("IMF_GDD", _three_year_change(_gdd_levels(gdd, gdd_indicator))),
            ("BIS", _three_year_change(_bis_levels(bis, bis_borrower))),
            ("JST", _three_year_change(_jst_ratio_levels(jst, jst_column))),
        ]
    )


def _real_debt_change(levels, macro, debt_is_ratio):
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


if __name__ == "__main__":
    panel = build_credit_panel(
        pd.read_parquet(DATA_DIR / "imf_gdd.parquet"),
        pd.read_parquet(DATA_DIR / "bis_total_credit.parquet"),
        pd.read_parquet(DATA_DIR / "jst_macrohistory.parquet"),
        pd.read_parquet(DATA_DIR / "macro_deflators.parquet"),
    )
    panel.to_parquet(DATA_DIR / "credit_panel.parquet", index=False)
    print(f"Saved credit_panel.parquet: {panel.shape}")
