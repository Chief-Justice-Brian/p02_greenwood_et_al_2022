"""Standardize the crisis chronologies into one country-year panel.

BVX's revised chronology (RC) is the baseline crisis outcome; JST's
crisisJST and the Reinhart-Rogoff indicator carried in the BVX kit are kept
for Table 1's chronology comparison, alongside BVX's bank equity crash,
bank failure, and panic flags and real GDP growth. Two support conventions:
bank equity crashes are recorded only in event years, so the indicator is
zero-filled elsewhere; the RR chronology is defined only for RR-covered
countries and only through 2010, and stays missing outside that support.
Output: ``crisis_panel.parquet``, one row per (country, year).
"""

from pathlib import Path

import numpy as np
import pandas as pd

from country_sample import GHSS_COUNTRIES
from pull_bvx_crises import load_bvx_annual_regdata
from pull_jst_macrohistory import load_jst_macrohistory
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
CRISIS_PANEL_FILENAME = "crisis_panel.parquet"


def build_crisis_panel(bvx, jst):
    """Standardize the BVX and JST chronologies into one crisis panel.

    :param bvx: BVX annual regression file (ISO3 | year | RC | RR and
        bank-stress columns).
    :param jst: JST macrohistory table carrying crisisJST.
    :returns: one row per (country, year); indicators are validated 0/1 and
        real GDP growth is rescaled to percent.
    :raises ValueError: if a crisis indicator is not binary or the panel
        contains duplicate country-years.
    """
    sample_codes = set(GHSS_COUNTRIES)
    bvx = bvx[bvx["ISO3"].isin(sample_codes)].copy()
    bvx["year"] = bvx["year"].astype(int)

    panel = bvx[
        [
            "ISO3",
            "year",
            "RC",
            "ReinhartRogoffCrisis",
            "bankeqdecline",
            "bankfailure_narrative",
            "PANIC_ind",
            "rgdp_gr",
        ]
    ].rename(
        columns={
            "ISO3": "country_iso3",
            "RC": "crisis_bvx",
            "bankeqdecline": "bank_equity_crash",
            "bankfailure_narrative": "bank_failures",
            "PANIC_ind": "banking_panic",
        }
    )

    # In the BVX annual file, bank-equity crashes are recorded only in event
    # years.  The country-year panel itself defines the zero-risk support.
    panel["bank_equity_crash"] = panel["bank_equity_crash"].fillna(0.0)
    panel["real_gdp_growth"] = 100.0 * panel.pop("rgdp_gr")

    jst_crisis = (
        jst[jst["iso"].isin(sample_codes)]
        .rename(columns={"iso": "country_iso3", "crisisJST": "crisis_jst"})[
            ["country_iso3", "year", "crisis_jst"]
        ]
        .copy()
    )
    jst_crisis["year"] = jst_crisis["year"].astype(int)
    panel = panel.merge(
        jst_crisis, on=["country_iso3", "year"], how="left", validate="one_to_one"
    )

    # RR event years are stored sparsely in the BVX kit.  Treat observations
    # as defined for countries represented by RR, through the chronology's
    # final year (2010), and leave other rows missing.
    rr_countries = set(bvx.loc[bvx["ReinhartRogoffCrisis"].notna(), "ISO3"].unique())
    rr_defined = panel["country_iso3"].isin(rr_countries) & panel["year"].le(2010)
    panel["crisis_rr"] = np.where(
        rr_defined, panel["ReinhartRogoffCrisis"].fillna(0.0), np.nan
    )
    panel = panel.drop(columns="ReinhartRogoffCrisis")

    indicator_columns = [
        "crisis_bvx",
        "crisis_jst",
        "crisis_rr",
        "bank_equity_crash",
        "bank_failures",
        "banking_panic",
    ]
    for column in indicator_columns:
        observed = panel[column].dropna()
        if not observed.isin([0, 1]).all():
            raise ValueError(f"{column} is not binary")

    if panel.duplicated(["country_iso3", "year"]).any():
        raise ValueError("crisis panel contains duplicate country-years")
    return panel.sort_values(["country_iso3", "year"]).reset_index(drop=True)


def load_crisis_panel(data_dir=DATA_DIR):
    """Load the standardized crisis-chronology panel: one row per (country, year).

    :param data_dir: directory holding the project's parquet files (defaults
        to the configured DATA_DIR).
    :returns: the loaded panel, one row per (country, year).
    """
    return pd.read_parquet(Path(data_dir) / CRISIS_PANEL_FILENAME)


if __name__ == "__main__":
    panel = build_crisis_panel(
        load_bvx_annual_regdata(DATA_DIR),
        load_jst_macrohistory(DATA_DIR),
    )
    panel.to_parquet(DATA_DIR / CRISIS_PANEL_FILENAME, index=False)
    print(f"Saved crisis_panel.parquet: {panel.shape}")
