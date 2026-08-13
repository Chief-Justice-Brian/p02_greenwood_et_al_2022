"""Standardize the crisis chronologies and related BVX outcome variables."""

from pathlib import Path

import numpy as np
import pandas as pd

from country_sample import GHSS_COUNTRIES
from settings import config

DATA_DIR = Path(config("DATA_DIR"))


def build_crisis_panel(bvx, jst):
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


if __name__ == "__main__":
    panel = build_crisis_panel(
        pd.read_parquet(DATA_DIR / "bvx_annual_regdata.parquet"),
        pd.read_parquet(DATA_DIR / "jst_macrohistory.parquet"),
    )
    panel.to_parquet(DATA_DIR / "crisis_panel.parquet", index=False)
    print(f"Saved crisis_panel.parquet: {panel.shape}")
