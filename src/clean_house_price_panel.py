"""Construct spliced three-year real house-price growth."""

from pathlib import Path

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


def _bis_real_levels(bis):
    selected = bis[bis["value_type"].eq("R") & bis["unit_measure"].eq("628")].copy()
    selected["country_iso3"] = selected["ref_area"].map(BIS_ISO2_TO_ISO3)
    selected = selected.dropna(subset=["country_iso3"])
    return annual_average_from_quarterly(selected[["country_iso3", "period", "value"]])


def _oecd_real_levels(oecd):
    selected = oecd[
        oecd["country_iso3"].isin(GHSS_COUNTRIES) & oecd["measure"].eq("RHP")
    ].copy()
    annual = selected[selected["freq"].eq("A")][
        ["country_iso3", "period", "value"]
    ].copy()
    annual["year"] = annual["period"].astype(int)

    # Use quarterly averages only where the published annual observation is
    # absent.  This keeps each country-year unique and maximizes coverage.
    quarterly = annual_average_from_quarterly(
        selected[selected["freq"].eq("Q")][["country_iso3", "period", "value"]]
    )
    annual = annual[["country_iso3", "year", "value"]]
    keys = pd.MultiIndex.from_frame(annual[["country_iso3", "year"]])
    qkeys = pd.MultiIndex.from_frame(quarterly[["country_iso3", "year"]])
    return pd.concat([annual, quarterly[~qkeys.isin(keys)]], ignore_index=True)


def _jst_real_levels(jst):
    selected = jst[jst["iso"].isin(GHSS_COUNTRIES)][
        ["iso", "year", "hpnom", "cpi"]
    ].copy()
    selected["value"] = selected["hpnom"] / selected["cpi"]
    return selected.rename(columns={"iso": "country_iso3"})[
        ["country_iso3", "year", "value"]
    ]


def _log_growth(levels):
    levels = levels.copy()
    levels["log_real_house_price"] = log_positive(levels["value"])
    return delta3_within_source(levels, "log_real_house_price", is_log=True)


def build_house_price_panel(bis, oecd, jst):
    result = splice_by_priority(
        [
            ("BIS", _log_growth(_bis_real_levels(bis))),
            ("OECD", _log_growth(_oecd_real_levels(oecd))),
            ("JST", _log_growth(_jst_real_levels(jst))),
        ]
    ).rename(
        columns={
            "delta3": "delta3_log_real_house_price",
            "source": "house_price_source",
        }
    )
    if result.duplicated(["country_iso3", "year"]).any():
        raise ValueError("house-price panel contains duplicate country-years")
    return result


if __name__ == "__main__":
    panel = build_house_price_panel(
        pd.read_parquet(DATA_DIR / "bis_property_prices.parquet"),
        pd.read_parquet(DATA_DIR / "oecd_house_prices.parquet"),
        pd.read_parquet(DATA_DIR / "jst_macrohistory.parquet"),
    )
    panel.to_parquet(DATA_DIR / "house_price_panel.parquet", index=False)
    print(f"Saved house_price_panel.parquet: {panel.shape}")
