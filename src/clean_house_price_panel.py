"""Construct spliced three-year real house-price growth.

Sources in splice priority: BIS real residential property prices, then OECD
analytical house prices, then JST. Output: ``house_price_panel.parquet``,
one row per (country, year).
"""

from pathlib import Path

import pandas as pd

from country_sample import BIS_ISO2_TO_ISO3, GHSS_COUNTRIES
from panel_utils import (
    annual_average_from_quarterly,
    delta3_within_source,
    log_positive,
    splice_by_priority,
)
from pull_bis_property_prices import load_bis_property_prices
from pull_jst_macrohistory import load_jst_macrohistory
from pull_oecd_house_prices import load_oecd_house_prices
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
HOUSE_PRICE_PANEL_FILENAME = "house_price_panel.parquet"


def _bis_real_levels(bis):
    """Annualize the BIS real (CPI-deflated) residential property price index.

    :param bis: quarterly BIS residential property price pull with ref_area,
        value_type, unit_measure, period, and value columns.
    :returns: annual country_iso3 | year | value levels.
    """
    # "R" = real (CPI-deflated), "628" = index (2010 = 100); the codes are
    # documented in pull_bis_property_prices' docstring.
    selected = bis[bis["value_type"].eq("R") & bis["unit_measure"].eq("628")].copy()
    selected["country_iso3"] = selected["ref_area"].map(BIS_ISO2_TO_ISO3)
    selected = selected.dropna(subset=["country_iso3"])
    return annual_average_from_quarterly(selected[["country_iso3", "period", "value"]])


def _oecd_real_levels(oecd):
    """Build annual OECD real house-price levels, preferring published annual rows.

    Quarterly averages fill only country-years with no annual observation.

    :param oecd: OECD analytical house-price pull mixing annual and quarterly
        rows, with country_iso3, measure, freq, period, and value columns.
    :returns: annual country_iso3 | year | value levels.
    """
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
    """Compute JST real house-price levels as hpnom deflated by JST CPI.

    :param jst: JST macrohistory table carrying iso, year, hpnom, and cpi.
    :returns: country_iso3 | year | value levels.
    """
    selected = jst[jst["iso"].isin(GHSS_COUNTRIES)][
        ["iso", "year", "hpnom", "cpi"]
    ].copy()
    selected["value"] = selected["hpnom"] / selected["cpi"]
    return selected.rename(columns={"iso": "country_iso3"})[
        ["country_iso3", "year", "value"]
    ]


def _log_growth(levels):
    """Take 3-year log changes (x100) of the real house-price levels.

    :param levels: country_iso3 | year | value real house-price levels from
        one source.
    :returns: country_iso3 | year | delta3 log changes (x100).
    """
    levels = levels.copy()
    levels["log_real_house_price"] = log_positive(levels["value"])
    return delta3_within_source(levels, "log_real_house_price", is_log=True)


def build_house_price_panel(bis, oecd, jst):
    """Build the spliced 3-year real house-price-growth panel (BIS > OECD > JST).

    :param bis: quarterly BIS residential property price pull.
    :param oecd: OECD analytical house-price pull (annual plus quarterly
        rows).
    :param jst: JST macrohistory table (hpnom and cpi).
    :returns: one row per (country, year) with delta3_log_real_house_price
        and its house_price_source label.
    :raises ValueError: if the panel contains duplicate country-years.
    """
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


def load_house_price_panel(data_dir=DATA_DIR):
    """Load the spliced real house-price-growth panel: one row per (country, year).

    :param data_dir: directory holding the project's parquet files (defaults
        to the configured DATA_DIR).
    :returns: the loaded panel, one row per (country, year).
    """
    return pd.read_parquet(Path(data_dir) / HOUSE_PRICE_PANEL_FILENAME)


if __name__ == "__main__":
    panel = build_house_price_panel(
        load_bis_property_prices(DATA_DIR),
        load_oecd_house_prices(DATA_DIR),
        load_jst_macrohistory(DATA_DIR),
    )
    panel.to_parquet(DATA_DIR / HOUSE_PRICE_PANEL_FILENAME, index=False)
    print(f"Saved house_price_panel.parquet: {panel.shape}")
