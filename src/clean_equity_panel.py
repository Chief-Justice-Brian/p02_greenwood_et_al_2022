"""Construct spliced three-year real equity-price growth."""

from pathlib import Path

import numpy as np
import pandas as pd

from clean_macro_deflators import load_macro_deflators
from country_sample import GHSS_COUNTRIES
from panel_utils import delta3_within_source, log_positive, splice_by_priority
from pull_imf_equity import load_imf_equity
from pull_jst_macrohistory import load_jst_macrohistory
from pull_oecd_share_prices import load_oecd_share_prices
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
EQUITY_PANEL_FILENAME = "equity_panel.parquet"


def select_imf_series(imf):
    """Select one stable (indicator, transformation) pair per country."""
    imf = imf[imf["country_iso3"].isin(GHSS_COUNTRIES)].copy()
    scores = (
        imf.groupby(["country_iso3", "indicator", "transformation"])
        .size()
        .rename("n")
        .reset_index()
    )
    scores["preferred_indicator"] = scores["indicator"].eq("EQTS").astype(int)
    scores["preferred_transformation"] = (
        scores["transformation"].eq("PA_IX").astype(int)
    )
    chosen = scores.sort_values(
        [
            "country_iso3",
            "n",
            "preferred_indicator",
            "preferred_transformation",
            "indicator",
        ],
        ascending=[True, False, False, False, True],
    ).drop_duplicates("country_iso3")[["country_iso3", "indicator", "transformation"]]
    return imf.merge(
        chosen,
        on=["country_iso3", "indicator", "transformation"],
        how="inner",
        validate="many_to_one",
    )


def _real_growth_from_nominal_index(levels, macro):
    merged = levels.merge(
        macro[["country_iso3", "year", "cpi"]],
        on=["country_iso3", "year"],
        how="left",
        validate="many_to_one",
    )
    merged["log_real_index"] = log_positive(merged["value"] / merged["cpi"])
    return delta3_within_source(merged, "log_real_index", is_log=True)


def _jst_real_growth(jst):
    frames = []
    for country_iso3, country in jst[jst["iso"].isin(GHSS_COUNTRIES)].groupby("iso"):
        country = country.set_index("year").sort_index()
        full_years = range(int(country.index.min()), int(country.index.max()) + 1)
        country = country.reindex(full_years)
        nominal_growth = np.log1p(country["eq_capgain"]).rolling(3, min_periods=3).sum()
        inflation = np.log(country["cpi"]) - np.log(country["cpi"].shift(3))
        delta = 100.0 * (nominal_growth - inflation)
        frame = delta.dropna().rename("delta3").reset_index()
        frame.columns = ["year", "delta3"]
        frame["country_iso3"] = country_iso3
        frames.append(frame)
    if not frames:
        return pd.DataFrame(columns=["country_iso3", "year", "delta3"])
    return pd.concat(frames, ignore_index=True)[["country_iso3", "year", "delta3"]]


def build_equity_panel(imf, jst, oecd, macro):
    imf_levels = select_imf_series(imf)[["country_iso3", "year", "value"]]
    oecd_levels = oecd[oecd["country_iso3"].isin(GHSS_COUNTRIES)][
        ["country_iso3", "year", "value"]
    ].copy()

    result = splice_by_priority(
        [
            ("IMF", _real_growth_from_nominal_index(imf_levels, macro)),
            ("JST", _jst_real_growth(jst)),
            ("OECD", _real_growth_from_nominal_index(oecd_levels, macro)),
        ]
    ).rename(
        columns={
            "delta3": "delta3_log_real_equity",
            "source": "equity_source",
        }
    )
    if result.duplicated(["country_iso3", "year"]).any():
        raise ValueError("equity panel contains duplicate country-years")
    return result


def load_equity_panel(data_dir=DATA_DIR):
    """Load the spliced real equity-growth panel: one row per (country, year)."""
    return pd.read_parquet(Path(data_dir) / EQUITY_PANEL_FILENAME)


if __name__ == "__main__":
    panel = build_equity_panel(
        load_imf_equity(DATA_DIR),
        load_jst_macrohistory(DATA_DIR),
        load_oecd_share_prices(DATA_DIR),
        load_macro_deflators(DATA_DIR),
    )
    panel.to_parquet(DATA_DIR / EQUITY_PANEL_FILENAME, index=False)
    print(f"Saved equity_panel.parquet: {panel.shape}")
