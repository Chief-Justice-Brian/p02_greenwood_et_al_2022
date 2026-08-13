"""Build continuous annual CPI and nominal-GDP panels for the GHSS countries.

World Bank WDI is the primary source.  JST fills early observations and gaps
for the 18 countries it covers.  Because the two providers use different
index bases and sometimes different currency-unit scales, JST levels are
rebased country by country using the median ratio during overlap years before
they are used as a supplement.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from country_sample import GHSS_COUNTRIES
from settings import config

DATA_DIR = Path(config("DATA_DIR"))


def _rebase_supplement(primary, supplement):
    """Put a supplementary level series on the primary series' scale."""
    overlap = primary.notna() & supplement.notna() & (supplement != 0)
    if not overlap.any():
        return supplement
    ratios = primary.loc[overlap] / supplement.loc[overlap]
    ratios = ratios.replace([np.inf, -np.inf], np.nan).dropna()
    if ratios.empty:
        return supplement
    return supplement * ratios.median()


def build_macro_panel(wdi, jst):
    sample_codes = set(GHSS_COUNTRIES)

    wdi = wdi[wdi["country_iso3"].isin(sample_codes)].copy()
    wdi_wide = wdi.pivot_table(
        index=["country_iso3", "year"],
        columns="indicator",
        values="value",
        aggfunc="first",
    ).rename(
        columns={
            "FP.CPI.TOTL": "wdi_cpi",
            "NY.GDP.MKTP.CN": "wdi_nominal_gdp",
        }
    )

    jst = jst[jst["iso"].isin(sample_codes)].copy()
    jst["year"] = jst["year"].astype(int)
    jst_wide = jst.rename(
        columns={
            "iso": "country_iso3",
            "cpi": "jst_cpi",
            "gdp": "jst_nominal_gdp",
        }
    ).set_index(["country_iso3", "year"])[["jst_cpi", "jst_nominal_gdp"]]

    panel = wdi_wide.join(jst_wide, how="outer").sort_index()
    pieces = []
    for country_iso3, country in panel.groupby(level="country_iso3"):
        country = country.droplevel("country_iso3").sort_index().copy()
        country["jst_cpi_rebased"] = _rebase_supplement(
            country.get("wdi_cpi", pd.Series(index=country.index, dtype=float)),
            country.get("jst_cpi", pd.Series(index=country.index, dtype=float)),
        )
        country["jst_gdp_rebased"] = _rebase_supplement(
            country.get("wdi_nominal_gdp", pd.Series(index=country.index, dtype=float)),
            country.get("jst_nominal_gdp", pd.Series(index=country.index, dtype=float)),
        )
        country["cpi"] = country.get("wdi_cpi").combine_first(
            country["jst_cpi_rebased"]
        )
        country["nominal_gdp"] = country.get("wdi_nominal_gdp").combine_first(
            country["jst_gdp_rebased"]
        )
        country["cpi_source"] = np.where(
            country.get("wdi_cpi").notna(),
            "WDI",
            np.where(country["jst_cpi_rebased"].notna(), "JST", pd.NA),
        )
        country["nominal_gdp_source"] = np.where(
            country.get("wdi_nominal_gdp").notna(),
            "WDI",
            np.where(country["jst_gdp_rebased"].notna(), "JST", pd.NA),
        )
        country["country_iso3"] = country_iso3
        pieces.append(country.reset_index())

    result = pd.concat(pieces, ignore_index=True)
    result = result[
        [
            "country_iso3",
            "year",
            "cpi",
            "cpi_source",
            "nominal_gdp",
            "nominal_gdp_source",
        ]
    ]
    result = result.dropna(subset=["cpi", "nominal_gdp"], how="all")
    if result.duplicated(["country_iso3", "year"]).any():
        raise ValueError("macro panel contains duplicate country-years")
    if (result["cpi"].dropna() <= 0).any():
        raise ValueError("CPI must be positive")
    return result.sort_values(["country_iso3", "year"]).reset_index(drop=True)


if __name__ == "__main__":
    panel = build_macro_panel(
        pd.read_parquet(DATA_DIR / "worldbank_wdi.parquet"),
        pd.read_parquet(DATA_DIR / "jst_macrohistory.parquet"),
    )
    panel.to_parquet(DATA_DIR / "macro_deflators.parquet", index=False)
    print(
        "Saved macro_deflators.parquet: "
        f"{panel.shape}, {panel.country_iso3.nunique()} countries, "
        f"{panel.year.min()}-{panel.year.max()}"
    )
