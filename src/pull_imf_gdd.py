"""Pull household and business credit-to-GDP from the IMF Global Debt Database.

The IMF Global Debt Database (Mbaye, Moreno Badia & Chae, 2018) provides
private debt split into household and non-financial-corporate sectors as a
percent of GDP, annually, back to 1950 for many countries. It is one of the two
credit sources in Greenwood-Hanson-Shleifer-Sorensen (2022) (the other is the
BIS Total Credit Statistics; the paper splices 3-year CHANGES across sources).

The data is served by the free IMF DataMapper JSON API (no key required):
    https://www.imf.org/external/datamapper/api/v1/{indicator}

Indicators pulled:

``HH_LS``
    Household debt (loans and securities), percent of GDP.
``NFC_LS``
    Non-financial corporate debt (loans and securities), percent of GDP.
``PVD_LS``
    Total private debt (loans and securities), percent of GDP
    (= HH_LS + NFC_LS; kept as a cross-check on the two sector series).

Output: ``imf_gdd.parquet``, long format, one row per
country-indicator-year::

    country_iso3 | year | indicator | value

References
----------
- DataMapper page (view the data manually, country by country, with charts):
  https://www.imf.org/external/datamapper/datasets/GDD
- DataMapper API documentation:
  https://www.imf.org/external/datamapper/api/help
- Update frequency: annual vintage, typically refreshed once a year; the max
  year in the API response reveals the current vintage (2024 as of this
  writing).
- Citation: Mbaye, Moreno Badia and Chae, "Global Debt Database:
  Methodology and Sources," IMF Working Paper 18/111, 2018.
"""

from pathlib import Path

import pandas as pd
import requests

from settings import config

DATA_DIR = config("DATA_DIR")
IMF_GDD_FILENAME = "imf_gdd.parquet"

DATAMAPPER_API_BASE = "https://www.imf.org/external/datamapper/api/v1"

GDD_INDICATORS = {
    "HH_LS": "household debt, loans and debt securities, percent of GDP",
    "NFC_LS": "nonfinancial corporate debt, loans and debt securities, percent of GDP",
    "PVD_LS": "private debt, loans and debt securities, percent of GDP",
}


def pull_gdd_indicator(indicator_code):
    """Pull one GDD indicator from the DataMapper API into a DataFrame.

    The API returns nested JSON: {"values": {indicator: {iso3: {year: value}}}}.
    """
    url = f"{DATAMAPPER_API_BASE}/{indicator_code}"
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    payload = response.json()
    values_by_country = payload["values"][indicator_code]

    rows = []
    for country_iso3, values_by_year in values_by_country.items():
        for year_string, value in values_by_year.items():
            rows.append(
                {
                    "country_iso3": country_iso3,
                    "year": int(year_string),
                    "indicator": indicator_code,
                    "value": float(value),
                }
            )

    indicator_df = pd.DataFrame(rows)
    return indicator_df


def pull_imf_gdd(indicator_codes=tuple(GDD_INDICATORS)):
    """Pull all GDD indicators and stack them into one long DataFrame."""
    indicator_frames = []
    for indicator_code in indicator_codes:
        indicator_df = pull_gdd_indicator(indicator_code)
        indicator_frames.append(indicator_df)

    gdd_df = pd.concat(indicator_frames, ignore_index=True)
    return gdd_df


def load_imf_gdd(data_dir=DATA_DIR):
    """Load the GDD long panel: country_iso3 | year | indicator | value."""
    return pd.read_parquet(Path(data_dir) / IMF_GDD_FILENAME)


if __name__ == "__main__":
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    gdd_df = pull_imf_gdd()
    gdd_df.to_parquet(data_dir / IMF_GDD_FILENAME)

    coverage = gdd_df.groupby("indicator")["country_iso3"].nunique()
    print(f"Saved imf_gdd.parquet: {gdd_df.shape}")
    print(f"Countries per indicator:\n{coverage}")
