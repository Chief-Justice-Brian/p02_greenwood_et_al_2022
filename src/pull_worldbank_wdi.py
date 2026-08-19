"""Pull CPI and nominal GDP from the World Bank World Development Indicators.

Greenwood-Hanson-Shleifer-Sorensen (2022) deflate asset price indices with CPI
and scale credit by nominal GDP. World Bank WDI provides both, annually, for
all countries from 1960; JST fills 1950-1959 for the advanced economies.

Indicators pulled (World Bank API v2, free, no key):

``FP.CPI.TOTL``
    Consumer price index (2010 = 100). Deflates the asset price indices.

``NY.GDP.MKTP.CN``
    GDP, current local currency units. Scales the credit aggregates.

Output: ``worldbank_wdi.parquet``, long format, one row per
country-indicator-year::

    country_iso3 | year | indicator | value

The API returns aggregates (regions, income groups) alongside countries; the
cleaning stage filters to our panel's ISO3 codes.

References
----------
- Indicator pages (view/chart the data manually):
  https://data.worldbank.org/indicator/FP.CPI.TOTL
  https://data.worldbank.org/indicator/NY.GDP.MKTP.CN
- Indicators API documentation:
  https://datahelpdesk.worldbank.org/knowledgebase/articles/889392
- Update frequency: WDI is revised several times a year; the API's JSON
  response header carries a ``lastupdated`` date.
- Citation: World Bank, World Development Indicators (CC BY 4.0).
"""

from pathlib import Path

import pandas as pd
import requests

from settings import config

DATA_DIR = config("DATA_DIR")
WORLDBANK_WDI_FILENAME = "worldbank_wdi.parquet"

WORLDBANK_API_BASE = "https://api.worldbank.org/v2/country/all/indicator"

WDI_INDICATORS = {
    "FP.CPI.TOTL": "consumer price index (2010=100)",
    "NY.GDP.MKTP.CN": "GDP, current local currency",
}

# 20,000 rows per page keeps each indicator to a few API requests.
ROWS_PER_PAGE = 20000


def pull_wdi_indicator(
    indicator_code,
    start_year=config("START_DATE").year,
    end_year=config("END_DATE").year,
):
    """Pull one WDI indicator into long format.

    :param indicator_code: WDI indicator id inserted in the API path, one of
        the WDI_INDICATORS keys (e.g. ``FP.CPI.TOTL``).
    :param start_year: first year of the requested date window (inclusive);
        defaults to the configured panel window.
    :param end_year: last year of the requested date window (inclusive);
        defaults to the configured panel window.
    :returns: long DataFrame with one row per country-year for the indicator.
    """
    rows = []
    page = 1

    while True:
        url = (
            f"{WORLDBANK_API_BASE}/{indicator_code}"
            f"?format=json&per_page={ROWS_PER_PAGE}&page={page}"
            f"&date={start_year}:{end_year}"
        )

        response = requests.get(url, timeout=120)
        response.raise_for_status()
        metadata, records = response.json()

        for record in records:
            if record["value"] is None:
                continue

            rows.append(
                {
                    "country_iso3": record["countryiso3code"],
                    "year": int(record["date"]),
                    "indicator": indicator_code,
                    "value": float(record["value"]),
                }
            )

        if page >= metadata["pages"]:
            break

        page += 1

    return pd.DataFrame(rows)


def pull_worldbank_wdi(indicator_codes=tuple(WDI_INDICATORS)):
    """Pull all WDI indicators and stack them into one long DataFrame.

    :param indicator_codes: iterable of WDI indicator ids to pull (defaults
        to all the WDI_INDICATORS keys).
    :returns: one long DataFrame with all indicators stacked.
    """
    indicator_frames = []

    for indicator_code in indicator_codes:
        indicator_df = pull_wdi_indicator(indicator_code)
        indicator_frames.append(indicator_df)

    return pd.concat(indicator_frames, ignore_index=True)


def load_worldbank_wdi(data_dir=DATA_DIR):
    """Load the WDI long panel: one row per country-indicator-year.

    :param data_dir: directory holding the project's parquet files (defaults
        to the configured DATA_DIR).
    """
    return pd.read_parquet(Path(data_dir) / WORLDBANK_WDI_FILENAME)


if __name__ == "__main__":
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    wdi_df = pull_worldbank_wdi()
    wdi_df.to_parquet(data_dir / WORLDBANK_WDI_FILENAME)

    coverage = wdi_df.groupby("indicator")["country_iso3"].nunique()

    print(f"Saved worldbank_wdi.parquet: {wdi_df.shape}")
    print(f"Economies per indicator:\n{coverage}")
