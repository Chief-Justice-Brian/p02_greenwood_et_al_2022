"""Pull historical share price indices from the IMF (former IFS series).

Greenwood-Hanson-Shleifer-Sorensen (2022) use IMF International Financial
Statistics share price indices as a supplementary equity source (their primary
source, Global Financial Data, is paywalled and off-limits for this course; we
mandate IFS + JST instead).

The IMF retired the old IFS API and moved these series into the dataflow
``IMF.STA/MFS_FMP`` ("Monetary and Financial Statistics: Financial Market
Prices") on the new SDMX API at api.imf.org. Coverage is annual and spans most
of the paper's sample, but ends in the mid-2010s: the IMF stopped collecting
share prices around 2017, so OECD share prices carry the panel forward from
there.

We pull BOTH transformations for all annual series and let the cleaning stage
choose one consistent (indicator, transformation) pair per country.

indicator
    One code per index, so a country can carry several (e.g. USA has
    ``EQTS`` total share prices, ``EPSP`` S&P 500, ``NASDAQ``).
transformation
    ``PA_IX`` period-average index, ``EOP_IX`` end-of-period index. A large
    minority of countries (e.g. Germany, Spain, Sweden, Peru) publish ONLY
    end-of-period, so neither one alone covers the panel.

Output: ``imf_equity.parquet``, long format, one row per
country-indicator-transformation-year::

    country_iso3 | indicator | transformation | year | value

References
----------
- Dataset page on the IMF data portal (view the series manually):
  https://data.imf.org/en/datasets/IMF.STA:MFS_FMP
- IMF SDMX API documentation:
  https://data.imf.org/en/Resource-Pages/IMF-API
- Update frequency: effectively frozen -- the IMF discontinued share price
  collection around 2017, so most countries end in 2016/17 (a handful
  report later). OECD share prices carry the panel forward from there
  (see pull_oecd_share_prices.py).
- Citation: IMF, "Monetary and Financial Statistics (MFS): Financial
  Market Prices" (successor to the International Financial Statistics
  share price indices), data.imf.org.
"""

import io
from pathlib import Path

import pandas as pd
import requests

from settings import config

DATA_DIR = config("DATA_DIR")

IMF_MFS_FMP_URL = (
    "https://api.imf.org/external/sdmx/2.1/data/IMF.STA,MFS_FMP/"
    "..PA_IX+EOP_IX.A?startPeriod=1948"
)

SDMX_CSV_ACCEPT_HEADER = {"Accept": "application/vnd.sdmx.data+csv"}


def pull_imf_equity(url=IMF_MFS_FMP_URL):
    """Download all annual period-average share price indices from the IMF."""
    response = requests.get(url, headers=SDMX_CSV_ACCEPT_HEADER, timeout=300)
    response.raise_for_status()

    raw_df = pd.read_csv(io.BytesIO(response.content))

    # The SDMX CSV carries many metadata columns; keep the identifying ones
    equity_df = raw_df[
        ["COUNTRY", "INDICATOR", "TYPE_OF_TRANSFORMATION", "TIME_PERIOD", "OBS_VALUE"]
    ].rename(
        columns={
            "COUNTRY": "country_iso3",
            "INDICATOR": "indicator",
            "TYPE_OF_TRANSFORMATION": "transformation",
            "TIME_PERIOD": "year",
            "OBS_VALUE": "value",
        }
    )
    equity_df["year"] = equity_df["year"].astype(int)
    equity_df["value"] = pd.to_numeric(equity_df["value"], errors="coerce")
    equity_df = equity_df.dropna(subset=["value"]).reset_index(drop=True)

    return equity_df


def load_imf_equity(data_dir=DATA_DIR):
    """Load the IMF share price long panel: country_iso3 | indicator | year | value."""
    return pd.read_parquet(Path(data_dir) / "imf_equity.parquet")


if __name__ == "__main__":
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    equity_df = pull_imf_equity()
    equity_df.to_parquet(data_dir / "imf_equity.parquet")

    n_countries = equity_df["country_iso3"].nunique()
    year_min, year_max = equity_df["year"].min(), equity_df["year"].max()
    print(
        f"Saved imf_equity.parquet: {equity_df.shape}, "
        f"{n_countries} countries, {year_min}-{year_max}"
    )
