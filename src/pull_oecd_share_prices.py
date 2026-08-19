"""Pull OECD share price indices (forward extension of the equity panel).

The IMF stopped collecting share price indices in 2017 (see pull_imf_equity).
The OECD short-term economic statistics "Financial market" dataflow
(OECD.SDD.STES/DSD_STES@DF_FINMARK) publishes national headline share price
indices (2015=100) through the present. These carry the equity leg of the
R-Zone Monitor forward from 2017; the cleaning stage splices 3-year GROWTH
RATES across sources (the paper's convention), so the different base years do
not matter.

Query: SDMX REST, key = ``REF_AREA.FREQ.MEASURE`` followed by six wildcarded
dimensions. We pull annual indices for every country the dataflow carries;
monthly can be added later if the 2023 case study needs within-year timing.

ref_area
    Wildcarded -- all countries the dataflow carries.
freq
    ``A`` annual (period average). ``M`` and ``Q`` also exist.
measure
    ``SHARE`` share prices. This is the only measure we pin; the dataflow
    also carries interest rates and exchange rates.
UNIT_MEASURE, ACTIVITY, ADJUSTMENT, TRANSFORMATION, TIME_HORIZ, METHODOLOGY
    Wildcarded. For share prices they collapse to a single combination
    (``IX`` index, the rest not applicable), so no filtering is needed.

Output: ``oecd_share_prices.parquet``, long format, one row per
country-year::

    country_iso3 | year | value

References
----------
- OECD Data Explorer (view the series manually): search "Financial market"
  at https://data-explorer.oecd.org -- dataflow
  OECD.SDD.STES/DSD_STES@DF_FINMARK, measure SHARE. The explorer's
  "Developer API" button shows the same SDMX REST query this module calls.
- Dimensions above re-verified by requesting one recent period with
  ``format=csvfilewithlabels``, which returns every dimension code beside a
  human-readable label.
- Update frequency: monthly and annual series, published within a couple of
  months of period end; the max TIME_PERIOD in the response shows how
  current the data is.
- Citation: OECD, "Short-Term Economic Statistics: Financial Market"
  (share prices, index 2015=100), OECD Data Explorer.
- Note: FRED's international share-price series (SPASTT01...) redistribute
  this same OECD data, and can serve as a mirror if the OECD API changes.
"""

import io
from pathlib import Path

import pandas as pd
import requests

from settings import config

DATA_DIR = config("DATA_DIR")
OECD_SHARE_PRICES_FILENAME = "oecd_share_prices.parquet"

OECD_SHARE_PRICES_URL = (
    "https://sdmx.oecd.org/public/rest/data/"
    "OECD.SDD.STES,DSD_STES@DF_FINMARK,4.0/"
    ".A.SHARE......?startPeriod=1950&format=csvfilewithlabels"
)


def pull_oecd_share_prices(url=OECD_SHARE_PRICES_URL):
    """Download annual OECD share price indices for all available countries.

    :param url: SDMX REST query; its key wildcards ref_area, pins annual frequency and
        the SHARE measure, and leaves the six remaining dimensions wildcarded.
    :returns: long DataFrame with one row per country-year.
    """
    response = requests.get(url, timeout=300)
    response.raise_for_status()

    raw_df = pd.read_csv(io.BytesIO(response.content))

    share_df = raw_df[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].rename(
        columns={
            "REF_AREA": "country_iso3",
            "TIME_PERIOD": "year",
            "OBS_VALUE": "value",
        }
    )
    share_df["year"] = share_df["year"].astype(int)
    share_df["value"] = pd.to_numeric(share_df["value"], errors="coerce")
    share_df = share_df.dropna(subset=["value"]).reset_index(drop=True)

    return share_df


def load_oecd_share_prices(data_dir=DATA_DIR):
    """Load the OECD share price panel: country_iso3 | year | value.

    :param data_dir: directory holding the project's parquet files (defaults to the configured DATA_DIR).
    :returns: DataFrame with one row per country-year.
    """
    return pd.read_parquet(Path(data_dir) / OECD_SHARE_PRICES_FILENAME)


if __name__ == "__main__":
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    share_df = pull_oecd_share_prices()
    share_df.to_parquet(data_dir / OECD_SHARE_PRICES_FILENAME)

    n_countries = share_df["country_iso3"].nunique()
    year_min, year_max = share_df["year"].min(), share_df["year"].max()
    print(
        f"Saved oecd_share_prices.parquet: {share_df.shape}, "
        f"{n_countries} countries, {year_min}-{year_max}"
    )
