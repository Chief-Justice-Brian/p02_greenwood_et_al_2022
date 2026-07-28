"""Pull the OECD Analytical House Price indicators.

The OECD analytical house price database (OECD.ECO.MPD/DSD_AN_HOUSE_PRICES)
provides real and nominal house price indices, quarterly and annual, for the
OECD membership plus major non-members, through the present. It supplements the
BIS selected residential property prices (which reach further back for some
countries) and JST (which reaches back furthest, for the advanced economies it
covers). The cleaning stage splices 3-year growth rates across the three
sources, the paper's convention.

We pull the ENTIRE dataflow (it is small) and let the cleaning stage pick the
measures it wants; the rest come along and are ignored.

freq
    ``A`` annual, ``Q`` quarterly -- both published for every measure.
measure
    ``RHP`` real house price index (the household-sector R-zone input),
    ``HPI`` nominal house price index, ``RPI`` rent prices,
    ``HPI_RPI`` price-to-rent ratio, ``HPI_YDH`` price-to-income ratio,
    ``HPI_RPI_AVG`` and ``HPI_YDH_AVG`` the same two ratios standardised to
    their long-run averages.

Output: ``oecd_house_prices.parquet``, long format, one row per
country-measure-period::

    country_iso3 | freq | measure | period | value

References
----------
- OECD Data Explorer (view the series manually): search "Analytical house
  prices" at https://data-explorer.oecd.org -- dataflow
  OECD.ECO.MPD/DSD_AN_HOUSE_PRICES@DF_HOUSE_PRICES. The explorer's
  "Developer API" button shows the same SDMX REST query this module calls.
- Measure codes above re-verified by requesting one recent period with
  ``format=csvfilewithlabels``, which returns the ``MEASURE`` code beside a
  human-readable ``Measure`` label.
- Update frequency: quarterly; the max TIME_PERIOD in the response shows how
  current the data is.
- Citation: OECD, "Analytical house price indicators," OECD Data Explorer.
"""

import io
from pathlib import Path

import pandas as pd
import requests

from settings import config

DATA_DIR = config("DATA_DIR")

OECD_HOUSE_PRICES_URL = (
    "https://sdmx.oecd.org/public/rest/data/"
    "OECD.ECO.MPD,DSD_AN_HOUSE_PRICES@DF_HOUSE_PRICES,1.0/"
    "all?startPeriod=1950&format=csvfilewithlabels"
)


def pull_oecd_house_prices(url=OECD_HOUSE_PRICES_URL):
    """Download the full OECD analytical house price dataflow."""
    response = requests.get(url, timeout=300)
    response.raise_for_status()

    raw_df = pd.read_csv(io.BytesIO(response.content))

    house_df = raw_df[["REF_AREA", "FREQ", "MEASURE", "TIME_PERIOD", "OBS_VALUE"]].rename(
        columns={
            "REF_AREA": "country_iso3",
            "FREQ": "freq",
            "MEASURE": "measure",
            "TIME_PERIOD": "period",
            "OBS_VALUE": "value",
        }
    )
    house_df["value"] = pd.to_numeric(house_df["value"], errors="coerce")
    house_df = house_df.dropna(subset=["value"]).reset_index(drop=True)

    return house_df


def load_oecd_house_prices(data_dir=DATA_DIR):
    """Load the OECD house price panel: country_iso3 | freq | measure | period | value."""
    return pd.read_parquet(Path(data_dir) / "oecd_house_prices.parquet")


if __name__ == "__main__":
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    house_df = pull_oecd_house_prices()
    house_df.to_parquet(data_dir / "oecd_house_prices.parquet")

    n_countries = house_df["country_iso3"].nunique()
    print(f"Saved oecd_house_prices.parquet: {house_df.shape}, {n_countries} countries")
    print(f"Measures: {sorted(house_df['measure'].unique())}")
