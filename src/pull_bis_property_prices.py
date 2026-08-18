"""Pull BIS residential property price statistics (selected series, WS_SPP).

The BIS "selected" residential property price dataset (WS_SPP) provides one
headline house price series per country, quarterly, in real and nominal terms.
Together with OECD house prices and JST, it feeds the household-sector R-zone
variables in Greenwood-Hanson-Shleifer-Sorensen (2022).

``ref_area`` includes regional aggregates (``4T`` emerging markets, ``5R``
advanced economies, ``XM`` euro area, ``XW`` world) as well as countries, so
the cleaning stage filters to the countries in our panel rather than treating
every ``ref_area`` as a country.

The bulk file is a single zip containing one wide CSV: one row per series with
metadata columns, followed by one column per quarter. This script melts it to
long format and keeps all series; the cleaning stage selects the relevant
slice.

Dimension codes:

freq
    ``Q`` quarterly.

value_type
    ``R`` real (CPI-deflated), ``N`` nominal. Named ``VALUE`` in the source
    CSV; renamed here to avoid colliding with the observation column.

unit_measure
    ``628`` index (2010 = 100), ``771`` year-on-year change in percent.

Output: ``bis_property_prices.parquet``, with one row per series-quarter::

    freq | ref_area | value_type | unit_measure | series_title | period | value

References
----------
- Topic page with interactive data explorer (view any series manually):
  https://data.bis.org/topics/RPP
- BIS methodology and series documentation:
  https://www.bis.org/statistics/pp.htm
- The dimension codes can be verified against the BIS SDMX registry:
  https://stats.bis.org/api/v1/availableconstraint/WS_SPP/all/all
  with labels at
  https://stats.bis.org/api/v2/structure/codelist/BIS/CL_VALUE/latest
- Update frequency: quarterly; the zip's HTTP Last-Modified header shows the
  current vintage date.
- Citation: BIS, "Residential property prices: selected series,"
  BIS Statistics (data.bis.org).
"""

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

from settings import config

DATA_DIR = config("DATA_DIR")
BIS_PROPERTY_PRICES_FILENAME = "bis_property_prices.parquet"

BIS_PROPERTY_ZIP_URL = "https://data.bis.org/static/bulk/WS_SPP_csv_col.zip"

METADATA_COLUMN_RENAMES = {
    "FREQ": "freq",
    "REF_AREA": "ref_area",
    "VALUE": "value_type",
    "UNIT_MEASURE": "unit_measure",
    "TITLE_TS": "series_title",
}


def pull_bis_property_prices(url=BIS_PROPERTY_ZIP_URL):
    """Download the BIS WS_SPP bulk file and return it in long format.

    :param url: BIS bulk-download link for the WS_SPP "CSV horizontal" zip
        (defaults to BIS_PROPERTY_ZIP_URL).
    :returns: long DataFrame with one row per series-quarter.
    """
    response = requests.get(url, timeout=300)
    response.raise_for_status()

    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    csv_member = next(
        member
        for member in zip_file.namelist()
        if member.lower().endswith(".csv")
    )

    with zip_file.open(csv_member) as csv_handle:
        wide_df = pd.read_csv(csv_handle, dtype=str)

    # Quarter columns look like "1927-Q1"; metadata columns never contain "-Q"
    period_columns = [column for column in wide_df.columns if "-Q" in column]

    long_df = wide_df.melt(
        id_vars=list(METADATA_COLUMN_RENAMES),
        value_vars=period_columns,
        var_name="period",
        value_name="value",
    ).rename(columns=METADATA_COLUMN_RENAMES)

    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["value"]).reset_index(drop=True)

    return long_df


def load_bis_property_prices(data_dir=DATA_DIR):
    """Load the BIS property price panel: one row per series-quarter.

    :param data_dir: directory holding the project's parquet files (defaults
        to the configured DATA_DIR).
    """
    return pd.read_parquet(Path(data_dir) / BIS_PROPERTY_PRICES_FILENAME)


if __name__ == "__main__":
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    bis_property_df = pull_bis_property_prices()
    bis_property_df.to_parquet(
        data_dir / BIS_PROPERTY_PRICES_FILENAME
    )

    n_areas = bis_property_df["ref_area"].nunique()

    print(
        f"Saved bis_property_prices.parquet: "
        f"{bis_property_df.shape}, {n_areas} reference areas"
    )
