"""Pull the BIS Residential Property Price statistics (selected series, WS_SPP).

The BIS "selected" residential property price dataset (https://data.bis.org,
dataset WS_SPP) provides one headline house price series per country,
quarterly, real (CPI-deflated) and nominal. Together with OECD house prices and
JST it feeds the household-sector R-zone variables in
Greenwood-Hanson-Shleifer-Sorensen (2022).

``ref_area`` mixes countries with regional aggregates (``4T`` emerging markets,
``5R`` advanced economies, ``XM`` euro area, ``XW`` world), so the cleaning
stage must filter to our panel's countries rather than treat every ``ref_area``
as one.

The bulk file is a single zip with one wide CSV: one row per series with
metadata columns, then one column per quarter. We melt it into long format and
keep ALL series; the cleaning stage selects the relevant slice.

Dimension codes:

freq
    ``Q`` quarterly -- the only frequency published.
value_type
    ``R`` real (CPI-deflated), ``N`` nominal. Named ``VALUE`` in the source
    CSV; renamed here to avoid colliding with the observation column.
unit_measure
    ``628`` index (2010 = 100), ``771`` year-on-year change in per cent.

Output: ``bis_property_prices.parquet``, long format, one row per
series-quarter::

    freq | ref_area | value_type | unit_measure | series_title | period | value

References
----------
- Topic page with interactive data explorer (view any series manually):
  https://data.bis.org/topics/RPP
- Methodology and series documentation:
  https://www.bis.org/statistics/pp.htm
- ``BIS_PROPERTY_ZIP_URL`` is the full-dataset bulk file linked from
  data.bis.org ("CSV horizontal" format).
- Codes above re-verified without downloading anything, via the BIS SDMX
  registry -- values actually published:
  https://stats.bis.org/api/v1/availableconstraint/WS_SPP/all/all
  and their labels:
  https://stats.bis.org/api/v2/structure/codelist/BIS/CL_VALUE/latest
  (likewise CL_BIS_UNIT for the unit_measure codes).
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

BIS_PROPERTY_ZIP_URL = "https://data.bis.org/static/bulk/WS_SPP_csv_col.zip"

METADATA_COLUMN_RENAMES = {
    "FREQ": "freq",
    "REF_AREA": "ref_area",
    "VALUE": "value_type",
    "UNIT_MEASURE": "unit_measure",
    "TITLE_TS": "series_title",
}


def pull_bis_property_prices(url=BIS_PROPERTY_ZIP_URL):
    """Download the BIS WS_SPP bulk zip and melt the wide CSV to long format."""
    response = requests.get(url, timeout=300)
    response.raise_for_status()

    zip_file = zipfile.ZipFile(io.BytesIO(response.content))
    csv_member = zip_file.namelist()[0]
    with zip_file.open(csv_member) as csv_handle:
        wide_df = pd.read_csv(csv_handle, dtype=str)

    # Quarter columns look like "1927-Q1"; metadata columns do not
    period_columns = [col for col in wide_df.columns if "-Q" in col]
    code_columns = list(METADATA_COLUMN_RENAMES)

    long_df = wide_df.melt(
        id_vars=code_columns,
        value_vars=period_columns,
        var_name="period",
        value_name="value",
    )
    long_df = long_df.rename(columns=METADATA_COLUMN_RENAMES)

    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["value"]).reset_index(drop=True)

    return long_df


def load_bis_property_prices(data_dir=DATA_DIR):
    """Load the BIS property price long panel (all series, quarterly)."""
    return pd.read_parquet(Path(data_dir) / "bis_property_prices.parquet")


if __name__ == "__main__":
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    bis_property_df = pull_bis_property_prices()
    bis_property_df.to_parquet(data_dir / "bis_property_prices.parquet")

    n_areas = bis_property_df["ref_area"].nunique()
    print(f"Saved bis_property_prices.parquet: {bis_property_df.shape}, {n_areas} reference areas")
