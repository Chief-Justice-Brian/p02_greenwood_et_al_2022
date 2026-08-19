"""Pull the BIS Total Credit Statistics (credit to the non-financial sector).

The BIS series on total credit (https://data.bis.org, dataset WS_TC) gives
credit to households and to non-financial corporations as a percent of GDP,
quarterly, for the advanced and larger emerging economies. This is the
primary credit source in Greenwood-Hanson-Shleifer-Sorensen (2022); the IMF
Global Debt Database supplements it (3-year changes are spliced across the
two sources).

The bulk file is a single zip with one CSV: one row per series with metadata
columns, then one column per quarter (e.g. ``1947-Q1``). We melt it into a
long/tidy format and keep all series; the cleaning stage selects the relevant
slice.

Dimension codes:

freq
    ``Q`` quarterly -- the only frequency published.

tc_borrowers
    ``H`` households & NPISHs, ``N`` non-financial corporations,
    ``P`` private non-financial sector (= H + N), ``G`` general government,
    ``C`` non-financial sector (= P + G).

tc_lenders
    ``A`` all sectors, ``B`` domestic banks.

valuation
    ``M`` market value, ``N`` nominal value. Only ``M`` exists for H/N at
    percent-of-GDP, so filtering on it is currently redundant.

unit_type
    ``770`` percent of GDP, ``799`` percent of GDP at PPP exchange rates,
    ``XDC`` domestic-currency level, ``USD`` US-dollar level.

tc_adjust
    ``A`` adjusted for breaks, ``U`` unadjusted.

Output: ``bis_total_credit.parquet``, long format, one row per
series-quarter::

    freq | borrowers_country | tc_borrowers | tc_lenders | valuation |
    unit_type | tc_adjust | unit_measure | series_title | period | value

References
----------
- Topic page with interactive data explorer:
  https://data.bis.org/topics/TOTAL_CREDIT
- Methodology and series documentation:
  https://www.bis.org/statistics/totcredit.htm
- ``BIS_TOTAL_CREDIT_ZIP_URL`` is the full-dataset bulk file linked from
  data.bis.org ("CSV horizontal" format).
- Update frequency: quarterly, per the BIS statistics release calendar; the
  zip's HTTP Last-Modified header shows the current vintage date.
- Citation: BIS, "Long series on total credit to the non-financial sectors,"
  BIS Statistics (data.bis.org).
"""

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

from settings import config

DATA_DIR = config("DATA_DIR")
BIS_TOTAL_CREDIT_FILENAME = "bis_total_credit.parquet"

BIS_TOTAL_CREDIT_ZIP_URL = "https://data.bis.org/static/bulk/WS_TC_csv_col.zip"

METADATA_COLUMN_RENAMES = {
    "FREQ": "freq",
    "BORROWERS_CTY": "borrowers_country",
    "TC_BORROWERS": "tc_borrowers",
    "TC_LENDERS": "tc_lenders",
    "VALUATION": "valuation",
    "UNIT_TYPE": "unit_type",
    "TC_ADJUST": "tc_adjust",
    "UNIT_MEASURE": "unit_measure",
    "TITLE_TS": "series_title",
}


def pull_bis_total_credit(url=BIS_TOTAL_CREDIT_ZIP_URL):
    """Download the BIS WS_TC bulk file and return it in long format.

    :param url: BIS bulk-download link for the WS_TC "CSV horizontal" zip
        (defaults to BIS_TOTAL_CREDIT_ZIP_URL).
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

    # Quarter columns look like "1947-Q1"; metadata columns do not.
    period_columns = [column for column in wide_df.columns if "-Q" in column]

    long_df = wide_df.melt(
        id_vars=list(METADATA_COLUMN_RENAMES),
        value_vars=period_columns,
        var_name="period",
        value_name="value",
    ).rename(columns=METADATA_COLUMN_RENAMES)

    # Most series do not span the full 1940-present grid.
    long_df["value"] = pd.to_numeric(long_df["value"], errors="coerce")
    long_df = long_df.dropna(subset=["value"]).reset_index(drop=True)

    return long_df


def load_bis_total_credit(data_dir=DATA_DIR):
    """Load the BIS total credit panel: one row per series-quarter.

    :param data_dir: directory holding the project's parquet files (defaults
        to the configured DATA_DIR).
    """
    return pd.read_parquet(Path(data_dir) / BIS_TOTAL_CREDIT_FILENAME)


if __name__ == "__main__":
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    bis_credit_df = pull_bis_total_credit()
    bis_credit_df.to_parquet(data_dir / BIS_TOTAL_CREDIT_FILENAME)

    n_countries = bis_credit_df["borrowers_country"].nunique()

    print(
        f"Saved bis_total_credit.parquet: "
        f"{bis_credit_df.shape}, {n_countries} borrower countries"
    )
