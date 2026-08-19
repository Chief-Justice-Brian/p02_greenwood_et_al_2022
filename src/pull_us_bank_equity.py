"""Pull the daily CRSP value-weighted US Banks portfolio from the Ken French Data Library.

The French Data Library's 49-industry daily portfolios include a "Banks"
portfolio: value-weighted total returns across US-listed commercial banks,
built from CRSP. Its role in this project: Baron-Verner-Xiong (2021) identify
banking crises partly from bank equity declines of 30 percent or more, and
their chronology ends in 2016. This module extracts the Banks portfolio and
builds a cumulative total-return index used to extend that criterion past
2016. The index-selection rationale and its validation against BVX's own US
series are documented in ``docs_src/project_overview/methodology.md``.

Source file layout (CSV inside a zip, after a prose preamble):

``Average Value Weighted Returns -- Daily``
    The first data block, used here. One row per trading day, dates as
    YYYYMMDD, one column per industry, returns in percent. A value of
    -99.99 marks a missing observation.
``Average Equal Weighted Returns -- Daily``
    A second block below the first; not used.

Output: ``us_bank_equity.parquet``, one row per trading day::

    date | daily_return | index_level

``daily_return`` is the Banks portfolio's value-weighted total return in
percent; ``index_level`` is the cumulative total-return index built from
those returns.

References
----------
- Data library landing page (the data can be viewed manually in a browser):
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/data_library.html
- File pulled by this module (49 Industry Portfolios [Daily], CSV zip):
  https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/49_Industry_Portfolios_daily_CSV.zip
- Update frequency: refreshed on the library's regular schedule (roughly
  monthly, with a lag of one to two months behind the market). Re-running
  the pull always retrieves the latest vintage; the file's preamble states
  the coverage end date.
- Citation: Kenneth R. French, "49 Industry Portfolios (Daily)," Data
  Library, Tuck School of Business at Dartmouth; underlying data from CRSP.
"""

import io
import zipfile
from pathlib import Path

import pandas as pd
import requests

from settings import config

DATA_DIR = config("DATA_DIR")
US_BANK_EQUITY_FILENAME = "us_bank_equity.parquet"

FRENCH_49_INDUSTRY_DAILY_URL = (
    "https://mba.tuck.dartmouth.edu/pages/faculty/ken.french/ftp/"
    "49_Industry_Portfolios_daily_CSV.zip"
)
# The library serves plain files but rejects the default python agent.
BROWSER_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
    )
}


def _is_daily_date_field(field: str) -> bool:
    """Return True when the field is an 8-digit YYYYMMDD date (a daily data row).

    :param field: first comma-separated field of a line from the source CSV,
        possibly padded with whitespace.
    :type field: str
    :returns: True when the field is an 8-digit YYYYMMDD date.
    :rtype: bool
    """
    stripped = field.strip()
    return len(stripped) == 8 and stripped.isdigit()


def _extract_value_weighted_block(text):
    """Carve the first (value-weighted) daily block out of the French file.

    The file is not a plain CSV; its layout is::

        This file was created by CMPT_IND_RETS ...    <- prose preamble
        Missing data are indicated by -99.99 ...
        ...
          Average Value Weighted Returns -- Daily     <- block title
        ,Agric,Food ,...,Banks,...                    <- header row (no name
        19260701,  0.43, 0.11, ...                       for the date column)
        19260702, ...                                 <- one row per trading
        ...                                              day
          Average Equal Weighted Returns -- Daily     <- next block: stop

    So the block starts at the first YYYYMMDD-dated row, its header sits on
    the line directly above, and it ends where the dated rows stop.

    :param text: full decoded text of the CSV member, preamble included.
    :returns: (column names, list of data lines) for the block.
    """
    lines = text.splitlines()
    data_start = next(
        i for i, line in enumerate(lines) if _is_daily_date_field(line.split(",")[0])
    )
    columns = ["date"] + [
        name.strip() for name in lines[data_start - 1].split(",")[1:]
    ]

    block_lines = []
    for line in lines[data_start:]:
        if not _is_daily_date_field(line.split(",")[0]):
            break
        block_lines.append(line)

    return columns, block_lines


def _banks_series(columns, block_lines):
    """Parse the block and reduce it to the Banks daily-return series.

    :param columns: column names for the block, date first.
    :param block_lines: the block's data lines, one per trading day.
    :returns: DataFrame with date and daily_return (percent) columns.
    """
    frame = pd.read_csv(io.StringIO("\n".join(block_lines)), names=columns)
    frame["date"] = pd.to_datetime(frame["date"], format="%Y%m%d")

    banks = frame[["date", "Banks"]].rename(columns={"Banks": "daily_return"})
    banks["daily_return"] = pd.to_numeric(banks["daily_return"], errors="coerce")
    # -99.99 marks a missing observation in the French files.
    return banks[banks["daily_return"] > -99].reset_index(drop=True)


def pull_us_bank_equity(url=FRENCH_49_INDUSTRY_DAILY_URL):
    """Download the 49-industry daily file and return the Banks series.

    :param url: link to the French library's 49-industry daily CSV zip
        (defaults to FRENCH_49_INDUSTRY_DAILY_URL).
    :returns: the Banks series as a DataFrame, one row per trading day.
    :raises RuntimeError: if the parsed file yields no usable Banks returns.
    """
    response = requests.get(url, headers=BROWSER_HEADERS, timeout=300)
    response.raise_for_status()

    archive = zipfile.ZipFile(io.BytesIO(response.content))
    csv_member = next(
        member
        for member in archive.namelist()
        if member.lower().endswith(".csv")
    )
    text = archive.read(csv_member).decode("latin-1")

    columns, block_lines = _extract_value_weighted_block(text)
    banks = _banks_series(columns, block_lines)
    if banks.empty:
        raise RuntimeError("French 49-industry file yielded no usable Banks returns")

    banks["index_level"] = (1.0 + banks["daily_return"] / 100.0).cumprod()
    return banks


def load_us_bank_equity(data_dir=DATA_DIR):
    """Load the daily US bank equity portfolio: one row per trading day.

    :param data_dir: directory holding the project's parquet files (defaults to the configured DATA_DIR).
    :returns: DataFrame with one row per trading day.
    """
    return pd.read_parquet(Path(data_dir) / US_BANK_EQUITY_FILENAME)


if __name__ == "__main__":
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    bank_equity = pull_us_bank_equity()
    bank_equity.to_parquet(data_dir / US_BANK_EQUITY_FILENAME, index=False)
    print(
        f"Saved us_bank_equity.parquet: {bank_equity.shape}, "
        f"{bank_equity.date.min().date()} to {bank_equity.date.max().date()}"
    )
