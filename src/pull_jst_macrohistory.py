"""Pull the Jorda-Schularick-Taylor Macrohistory Database (Release 6).

The JST database (https://www.macrohistory.net/database/) is an annual panel of
advanced economies reaching back to the nineteenth century -- the deepest
history of any source in this project. It is free for academic use.

In this project JST serves as a SUPPLEMENTARY source, exactly as in
Greenwood-Hanson-Shleifer-Sorensen (2022):

equity prices
    Fill country-years missing from the IMF IFS share price indices
    (e.g. pre-1957 for several European countries).
house prices
    Fill country-years missing from the BIS/OECD residential property price
    series.
CPI / GDP
    Fill gaps in World Bank WDI, which starts only in 1960.
``crisisJST``
    The JST crisis chronology, compared against BVX and Reinhart-Rogoff in
    Table 1.

Output: ``jst_macrohistory.parquet``, the full R6 panel, one row per
country-year with every JST variable as a column. The file is small, so we keep
them all and let the cleaning stage choose.

References
----------
- Database landing page (view/download the data manually -- .dta, .xlsx,
  and full variable documentation): https://www.macrohistory.net/database/
- ``JST_DTA_URL`` is the R6 .dta link from that page (302-redirects to the
  site's CDN; requests follows it).
- Update frequency: a new release every few years; R6 (May 2022) is current.
  The landing page names the latest release -- if it advertises R7+, take
  the new download link from there.
- Citation: Jorda, Schularick and Taylor, "Macrofinancial History and the
  New Business Cycle Facts," NBER Macroeconomics Annual 31, 2017; returns
  data as augmented by Jorda et al., "The Rate of Return on Everything,
  1870-2015," QJE 134(3), 2019. Free for academic use.
"""

import io
from pathlib import Path

import pandas as pd
import requests

from settings import config

DATA_DIR = config("DATA_DIR")
JST_MACROHISTORY_FILENAME = "jst_macrohistory.parquet"

# Canonical R6 .dta link from https://www.macrohistory.net/database/
# (the site 302-redirects to a jimcontent CDN URL; requests follows it)
JST_DTA_URL = "https://www.macrohistory.net/app/download/9834512469/JSTdatasetR6.dta"

# The site occasionally rejects requests without a browser-like user agent
REQUEST_HEADERS = {"User-Agent": "Mozilla/5.0 (research replication script)"}


def pull_jst_macrohistory(url=JST_DTA_URL):
    """Download the JST R6 Stata file and return it as a DataFrame."""
    response = requests.get(url, headers=REQUEST_HEADERS, timeout=300)
    response.raise_for_status()

    jst_df = pd.read_stata(io.BytesIO(response.content))
    return jst_df


def load_jst_macrohistory(data_dir=DATA_DIR):
    """Load the JST panel: one row per (country, year)."""
    return pd.read_parquet(Path(data_dir) / JST_MACROHISTORY_FILENAME)


if __name__ == "__main__":
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    jst_df = pull_jst_macrohistory()
    jst_df.to_parquet(data_dir / JST_MACROHISTORY_FILENAME)

    n_countries = jst_df["country"].nunique()
    year_min, year_max = int(jst_df["year"].min()), int(jst_df["year"].max())
    print(
        f"Saved jst_macrohistory.parquet: {jst_df.shape}, "
        f"{n_countries} countries, {year_min}-{year_max}"
    )
