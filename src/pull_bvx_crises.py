"""Pull the Baron-Verner-Xiong (2021) banking-crisis chronology from Harvard Dataverse.

Baron, Verner, and Xiong, "Banking Crises Without Panics," Quarterly Journal of
Economics 136(1), 2021. Their replication kit is posted on Harvard Dataverse
(doi:10.7910/DVN/ECC9GE, CC0 license) as a single zip file. This is the crisis
chronology that Greenwood-Hanson-Shleifer-Sorensen (2022) use as their baseline
crisis definition.

Two files inside the zip are extracted and saved as parquet.

Output 1: ``bvx_crisis_list.parquet``, from ``Narrative Crisis List, Panics
List, and BVX List.dta`` -- one row per country-episode. Key columns:

``revised``
    Year of the crisis under the revised BVX chronology. This is THE BVX
    crisis list; NaN means the episode is not a BVX crisis.
``panic``
    1 if there was a banking panic.
``bankeqdecline``
    1 if bank equity declined more than 30 percent.
``bankfailures_widespread``
    1 if there were widespread bank failures.

Output 2: ``bvx_annual_regdata.parquet``, from ``BVX_annual_regdata.dta`` --
one row per country-year, carrying the paper's full set of regression
variables. Key columns for our project:

``RC``
    Revised BVX crisis indicator (the chronology GHSS use).
``JC``
    Joint crisis indicator.
``ReinhartRogoffCrisis``, ``LaevenValenciaCrisis``
    Alternative chronologies compared in the paper's Table 1.

The BVX list ends in 2016. The 2023 out-of-sample episode is scored separately
by our own code against BVX's stated criteria.

References
----------
- Dataset landing page (CC0 license):
  https://doi.org/10.7910/DVN/ECC9GE
- Dataverse Data Access API (the download endpoint we call):
  https://guides.dataverse.org/en/latest/api/dataaccess.html
- Dataverse Native API (dataset metadata: file ids, version history):
  https://guides.dataverse.org/en/latest/api/native-api.html
- Update frequency: a frozen replication kit, not a living dataset -- version
  1.2 since 2020-12-23 (stability is the desired property for a replication
  input). To check whether the authors ever publish a new version::

      GET https://dataverse.harvard.edu/api/datasets/:persistentId/versions
          ?persistentId=doi:10.7910/DVN/ECC9GE

  If a new version replaces the zip, the numeric file id in ``BVX_ZIP_URL``
  changes; re-derive it from the same endpoint's ``files`` listing.
- Citation: Baron, Verner and Xiong, "Replication Data for: 'Banking Crises
  without Panics'," Harvard Dataverse, V1.2, doi:10.7910/DVN/ECC9GE.
"""

import io
import time
import zipfile
from pathlib import Path

import pandas as pd
import requests

from settings import config

DATA_DIR = config("DATA_DIR")
BVX_CRISIS_LIST_FILENAME = "bvx_crisis_list.parquet"
BVX_ANNUAL_REGDATA_FILENAME = "bvx_annual_regdata.parquet"

# Direct download of "replication kit.zip" (file id 4095458) from the
# Dataverse dataset doi:10.7910/DVN/ECC9GE
BVX_ZIP_URL = "https://dataverse.harvard.edu/api/access/datafile/4095458"

# Members inside the zip that we parse into parquet files
CRISIS_LIST_MEMBER_KEYWORD = "Narrative Crisis List"
ANNUAL_REGDATA_MEMBER_SUFFIX = "BVX_annual_regdata.dta"


def pull_bvx_replication_kit(url=BVX_ZIP_URL, max_attempts=6, retry_wait_seconds=20):
    """Download the BVX replication kit zip and return it as raw bytes.

    Dataverse keeps this file in archival storage, so a request can be
    answered with "202 Accepted" and an empty body while the file is being
    restored -- and ``raise_for_status()`` treats any 2xx as success.  Only
    a 200 response whose payload starts with the zip magic bytes ``PK`` is
    accepted here; anything else is retried, then reported loudly, so a
    non-zip payload can never reach the parser or be written to disk.
    """
    last_status = None
    last_size = None
    for attempt in range(max_attempts):
        if attempt > 0:
            time.sleep(retry_wait_seconds)
        response = requests.get(url, timeout=300)
        response.raise_for_status()
        last_status = response.status_code
        last_size = len(response.content)
        if response.status_code == 200 and response.content[:2] == b"PK":
            return response.content
    raise RuntimeError(
        f"Dataverse did not serve the BVX zip after {max_attempts} attempts "
        f"(last response: status {last_status}, {last_size} bytes). The file "
        "is likely mid-restore from archival storage; re-run doit later."
    )


def parse_bvx_zip(zip_bytes):
    """Extract the crisis list and annual regression panel from the raw zip.

    Returns
    -------
    (crisis_list_df, annual_regdata_df) : tuple of pd.DataFrame
        crisis_list_df: one row per crisis episode
        annual_regdata_df: one row per country-year
    """
    zip_file = zipfile.ZipFile(io.BytesIO(zip_bytes))
    member_names = zip_file.namelist()

    crisis_list_member = [
        name for name in member_names if CRISIS_LIST_MEMBER_KEYWORD in name
    ][0]
    regdata_member = [
        name for name in member_names if name.endswith(ANNUAL_REGDATA_MEMBER_SUFFIX)
    ][0]

    # Read the .dta members straight from memory: their filenames contain
    # commas/spaces that Windows mangles if extracted to disk
    crisis_list_df = pd.read_stata(io.BytesIO(zip_file.read(crisis_list_member)))
    annual_regdata_df = pd.read_stata(io.BytesIO(zip_file.read(regdata_member)))

    return crisis_list_df, annual_regdata_df


def load_bvx_crisis_list(data_dir=DATA_DIR):
    """Load the parsed BVX crisis episode list (one row per episode)."""
    return pd.read_parquet(Path(data_dir) / BVX_CRISIS_LIST_FILENAME)


def load_bvx_annual_regdata(data_dir=DATA_DIR):
    """Load the parsed BVX country-year panel."""
    return pd.read_parquet(Path(data_dir) / BVX_ANNUAL_REGDATA_FILENAME)


if __name__ == "__main__":
    data_dir = Path(DATA_DIR)
    data_dir.mkdir(parents=True, exist_ok=True)

    zip_bytes = pull_bvx_replication_kit()

    # Keep the raw zip on disk for provenance (it is small, ~5 MB)
    raw_zip_path = data_dir / "bvx_replication_kit.zip"
    raw_zip_path.write_bytes(zip_bytes)

    crisis_list_df, annual_regdata_df = parse_bvx_zip(zip_bytes)
    crisis_list_df.to_parquet(data_dir / BVX_CRISIS_LIST_FILENAME)
    annual_regdata_df.to_parquet(data_dir / BVX_ANNUAL_REGDATA_FILENAME)

    print(f"Saved bvx_crisis_list.parquet: {crisis_list_df.shape}")
    print(f"Saved bvx_annual_regdata.parquet: {annual_regdata_df.shape}")
