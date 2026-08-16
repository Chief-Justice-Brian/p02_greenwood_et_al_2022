"""Construct the post-publication systemic-crisis onset series.

Historical observations retain the paper's BVX definition through 2016. From
2017 through 2025, coverage comes from Laeven and Valencia (2026), *Systemic
Banking Crises Database: 1970--2025*, IMF Working Paper 26/94, Table A1.

The confirmed non-borderline onsets after 2016 are Ghana (2017), Republic of
Congo (2017), and Lebanon (2019). None belongs to the GHSS 42-country sample.
Nicaragua (2018), Vietnam (2022), and Sri Lanka (2023) are marked borderline
by the IMF and are deliberately excluded. The 2023 US and Swiss bank failures
are discussed by the IMF but are not classified as systemic crisis onsets.
"""

import numpy as np
import pandas as pd

BVX_END_YEAR = 2016
UPDATED_CRISIS_END_YEAR = 2025
UPDATED_CRISIS_SOURCE = "BVX through 2016; IMF Laeven-Valencia (2026) 2017-2025"

IMF_CONFIRMED_POST_2016_ONSETS = {
    ("GHA", 2017),
    ("COG", 2017),
    ("LBN", 2019),
}

IMF_BORDERLINE_POST_2016_ONSETS = {
    ("NIC", 2018),
    ("VNM", 2022),
    ("LKA", 2023),
}


def add_updated_crisis_series(panel: pd.DataFrame) -> pd.DataFrame:
    """Add a complete hybrid crisis onset through 2025 and future windows."""
    result = panel.sort_values(["country_iso3", "year"]).copy()
    result["crisis_updated"] = np.nan
    historical = result["year"].le(BVX_END_YEAR)
    result.loc[historical, "crisis_updated"] = result.loc[historical, "crisis_bvx"]

    extension = result["year"].between(BVX_END_YEAR + 1, UPDATED_CRISIS_END_YEAR)
    result.loc[extension, "crisis_updated"] = 0.0
    for country_iso3, year in IMF_CONFIRMED_POST_2016_ONSETS:
        event = (
            extension
            & result["country_iso3"].eq(country_iso3)
            & result["year"].eq(year)
        )
        result.loc[event, "crisis_updated"] = 1.0

    grouped = result.groupby("country_iso3", sort=False)["crisis_updated"]
    for horizon in range(1, 5):
        future = pd.concat(
            [grouped.shift(-lead) for lead in range(1, horizon + 1)], axis=1
        )
        outcome = future.max(axis=1)
        outcome[~future.notna().all(axis=1)] = np.nan
        result[f"crisis_updated_next_{horizon}y"] = outcome
    result["updated_crisis_source"] = UPDATED_CRISIS_SOURCE
    return result
