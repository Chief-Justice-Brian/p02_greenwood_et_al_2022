"""Shared helpers for building the spliced country-year panel.

The paper's splicing convention (WP 20-130, footnote 6): when combining data
from different sources for a country, compute 3-year CHANGES separately within
each source, then splice the resulting changes. These helpers implement that
convention once, for use by the credit, equity, and house-price cleaning
modules.
"""

import numpy as np
import pandas as pd


def delta3_within_source(source_df, value_column, is_log=False):
    """Compute 3-year changes of ``value_column`` within one source.

    Parameters
    ----------
    source_df : DataFrame with columns country_iso3 | year | <value_column>
    value_column : name of the level column
    is_log : if True, the input is a LOG level, so the 3-year change is a
        log change and is multiplied by 100 (percent); if False, the input is
        already in the final unit (e.g. percent of GDP) and the 3-year change
        is a simple difference (percentage points).

    Returns
    -------
    DataFrame with columns country_iso3 | year | delta3

    The 3-year change at year t uses the value at t-3 ONLY if the source has
    an unbroken (t-3, t) pair; interior gaps produce NaN, never a change
    computed across a hole.
    """
    frames = []
    for country, country_df in source_df.groupby("country_iso3"):
        # Reindex to a full year grid so .shift(3) means "exactly 3 years ago"
        series = (
            country_df.set_index("year")[value_column]
            .sort_index()
        )
        full_grid = range(int(series.index.min()), int(series.index.max()) + 1)
        series = series.reindex(full_grid)

        delta3 = series - series.shift(3)
        if is_log:
            delta3 = 100.0 * delta3

        country_frame = delta3.dropna().rename("delta3").reset_index()
        country_frame.columns = ["year", "delta3"]
        country_frame["country_iso3"] = country
        frames.append(country_frame)

    if not frames:
        return pd.DataFrame(columns=["country_iso3", "year", "delta3"])
    result = pd.concat(frames, ignore_index=True)
    return result[["country_iso3", "year", "delta3"]]


def splice_by_priority(sources_in_priority_order):
    """Splice per-source 3-year changes: first non-missing source wins.

    Parameters
    ----------
    sources_in_priority_order : list of (source_label, DataFrame) where each
        DataFrame has columns country_iso3 | year | delta3. Earlier entries
        have higher priority.

    Returns
    -------
    DataFrame with columns country_iso3 | year | delta3 | source
    """
    spliced = None
    for source_label, source_df in sources_in_priority_order:
        labeled = source_df.copy()
        labeled["source"] = source_label
        if spliced is None:
            spliced = labeled
            continue
        # Keep only country-years the higher-priority sources do not cover
        existing_keys = pd.MultiIndex.from_frame(spliced[["country_iso3", "year"]])
        candidate_keys = pd.MultiIndex.from_frame(labeled[["country_iso3", "year"]])
        is_new = ~candidate_keys.isin(existing_keys)
        spliced = pd.concat([spliced, labeled[is_new]], ignore_index=True)

    spliced = spliced.sort_values(["country_iso3", "year"]).reset_index(drop=True)
    return spliced


def annual_average_from_quarterly(quarterly_df, value_column="value"):
    """Collapse a quarterly panel to annual averages.

    Input: DataFrame with columns country_iso3 | period ("1990-Q1") | <value>.
    Output: DataFrame with columns country_iso3 | year | <value> where the
    value is the mean over available quarters in that year.
    """
    working = quarterly_df.copy()
    working["year"] = working["period"].str.slice(0, 4).astype(int)
    annual = (
        working.groupby(["country_iso3", "year"])[value_column]
        .mean()
        .reset_index()
    )
    return annual


def log_positive(series):
    """np.log that maps non-positive values to NaN instead of raising."""
    values = series.astype(float).copy()
    values[values <= 0] = np.nan
    return np.log(values)
