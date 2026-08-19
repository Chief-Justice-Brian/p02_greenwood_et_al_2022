"""Feature construction shared by the GHSS replication and its extensions."""

import numpy as np
import pandas as pd


def add_future_event_windows(panel, event_column="crisis_bvx", horizons=(1, 2, 3, 4)):
    """Add ``crisis_next_{h}y`` indicators for an event between t+1 and t+h.

    A forecast outcome is missing unless every annual event observation in its
    future window is defined.  This prevents the end of a chronology from
    silently being interpreted as a no-crisis period.

    :param panel: country-year panel containing ``event_column``.
    :param event_column: 0/1 annual event series to look ahead on.
    :param horizons: window lengths h, one new ``crisis_next_{h}y`` column
        each.
    :returns: the panel with the ``crisis_next_{h}y`` columns added.
    """
    result = panel.sort_values(["country_iso3", "year"]).copy()
    for horizon in horizons:
        leads = [
            result.groupby("country_iso3", sort=False)[event_column].shift(-lead)
            for lead in range(1, horizon + 1)
        ]
        future = pd.concat(leads, axis=1)
        outcome = future.max(axis=1)
        outcome[~future.notna().all(axis=1)] = np.nan
        result[f"crisis_next_{horizon}y"] = outcome
    return result


def quantile_cutoffs(values, quantiles):
    """Compute empirical quantile cutoffs of a sample, ignoring missing values.

    :param values: sample whose distribution defines the cutoffs.
    :param quantiles: quantile levels in (0, 1), e.g. [0.2, 0.4, 0.6, 0.8].
    :returns: {quantile: cutoff}.
    :raises ValueError: on an all-missing sample.
    """
    clean = pd.Series(values).dropna()
    if clean.empty:
        raise ValueError("cannot calculate cutoffs from an empty sample")
    return {quantile: float(clean.quantile(quantile)) for quantile in quantiles}


def assign_bucket(values, cutoffs):
    """Assign 1-based quantile buckets using fixed numeric cutoffs.

    :param values: the series to bucket; missing entries stay missing (Int64
        NA).
    :param cutoffs: quantile -> cutoff value, as produced by quantile_cutoffs.
    :returns: the bucket per value, 1 plus the number of cutoffs at or below
        the value, so buckets run from 1 to len(cutoffs) + 1.
    """
    values = pd.Series(values)
    boundaries = np.array([cutoffs[key] for key in sorted(cutoffs)], dtype=float)
    assigned = pd.Series(pd.NA, index=values.index, dtype="Int64")
    observed = values.notna()
    assigned.loc[observed] = (
        np.searchsorted(boundaries, values.loc[observed].to_numpy(), side="right") + 1
    )
    return assigned


def indicator_above(values, cutoff):
    """Flag values strictly above ``cutoff`` as 1, others as 0.

    :param values: numeric series to threshold.
    :param cutoff: fixed numeric threshold the values are compared against.
    :returns: 0/1 flags; missing values stay missing (Int64 NA) instead of
        counting as 0.
    """
    values = pd.Series(values)
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    observed = values.notna()
    result.loc[observed] = values.loc[observed].gt(cutoff).astype(int)
    return result


def add_zone_exit_buckets(panel, zone_column, max_years_since=3):
    """Add mutually exclusive time-since-exit buckets for a 0/1 zone series.

    Adds ``{zone_column}_exited_{k}y`` for k = 1..max_years_since: 1 when the
    country was last in the zone exactly k years ago and has been out ever
    since (including now), 0 otherwise, NA only when the observed years
    cannot determine the value (a currently-in-zone row is 0 in every bucket
    regardless of unobserved lags, and one observed disqualifying year is
    enough for a 0). Assumes the panel is a complete year grid per country,
    so a shift of k rows means exactly k years.

    :param panel: country-year panel containing ``zone_column``.
    :param zone_column: 0/1 zone membership series.
    :param max_years_since: deepest lookback k, one bucket column per k.
    :returns: the panel with the bucket columns added.
    """
    result = panel.sort_values(["country_iso3", "year"]).copy()
    # Force the nullable dtype: comparisons on plain int/float treat an
    # unobserved lag as a plain False instead of NA, which would turn
    # unknown history into a definite "no exit".
    zone_values = result[zone_column].astype("Int64")
    grouped = zone_values.groupby(result["country_iso3"], sort=False)
    # Three-valued (Kleene) logic on nullable booleans: False & NA is False,
    # so an observed disqualifying year yields a determinate 0 even when
    # deeper lags are unobserved.
    out_now = zone_values.eq(0)
    for years_since in range(1, max_years_since + 1):
        bucket = out_now.copy()
        # Out of the zone every year since the exit year...
        for nearer in range(1, years_since):
            bucket = bucket & grouped.shift(nearer).eq(0)
        # ...and in the zone exactly years_since years ago.
        bucket = bucket & grouped.shift(years_since).eq(1)
        result[f"{zone_column}_exited_{years_since}y"] = bucket.astype("Int64")
    return result


def indicator_below(values, cutoff):
    """Flag values strictly below ``cutoff`` as 1, others as 0.

    :param values: numeric series to threshold.
    :param cutoff: fixed numeric threshold the values are compared against.
    :returns: 0/1 flags; missing values stay missing (Int64 NA) instead of
        counting as 0.
    """
    values = pd.Series(values)
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    observed = values.notna()
    result.loc[observed] = values.loc[observed].lt(cutoff).astype(int)
    return result
