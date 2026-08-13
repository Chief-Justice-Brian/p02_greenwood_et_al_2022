"""Feature construction shared by the GHSS replication and its extensions."""

import numpy as np
import pandas as pd


def add_future_event_windows(panel, event_column="crisis_bvx", horizons=(1, 2, 3, 4)):
    """Add indicators for an event occurring between t+1 and t+h.

    A forecast outcome is missing unless every annual event observation in its
    future window is defined.  This prevents the end of a chronology from
    silently being interpreted as a no-crisis period.
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
    clean = pd.Series(values).dropna()
    if clean.empty:
        raise ValueError("cannot calculate cutoffs from an empty sample")
    return {quantile: float(clean.quantile(quantile)) for quantile in quantiles}


def assign_bucket(values, cutoffs):
    """Assign 1-based quantile buckets using fixed numeric cutoffs."""
    values = pd.Series(values)
    boundaries = np.array([cutoffs[key] for key in sorted(cutoffs)], dtype=float)
    assigned = pd.Series(pd.NA, index=values.index, dtype="Int64")
    observed = values.notna()
    assigned.loc[observed] = (
        np.searchsorted(boundaries, values.loc[observed].to_numpy(), side="right") + 1
    )
    return assigned


def indicator_above(values, cutoff):
    values = pd.Series(values)
    result = pd.Series(pd.NA, index=values.index, dtype="Int64")
    observed = values.notna()
    result.loc[observed] = values.loc[observed].gt(cutoff).astype(int)
    return result
