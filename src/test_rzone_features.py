"""Unit tests for the ``rzone_features`` functions on small hand-built inputs."""

import numpy as np
import pandas as pd

from rzone_features import (
    add_future_event_windows,
    add_zone_exit_buckets,
    assign_bucket,
    indicator_above,
    indicator_below,
    quantile_cutoffs,
)


def test_future_event_windows_excludes_current_year():
    # A crisis in year t must not count toward crisis_next_h at origin t.
    panel = pd.DataFrame(
        {
            "country_iso3": ["AAA"] * 5,
            "year": [2000, 2001, 2002, 2003, 2004],
            "crisis_bvx": [1, 0, 0, 1, 0],
        }
    )
    result = add_future_event_windows(panel, horizons=(1, 3))
    assert result.loc[result.year.eq(2000), "crisis_next_1y"].item() == 0
    assert result.loc[result.year.eq(2000), "crisis_next_3y"].item() == 1


def test_future_event_window_requires_complete_chronology():
    # A chronology gap inside the window must yield NaN, not a false zero.
    panel = pd.DataFrame(
        {
            "country_iso3": ["AAA"] * 4,
            "year": [2000, 2001, 2002, 2003],
            "crisis_bvx": [0, 0, np.nan, 0],
        }
    )
    result = add_future_event_windows(panel, horizons=(2,))
    assert np.isnan(result.loc[result.year.eq(2000), "crisis_next_2y"].item())


def test_fixed_cutoffs_and_rzone_assignment():
    # Cutoffs, buckets, and indicators must compose into the R-zone AND
    # condition, with missing inputs propagating to NA instead of a silent zero.
    cutoffs = quantile_cutoffs(pd.Series(range(1, 11)), [0.2, 0.8])
    buckets = assign_bucket(pd.Series([1.0, 5.0, 10.0, np.nan]), cutoffs)
    high_debt = indicator_above(pd.Series([5.0, 10.0, np.nan]), cutoffs[0.8])
    high_price = indicator_above(pd.Series([9.0, 10.0, np.nan]), 9.5)
    rzone = high_debt * high_price

    assert buckets.tolist() == [1, 2, 3, pd.NA]
    assert high_debt.tolist() == [0, 1, pd.NA]
    assert rzone.tolist() == [0, 1, pd.NA]


def test_zone_exit_buckets_track_time_since_membership():
    # Exit buckets must be mutually exclusive, fire at the exact distance
    # since the last zone year, and be NA only when the observed years truly
    # cannot determine the value: an in-zone row is 0 in every bucket, and
    # one observed disqualifying year is enough for a 0 despite missing lags.
    panel = pd.DataFrame(
        {
            "country_iso3": ["AAA"] * 6,
            "year": [2000, 2001, 2002, 2003, 2004, 2005],
            "zone": [0, 1, 0, 0, 0, 1],
        }
    )
    result = add_zone_exit_buckets(panel, "zone")
    assert result["zone_exited_1y"].tolist() == [pd.NA, 0, 1, 0, 0, 0]
    assert result["zone_exited_2y"].tolist() == [pd.NA, 0, 0, 1, 0, 0]
    assert result["zone_exited_3y"].tolist() == [pd.NA, 0, 0, 0, 1, 0]


def test_indicator_below_flags_thin_values_and_keeps_na():
    # The lev fragility flag marks THIN capital, so the flipped indicator must
    # fire strictly below the cutoff, stay 0 at it, and propagate missing.
    flags = indicator_below(pd.Series([1.0, 5.0, 9.0, np.nan]), 5.0)
    assert flags.tolist() == [1, 0, 0, pd.NA]
