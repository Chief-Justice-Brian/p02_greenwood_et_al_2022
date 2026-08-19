"""Construct the post-publication systemic-crisis onset series.

Historical observations retain the paper's BVX definition through 2016; the
extension applies the same definition forward, so the series means one thing
in every year. The inputs are the BVX replication kit and the CRSP bank
portfolio (``pull_us_bank_equity.py``).

``crisis_bvx_extended``
    BVX through 2016, then our own extension of BVX's stated criteria
    (Baron, Verner and Xiong 2021: a bank equity index decline of 30 percent
    or more, plus narrative evidence of widespread bank failures or a
    banking panic). No published chronology applies these criteria past
    2016, so the label is constructed here. The narrative screen over the
    GHSS sample for 2017-2025 yields three candidate episodes:

    United States, 2023
        Failures: SVB (closed 2023-03-10, $209bn assets, FDIC PR-23-019),
        Signature (2023-03-12, $110bn, FDIC PR-23-018), and First Republic
        (2023-05-01, $229bn, sold to JPMorgan) -- the second, third, and
        fourth largest US bank failures on record. Panic: $42bn left SVB in
        one day with roughly $100bn queued (Fed Vice Chair Barr testimony,
        2023-03-28); the systemic risk exception was invoked and discount
        window borrowing hit a record $152.9bn. The equity criterion is not
        asserted -- it is computed below from the pulled CRSP value-weighted
        bank portfolio (``pull_us_bank_equity.py``), and the label is set
        only if the data clears the 30 percent bar. The sensitivity of this
        call to the index choice is discussed in
        ``docs_src/project_overview/methodology.md``.
    Switzerland, 2023
        Credit Suisse suffered an institution-specific run and assisted
        merger, but failures were not widespread (one institution) and the
        aggregate Swiss bank index, dominated by UBS, did not approach a 30
        percent decline. Does not fire.
    Russia, 2022
        Bank equity collapsed and deposit runs occurred after the invasion
        of Ukraine, but BVX's own chronology carries an ``exclude_war``
        convention for war-driven episodes, which we apply. Does not fire.

    All other sample country-years in 2017-2025 show no narrative evidence
    of widespread failures or panics, so under BVX's joint (AND) condition
    no equity measurement is required for them and they are coded zero.

Forward-looking crisis windows are attached at horizons one through four,
requiring the full future window to be observed.
"""

import numpy as np
import pandas as pd

BVX_END_YEAR = 2016
UPDATED_CRISIS_END_YEAR = 2025
BVX_EXTENDED_CRISIS_SOURCE = (
    "BVX through 2016; authors' extension of BVX criteria 2017-2025"
)

# BVX's mechanical criterion: bank equity down 30%+ from its peak within
# roughly the preceding trading year.
BVX_EQUITY_DECLINE_THRESHOLD = 0.30
TRAILING_TRADING_DAYS = 252

# The one post-2016 candidate that passes the narrative screen and awaits
# the data's verdict on the equity criterion (see module docstring).
US_CRISIS_CANDIDATE_YEAR = 2023


def max_drawdown_in_year(bank_equity: pd.DataFrame, year: int) -> float:
    """Compute the deepest decline from the trailing one-year peak in a year.

    :param bank_equity: daily series with date and index_level columns.
    :param year: calendar year whose maximum drawdown is returned.
    :returns: the drawdown as a positive fraction (0.35 = a 35% decline).
    :raises ValueError: if the bank equity series has no observations in the
        requested year.
    """
    levels = bank_equity.sort_values("date").reset_index(drop=True)
    trailing_peak = (
        levels["index_level"].rolling(TRAILING_TRADING_DAYS, min_periods=1).max()
    )
    drawdown = 1.0 - levels["index_level"] / trailing_peak
    in_year = levels["date"].dt.year.eq(year)
    if not in_year.any():
        raise ValueError(f"bank equity series has no observations in {year}")
    return float(drawdown[in_year].max())


def _add_forward_windows(result: pd.DataFrame, column: str) -> None:
    """Attach {column}_next_{h}y outcomes requiring full future windows.

    Modifies result in place; horizons run one through four years.

    :param result: country-year frame ordered by year within each country.
    :param column: name of the 0/1 onset column the forward windows are built
        from.
    """
    grouped = result.groupby("country_iso3", sort=False)[column]
    for horizon in range(1, 5):
        future = pd.concat(
            [grouped.shift(-lead) for lead in range(1, horizon + 1)], axis=1
        )
        outcome = future.max(axis=1)
        outcome[~future.notna().all(axis=1)] = np.nan
        result[f"{column}_next_{horizon}y"] = outcome


def add_bvx_extended_crisis_series(
    panel: pd.DataFrame, bank_equity: pd.DataFrame
) -> pd.DataFrame:
    """Add the BVX-criteria onset series extended by us through 2025.

    The narrative screen (module docstring) leaves the United States 2023 as
    the only candidate; its label is decided here by the pulled bank equity
    data, not asserted.

    :param panel: country-year analysis panel carrying the crisis_bvx column.
    :param bank_equity: daily US bank equity series with date and index_level
        columns.
    """
    result = panel.sort_values(["country_iso3", "year"]).copy()
    result["crisis_bvx_extended"] = np.nan
    historical = result["year"].le(BVX_END_YEAR)
    result.loc[historical, "crisis_bvx_extended"] = result.loc[
        historical, "crisis_bvx"
    ]

    extension = result["year"].between(BVX_END_YEAR + 1, UPDATED_CRISIS_END_YEAR)
    result.loc[extension, "crisis_bvx_extended"] = 0.0

    us_2023_drawdown = max_drawdown_in_year(bank_equity, US_CRISIS_CANDIDATE_YEAR)
    if us_2023_drawdown >= BVX_EQUITY_DECLINE_THRESHOLD:
        us_2023 = (
            extension
            & result["country_iso3"].eq("USA")
            & result["year"].eq(US_CRISIS_CANDIDATE_YEAR)
        )
        result.loc[us_2023, "crisis_bvx_extended"] = 1.0

    _add_forward_windows(result, "crisis_bvx_extended")
    # Keep the measured drawdown on the panel so every report statistic
    # about the 2023 call is generated from data, not typed by hand.
    result["us_2023_bank_equity_drawdown"] = us_2023_drawdown
    result["bvx_extended_crisis_source"] = BVX_EXTENDED_CRISIS_SOURCE
    return result
