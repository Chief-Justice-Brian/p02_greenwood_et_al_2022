"""Published values from Greenwood, Hanson, Shleifer and Sorensen (WP 20-130).

Every number our replication is validated against lives here, transcribed from
Table 1 (printed p. 39). ``test_table1_validation.py`` checks our recomputed
panel against paper-derived tolerances, and ``table1_summary_stats.py``
builds the side-by-side comparison table from them. Nothing else in the
codebase should hard-code a number from the paper.

Units: debt/GDP changes in percentage points, price growth in percent (3-year
log changes in real indices), indicator rows in percent.

Published values:

``TABLE1_PUBLISHED_ROWS``
    row key -> (N, mean, SD), all twelve rows of Table 1. The full layout, and
    the superset the three convenience views below draw from.
``DEBT_GROWTH_QUANTILES``
    sector -> {quantile: cutoff}. The 0.80 entries are the R-zone debt gates.
``PRICE_GROWTH_TERCILES``
    variable -> {quantile: cutoff}. The 2/3 entries are the R-zone price gates.
``PRIVATE_DEBT_LOG_QUANTILES``
    quantile -> cutoff, for the total real debt row.
``SAMPLE_SIZES``, ``PREDICTOR_MOMENTS``, ``CRISIS_INDICATOR_MEANS``
    Views of the N, (mean, SD) and mean columns of ``TABLE1_PUBLISHED_ROWS``
    for the rows the tests reach for most.
``N_SAMPLE_COUNTRIES``
    The paper's country count; ``country_sample.py`` holds the names.

Tolerance formulas live in ``replication_validation.py`` and are derived from
the published standard deviations and quantile spans stored here.
"""

# Bottom panel of Table 1: quantiles of the predictor distributions,
# computed on the paper's full 1950-2016 panel
DEBT_GROWTH_QUANTILES = {
    # sector -> {quantile: published value}
    "business": {0.20: -2.75, 0.40: 1.03, 0.60: 3.99, 0.80: 8.99},
    "household": {0.20: -0.26, 0.40: 1.63, 0.60: 3.94, 0.80: 7.60},
}

PRICE_GROWTH_TERCILES = {
    # variable -> {quantile: published value}
    "equity": {1 / 3: -8.52, 2 / 3: 26.56},
    "house_price": {1 / 3: -0.35, 2 / 3: 12.67},
}

# Table 1: sample sizes (country-years)
SAMPLE_SIZES = {
    "bvx_crisis_indicator": 1281,
    "business_pairs": 1258,  # business debt growth rows (needs equity pair)
    "household_pairs": 1107,  # household debt growth rows (needs house pair)
}

# Table 1: means and standard deviations of the predictors (percent)
PREDICTOR_MOMENTS = {
    # variable -> (mean, sd)
    "delta3_business_debt_gdp": (3.86, 20.74),
    "delta3_household_debt_gdp": (3.58, 5.74),
    "delta3_log_real_equity": (8.65, 48.80),
    "delta3_log_real_house_price": (6.47, 17.89),
}

# Table 1: crisis indicator means in percent (unconditional onset rates)
CRISIS_INDICATOR_MEANS = {
    "crisis_bvx": 3.98,
    "crisis_jst": 2.64,
    "crisis_rr": 3.61,
}

# Table 1: full published layout, used by table1_summary_stats.py to build
# the side-by-side comparison. row label -> (N, mean, sd)
TABLE1_PUBLISHED_ROWS = {
    "crisis_bvx": (1281, 3.98, 19.56),
    "crisis_jst": (909, 2.64, 16.04),
    "crisis_rr": (1109, 3.61, 18.65),
    "bank_equity_crash": (1280, 8.52, 27.92),
    "bank_failures": (1281, 3.51, 18.42),
    "banking_panic": (1281, 3.04, 17.19),
    "real_gdp_growth": (1281, 3.28, 3.21),
    "delta3_business_debt_gdp": (1258, 3.86, 20.74),
    "delta3_household_debt_gdp": (1107, 3.58, 5.74),
    "delta3_log_real_private_debt": (1281, 17.90, 16.85),
    "delta3_log_real_equity": (1258, 8.65, 48.80),
    "delta3_log_real_house_price": (1107, 6.47, 17.89),
}

# Debt quantiles for the total-debt row (Table 1 bottom panel)
PRIVATE_DEBT_LOG_QUANTILES = {0.20: 5.26, 0.40: 13.05, 0.60: 20.42, 0.80: 29.26}

N_SAMPLE_COUNTRIES = 42
