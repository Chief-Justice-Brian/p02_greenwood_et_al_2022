"""Published benchmarks for the assigned GHSS replication exhibits.

The values below are transcribed from Greenwood, Hanson, Shleifer, and
Sorensen (March 2021 HBS working paper 20-130), the version whose numbering
matches this project: Tables 3--4 on printed pages 41--44 and Figures 1 and 3
on printed pages 52--55. Table 1 values live in :mod:`paper_benchmarks`.

Tolerance policy
----------------
Exact reproduction is not expected (the paper's GFD/Bloomberg equity series
are proprietary and public databases have been revised); absolute tolerances
in the units printed by the paper are:

* Table 1: 0.75 percentage points for means, 2.5 for standard deviations,
  and 15% for observation counts.
* Table 3: 3 percentage points for cell shares and 14 percentage points for
  cell crisis probabilities/differences.
* Table 4: 8.25 percentage points for coefficients, 1.5 for t-statistics,
  2.0 for within-R2, and 15% for N.
* Figure 1: exact key BVX event years, plus count/rate tolerances for
  R-Zone markers whose assignments depend on the substituted price series.
* Figure 3: plot-derived peak and window benchmarks; percentages allow 8
  points and peak years allow one year. These validate the numerical series,
  not rendered pixels.

The rationale for each tolerance is documented in
``docs_src/project_overview/methodology.md``.
"""

TABLE1_MEAN_TOLERANCE_PP = 0.75
TABLE1_SD_TOLERANCE_PP = 2.50
TABLE1_N_TOLERANCE_FRACTION = 0.15

TABLE3_DISTRIBUTION_TOLERANCE_PP = 3.0
TABLE3_PROBABILITY_TOLERANCE_PP = 14.0
TABLE3_DISTRIBUTION_RMSE_TOLERANCE_PP = 1.0
TABLE3_PROBABILITY_RMSE_TOLERANCE_PP = 4.0

TABLE4_COEFFICIENT_TOLERANCE_PP = 8.25
TABLE4_TSTAT_TOLERANCE = 1.50
TABLE4_R2_TOLERANCE_PP = 2.0
TABLE4_N_TOLERANCE_FRACTION = 0.15
TABLE4_COEFFICIENT_RMSE_TOLERANCE_PP = 4.0
TABLE4_TSTAT_RMSE_TOLERANCE = 0.75
TABLE4_R2_RMSE_TOLERANCE_PP = 1.0

FIGURE1_RZONE_COUNT_TOLERANCE = 15
FIGURE1_FOLLOWED_COUNT_TOLERANCE = 10
FIGURE1_CRISIS_COUNT_TOLERANCE = 6
FIGURE1_PRECEDED_COUNT_TOLERANCE = 6
FIGURE1_PPV_TOLERANCE_PP = 10.0

FIGURE3_SHARE_TOLERANCE_PP = 8.0
FIGURE3_PEAK_YEAR_TOLERANCE = 1


TABLE3_DISTRIBUTIONS = {
    "business": [
        [5.6, 6.5, 5.8, 6.8, 8.7],
        [6.8, 7.6, 7.0, 6.7, 5.3],
        [7.6, 6.0, 7.2, 6.6, 6.0],
    ],
    "household": [
        [10.5, 7.5, 5.7, 5.5, 4.2],
        [6.2, 6.8, 8.1, 6.7, 5.5],
        [3.3, 5.7, 6.2, 7.8, 10.3],
    ],
}

TABLE3_FREQUENCIES = {
    "business": {
        1: [
            [1.4, 2.4, 0.0, 3.5, 6.4],
            [2.4, 3.2, 4.5, 3.6, 11.9],
            [2.1, 1.3, 2.2, 3.6, 13.3],
        ],
        2: [
            [1.4, 4.9, 2.7, 4.7, 14.7],
            [2.4, 4.2, 6.8, 7.1, 16.4],
            [8.3, 5.3, 8.9, 8.4, 26.7],
        ],
        3: [
            [4.2, 4.9, 4.1, 7.1, 19.3],
            [3.5, 5.3, 8.0, 9.5, 19.4],
            [11.5, 9.3, 11.1, 19.3, 45.3],
        ],
        4: [
            [5.6, 13.4, 4.1, 8.2, 20.2],
            [4.7, 6.3, 10.2, 17.9, 23.9],
            [12.5, 12.0, 13.3, 26.5, 48.0],
        ],
    },
    "household": {
        1: [
            [2.6, 2.4, 3.2, 3.3, 10.9],
            [2.9, 0.0, 3.3, 2.7, 1.6],
            [2.7, 3.2, 0.0, 4.7, 14.0],
        ],
        2: [
            [6.0, 3.6, 7.9, 4.9, 21.7],
            [5.8, 2.7, 3.3, 6.8, 8.2],
            [2.7, 3.2, 1.4, 10.5, 26.3],
        ],
        3: [
            [9.5, 4.8, 11.1, 8.2, 28.3],
            [7.2, 4.0, 3.3, 16.2, 13.1],
            [2.7, 3.2, 1.4, 17.4, 36.8],
        ],
        4: [
            [10.3, 8.4, 14.3, 11.5, 30.4],
            [8.7, 4.0, 6.7, 20.3, 23.0],
            [5.4, 4.8, 5.8, 20.9, 41.2],
        ],
    },
}

TABLE3_DIFFERENCES = {
    "business": {
        1: [
            [-3.1, -2.1, -4.5, -1.0, 1.9],
            [-2.2, -1.4, 0.0, -1.0, 7.4],
            [-2.5, -3.2, -2.3, -0.9, 8.8],
        ],
        2: [
            [-5.4, -1.9, -4.1, -2.1, 7.9],
            [-4.5, -2.6, 0.0, 0.3, 9.6],
            [1.5, -1.5, 2.1, 1.6, 19.8],
        ],
        3: [
            [-3.7, -3.1, -3.8, -0.9, 11.3],
            [-4.4, -2.7, 0.0, 1.6, 11.4],
            [3.5, 1.4, 3.2, 11.3, 37.4],
        ],
        4: [
            [-4.6, 3.2, -6.1, -2.0, 10.0],
            [-5.5, -3.9, 0.0, 7.6, 13.7],
            [2.3, 1.8, 3.1, 16.3, 37.8],
        ],
    },
    "household": {
        1: [
            [-0.7, -0.9, -0.2, -0.1, 7.5],
            [-0.4, -3.3, 0.0, -0.6, -1.7],
            [-0.6, -0.2, -3.3, 1.3, 10.7],
        ],
        2: [
            [2.7, 0.3, 4.6, 1.6, 18.4],
            [2.5, -0.7, 0.0, 3.4, 4.9],
            [-0.6, -0.2, -1.9, 7.1, 23.0],
        ],
        3: [
            [6.1, 1.5, 7.8, 4.9, 24.9],
            [3.9, 0.7, 0.0, 12.9, 9.8],
            [-0.6, -0.2, -1.9, 14.1, 33.5],
        ],
        4: [
            [3.7, 1.8, 7.6, 4.8, 23.8],
            [2.0, -2.7, 0.0, 13.6, 16.3],
            [-1.3, -1.9, -0.9, 14.3, 34.6],
        ],
    },
}


# Table 4 arrays are ordered by horizon, then specifications
# [high_debt_only, high_price_only, full, rzone_only].
TABLE4_COEFFICIENTS = {
    "business": {
        "high_business_debt_growth": [
            [6.9, 5.3],
            [11.6, 9.5],
            [16.8, 11.5],
            [15.6, 10.3],
        ],
        "high_business_price_growth": [
            [0.4, -0.4],
            [4.8, 3.8],
            [10.5, 7.4],
            [10.7, 7.6],
        ],
        "rzone_business": [[5.3, 9.0], [7.8, 17.9], [19.4, 33.7], [19.4, 33.0]],
    },
    "household": {
        "high_household_debt_growth": [
            [7.3, 2.4],
            [15.1, 7.3],
            [20.5, 9.1],
            [23.7, 14.2],
        ],
        "high_household_price_growth": [[3.6, 0.4], [6.0, 0.4], [8.1, 0.0], [8.5, 0.8]],
        "rzone_household": [[8.9, 11.2], [14.1, 20.5], [20.9, 28.6], [17.1, 29.6]],
    },
}

TABLE4_TSTATS = {
    "business": {
        "high_business_debt_growth": [[2.3, 2.1], [3.0, 2.5], [3.3, 2.7], [2.7, 2.2]],
        "high_business_price_growth": [[0.1, -0.2], [0.9, 0.8], [1.4, 1.1], [1.5, 1.2]],
        "rzone_business": [[0.8, 1.1], [1.3, 2.1], [2.8, 3.3], [2.6, 3.1]],
    },
    "household": {
        "high_household_debt_growth": [[2.2, 1.6], [2.8, 2.2], [3.3, 2.3], [3.9, 2.5]],
        "high_household_price_growth": [
            [1.7, 0.3],
            [1.4, 0.2],
            [1.5, 0.001],
            [1.5, 0.2],
        ],
        "rzone_household": [[1.8, 2.2], [2.4, 2.7], [3.2, 3.4], [2.0, 4.1]],
    },
}

TABLE4_COMBINED_EFFECTS = {
    "business": [
        [6.9, 0.4, 10.2, 9.0],
        [11.6, 4.8, 21.1, 17.9],
        [16.8, 10.5, 38.2, 33.7],
        [15.6, 10.7, 37.3, 33.0],
    ],
    "household": [
        [7.3, 3.6, 11.7, 11.2],
        [15.1, 6.0, 21.8, 20.5],
        [20.5, 8.1, 30.1, 28.6],
        [23.7, 8.5, 32.1, 29.6],
    ],
}

TABLE4_COMBINED_TSTATS = {
    "business": [1.2, 2.1, 3.2, 3.1],
    "household": [2.2, 2.7, 3.3, 4.0],
}

TABLE4_WITHIN_R2 = {
    "business": [
        [1.6, 0.0, 1.9, 1.1],
        [2.5, 0.7, 3.6, 2.3],
        [3.8, 2.4, 7.8, 6.1],
        [2.8, 2.1, 6.2, 4.8],
    ],
    "household": [
        [1.8, 0.7, 2.8, 2.7],
        [4.1, 1.0, 5.5, 4.9],
        [5.6, 1.4, 7.6, 7.0],
        [6.2, 1.3, 7.4, 6.2],
    ],
}

TABLE4_N = {"business": 1258, "household": 1107}


FIGURE1_AGGREGATES = {
    "business": {
        "rzone_events": 75,
        "rzone_followed_by_crisis_3y": 34,
        "crisis_count": 50,
        "crises_preceded_by_rzone_3y": 20,
        "ppv_pct": 45.3,
    },
    "household": {
        "rzone_events": 114,
        "rzone_followed_by_crisis_3y": 42,
        "crisis_count": 44,
        "crises_preceded_by_rzone_3y": 21,
        "ppv_pct": 36.8,
    },
}

# Key crisis markers visibly plotted in both panels of Figure 1. Exact because
# the paper and this project use the same BVX chronology.
FIGURE1_KEY_CRISIS_YEARS = {
    "USA": [1984, 1990, 2007],
    "GBR": [1973, 1991, 2008],
    "JPN": [1990, 1997, 2001],
    "ESP": [1975, 2008, 2010],
    "SWE": [1991, 2008],
    "NOR": [1987, 2008],
    "IRL": [2007, 2010],
    "GRC": [2008, 2010],
    "BEL": [2008, 2011],
    "ARG": [1980, 1985, 1989, 1995, 2000],
}


# Figure 3 benchmarks are numeric features read from the published annual
# series. Window maxima are more robust than individual digitized pixels.
FIGURE3_BENCHMARKS = {
    "business_peak_pct": 32.5,
    "business_peak_year": 2008,
    "household_peak_pct": 44.0,
    "household_peak_year": 1989,
    "business_1950s_max_pct": 11.1,
    "household_1970s_max_pct": 7.7,
    "household_2002_2008_max_pct": 36.4,
}
