# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Predictable Financial Crises: project tour
#
# ## Summary
#
# Greenwood, Hanson, Shleifer, and Sørensen (GHSS) study whether financial
# crises become more likely when rapid credit growth occurs alongside rapid
# asset-price appreciation. Their central indicator is the **R-Zone**: the
# intersection of debt growth in the historical top quintile and real
# asset-price growth in the historical top tercile.
#
# This notebook gives a guided tour of the cleaned 42-country panel and the
# analysis built from it. It does not replace the reproducible pipeline; the
# production scripts remain responsible for downloading, cleaning, estimating,
# testing, and typesetting the results. Instead, the notebook opens those
# artifacts and reproduces a few central calculations interactively so a reader
# can understand how the pieces fit together.
#
# ## Learning outcomes
#
# By the end of the tour, you should be able to:
#
# 1. Read and audit the cleaned country-year panel.
# 2. Understand the sample flags and source-splicing metadata.
# 3. Reconstruct a three-year growth observation and the R-Zone rule.
# 4. Interpret the historical quantile cells and fixed-effects regressions.
# 5. Distinguish the historical replication from the update through 2025.
# 6. Locate the tables, figures, tests, and final report produced by the code.
#
# ## Game plan
#
# We move in the same order as the pipeline:
#
# **cleaned panel → sample construction → variables → descriptive replication
# → regression replication → post-publication update → validation**.

# %%
from pathlib import Path

import pandas as pd
from IPython.display import Image, display

from build_analysis_panel import load_analysis_panel
from pull_imf_gdd import load_imf_gdd
from settings import config
from table1_summary_stats import calculate_cutoff_comparison, calculate_table1
from updated_crisis_chronology import add_updated_crisis_series

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
BASE_DIR = Path(config("BASE_DIR"))

pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", "{:,.2f}".format)

# %% [markdown]
# ## Step 1. Load the cleaned analysis panel
#
# The pipeline's main shareable dataset is a Parquet file keyed by ISO-3
# country code and calendar year. Each row combines crisis outcomes, cleaned
# predictors, source labels, sample flags, historical quantile assignments,
# and R-Zone indicators.

# %%
panel = load_analysis_panel(DATA_DIR)

print(f"Rows: {len(panel):,}")
print(f"Columns: {panel.shape[1]}")
print(f"Countries: {panel['country_iso3'].nunique()}")
print(f"Years: {panel['year'].min()}–{panel['year'].max()}")
print(f"Duplicate country-years: {panel.duplicated(['country_iso3', 'year']).sum()}")

# %% [markdown]
# A valid panel must have exactly one observation for each country-year key.
# The file extends beyond 2025 because source panels may contain a newer raw
# observation, but complete debt/price predictor pairs currently end in 2025.

# %%
tour_columns = [
    "country_iso3",
    "country",
    "year",
    "crisis_bvx",
    "delta3_business_debt_gdp",
    "business_debt_source",
    "delta3_log_real_equity",
    "equity_source",
    "rzone_business",
    "delta3_household_debt_gdp",
    "household_debt_source",
    "delta3_log_real_house_price",
    "house_price_source",
    "rzone_household",
]
panel.loc[panel["country_iso3"].eq("USA"), tour_columns].tail(8)

# %% [markdown]
# ## Step 2. Inspect coverage and source splicing
#
# A long international panel cannot be constructed from a single public source.
# The cleaning code calculates each three-year change *within* a source and only
# then chooses among sources. The source columns make this choice auditable.
# The priorities are:
#
# - credit: IMF GDD, then BIS, then JST;
# - equity prices: IMF, then JST, then OECD;
# - house prices: BIS, then OECD, then JST;
# - CPI and GDP: World Bank WDI, supplemented by JST.
#
# Missing observations are not filled with zero.

# %%
source_columns = [
    "business_debt_source",
    "household_debt_source",
    "equity_source",
    "house_price_source",
]
source_counts = {
    column: panel[column].value_counts(dropna=True).rename(column)
    for column in source_columns
}
pd.concat(source_counts.values(), axis=1).fillna(0).astype(int)

# %%
coverage = pd.read_csv(OUTPUT_DIR / "data_overview_summary.csv")
coverage[
    [
        "sector_label",
        "start_year",
        "end_year",
        "country_years",
        "countries",
        "coverage_pct",
        "rzone_share_pct",
    ]
]

# %% [markdown]
# Coverage rises substantially in the later period: a complete business pair
# covers about 48% of the balanced historical panel and 94% of 2013–2025; the
# corresponding household shares are about 46% and 97%. At the same time,
# R-Zones become less frequent. This is why the update must report changing
# sample counts rather than treating the panel as balanced.

# %%
display(Image(filename=OUTPUT_DIR / "data_overview_figure.png", width=900))

# %% [markdown]
# ## Step 3. Understand the historical forecasting sample
#
# GHSS compare crisis probabilities over horizons from one to four years. To
# keep the estimation sample identical across horizons, a forecast origin at
# year $t$ is historical only when the BVX chronology is observed in every year
# from $t+1$ through $t+4$. This produces a common 1950–2012 forecast-origin
# window. Business and household samples then require their respective complete
# predictor pairs.

# %%
sample_summary = pd.DataFrame(
    [
        {
            "sample": "Common paper years",
            "country_years": int(panel["in_paper_sample"].sum()),
            "countries": panel.loc[panel["in_paper_sample"], "country_iso3"].nunique(),
            "first_year": panel.loc[panel["in_paper_sample"], "year"].min(),
            "last_year": panel.loc[panel["in_paper_sample"], "year"].max(),
        },
        {
            "sample": "Business estimation",
            "country_years": int(panel["in_business_sample"].sum()),
            "countries": panel.loc[panel["in_business_sample"], "country_iso3"].nunique(),
            "first_year": panel.loc[panel["in_business_sample"], "year"].min(),
            "last_year": panel.loc[panel["in_business_sample"], "year"].max(),
        },
        {
            "sample": "Household estimation",
            "country_years": int(panel["in_household_sample"].sum()),
            "countries": panel.loc[panel["in_household_sample"], "country_iso3"].nunique(),
            "first_year": panel.loc[panel["in_household_sample"], "year"].min(),
            "last_year": panel.loc[panel["in_household_sample"], "year"].max(),
        },
    ]
)
display(sample_summary)

# %%
outcome_columns = [f"crisis_next_{horizon}y" for horizon in range(1, 5)]
panel.loc[panel["in_paper_sample"], outcome_columns].isna().sum().rename(
    "missing_in_common_sample"
)

# %% [markdown]
# Every future outcome is defined in the common sample. This check prevents a
# missing future crisis year from being silently interpreted as “no crisis.”

# %% [markdown]
# ## Step 4. Reconstruct the core variables
#
# Debt growth is the three-year percentage-point change in debt/GDP. Real
# asset-price growth is 100 times a three-year log change. The tidy scripts make
# those changes within each source so a calculation never jumps between two
# incompatible index levels.
#
# The small example below verifies one business-debt observation using the
# underlying GDD levels. This is illustrative; the production transformation
# and its edge cases live in the cleaning modules and unit tests.

# %%
gdd = load_imf_gdd(DATA_DIR)
gdd.head()

# %%
usa_business_debt = (
    gdd.query("country_iso3 == 'USA' and indicator == 'NFC_LS'")
    .set_index("year")["value"]
    .sort_index()
)
manual_2020_change = usa_business_debt.loc[2020] - usa_business_debt.loc[2017]
cleaned_2020_change = panel.loc[
    panel["country_iso3"].eq("USA") & panel["year"].eq(2020),
    "delta3_business_debt_gdp",
].iloc[0]

pd.Series(
    {
        "2017 GDD business debt/GDP": usa_business_debt.loc[2017],
        "2020 GDD business debt/GDP": usa_business_debt.loc[2020],
        "manual 2020 three-year change": manual_2020_change,
        "cleaned-panel 2020 change": cleaned_2020_change,
    }
).to_frame("value")

# %% [markdown]
# The cleaned panel stores the finished source-consistent changes. We can verify
# their time arithmetic directly: a three-year change must be missing when the
# same series is not available exactly three years earlier within its source.
# More importantly for interpretation, the high-growth gates are pooled
# historical thresholds—not country-specific percentiles.

# %%
cutoffs = calculate_cutoff_comparison(panel)
cutoffs[["label", "quantile_label", "replicated", "paper", "difference"]]

# %%
frozen_cutoffs = pd.Series(
    {
        "Business debt Q80": panel["business_debt_q80_cutoff"].dropna().iloc[0],
        "Business price T66.7": panel["business_price_t667_cutoff"].dropna().iloc[0],
        "Household debt Q80": panel["household_debt_q80_cutoff"].dropna().iloc[0],
        "Household price T66.7": panel["household_price_t667_cutoff"].dropna().iloc[0],
    },
    name="frozen historical cutoff",
)
frozen_cutoffs.to_frame()

# %% [markdown]
# An R-Zone equals one only when **both** high-growth indicators equal one.
# Here is a direct consistency check across every classified row.

# %%
for sector in ["business", "household"]:
    expected = (
        panel[f"high_{sector}_debt_growth"]
        * panel[f"high_{sector}_price_growth"]
    )
    classified = panel[f"rzone_{sector}"].notna()
    mismatches = (
        panel.loc[classified, f"rzone_{sector}"] != expected.loc[classified]
    ).sum()
    print(f"{sector.title()} R-Zone assignment mismatches: {mismatches}")

# %% [markdown]
# ## Step 5. Reproduce the historical descriptive analysis
#
# Table 1 checks whether the reconstructed moments and thresholds resemble the
# paper before any regressions are interpreted.

# %%
table1 = calculate_table1(panel)
table1[
    [
        "label",
        "replicated_n",
        "replicated_mean",
        "replicated_sd",
        "paper_n",
        "paper_mean",
        "paper_sd",
    ]
]

# %% [markdown]
# The largest economically meaningful discrepancy is equity-price dispersion.
# The paper primarily uses proprietary Global Financial Data and Bloomberg
# series, while this public reconstruction substitutes IMF, JST, and OECD data.
# The project reports this difference rather than adjusting observations to
# force agreement.

# %%
rzone_validation = pd.read_csv(OUTPUT_DIR / "rzone_validation.csv")
display(rzone_validation)

# %% [markdown]
# The reconstructed R-Zone frequencies are close to the paper: 6.46% versus
# 6.0% for business and 9.81% versus 10.3% for households. Three-year crisis
# risk remains strongly elevated in the R-Zone, although the business frequency
# is below the published value because some equity histories differ.

# %%
table3 = pd.read_csv(OUTPUT_DIR / "table3_crisis_probabilities.csv")
business_h3 = table3.query("sector == 'business' and horizon == 3")
business_h3.pivot(
    index="price_tercile",
    columns="debt_quintile",
    values="crisis_frequency_pct",
).style.format("{:.1f}").set_caption(
    "Business-sector probability of a crisis within three years (%)"
)

# %% [markdown]
# Read the matrix from lower price/debt ranks toward the upper-right R-Zone
# cell. Crisis probabilities are not perfectly monotonic in every sparse cell,
# but the joint upper tail is clearly riskier than the median cell.

# %%
display(Image(filename=OUTPUT_DIR / "figure1_event_history.png", width=900))

# %% [markdown]
# Figure 1 makes the limits of the indicator visible. Teal crosses often cluster
# near coral crisis onsets, but an R-Zone is neither necessary nor sufficient:
# some R-Zones do not lead to a crisis, and some crises have no preceding
# R-Zone.

# %% [markdown]
# ## Step 6. Read the fixed-effects regression output
#
# Table 4 estimates country-fixed-effects linear probability models. The full
# specification includes high debt growth, high price growth, and their R-Zone
# interaction. Coefficients are reported in percentage points, and inference
# uses horizon-specific Driscoll–Kraay covariance estimates.

# %%
coefficients = pd.read_csv(OUTPUT_DIR / "table4_baseline_coefficients.csv")
models = pd.read_csv(OUTPUT_DIR / "table4_baseline_models.csv")

full_h3 = coefficients.query("horizon == 3 and specification == 'full'")
full_h3[
    ["sector", "variable", "coefficient_pp", "std_error_pp", "t_stat", "p_value"]
]

# %%
models.query("horizon == 3")[
    [
        "sector",
        "specification",
        "n",
        "within_r2_pct",
        "combined_effect_pp",
        "combined_t_stat",
    ]
]

# %% [markdown]
# The combined effect is the model-implied change when every included indicator
# equals one. The R-Zone coefficients are economically large at the three-year
# horizon, but exact equality with the paper is not expected because individual
# R-Zone assignments depend on the substituted equity histories.

# %% [markdown]
# ## Step 7. Tour the post-publication update
#
# The update preserves the historical gates. It appends documented crisis
# coverage through 2025 and creates a different valid final forecast origin for
# each horizon: 2024 for one year, 2023 for two years, 2022 for three years, and
# 2021 for four years.

# %%
updated_panel = add_updated_crisis_series(panel)
updated_coverage = pd.read_csv(OUTPUT_DIR / "updated_data_coverage.csv")
display(updated_coverage)

# %%
tracker = pd.read_csv(OUTPUT_DIR / "post_2016_rzone_tracker.csv")
recent_tracker = (
    tracker.groupby("year")
    .agg(
        countries=("country_iso3", "nunique"),
        business_rzones=("rzone_business", "sum"),
        household_rzones=("rzone_household", "sum"),
        either_rzones=("rzone_either", "sum"),
    )
    .astype(int)
)
display(recent_tracker)

# %% [markdown]
# The later panel has broad coverage but relatively few R-Zones. No country is
# classified in either sector in 2024 or 2025. This is a descriptive statement,
# not proof that crisis risk disappeared.

# %%
updated_models = pd.read_csv(OUTPUT_DIR / "table4_updated_models.csv")
updated_models.groupby(["sector", "horizon"]).agg(
    forecast_end_year=("forecast_end_year", "first"),
    observations=("n", "first"),
)

# %%
display(
    Image(
        filename=OUTPUT_DIR / "figure3_fraction_countries_rzone_updated.png",
        width=850,
    )
)

# %% [markdown]
# The updated global series retains the major historical waves but shows low
# recent R-Zone shares. Because the newer IMF chronology contains no confirmed
# post-2016 systemic onset inside these 42 countries, the update adds evidence
# about the prevalence of warning states but is not yet a powerful realized
# out-of-sample crisis test.

# %% [markdown]
# ## Step 8. See what the extensions add
#
# The project also compares the GHSS model with bank-fragility and
# autoregressive extensions. These are new analyses, not paper replications.

# %%
dynamic_models = pd.read_csv(OUTPUT_DIR / "dynamic_regression_models.csv")
dynamic_models.pivot_table(
    index=["sector", "horizon"],
    columns="specification",
    values=["within_r2_pct", "in_sample_auc"],
).round(3)

# %% [markdown]
# Adding crisis memory and lagged predictors improves within fit and in-sample
# AUC at every horizon. The AUC calculation is in-sample, so it demonstrates
# incremental fit rather than a definitive out-of-sample forecast gain.

# %%
missed = pd.read_csv(OUTPUT_DIR / "missed_crisis_fragility.csv")
pd.Series(
    {
        "Historical crisis onsets examined": len(missed),
        "Missed by both R-Zones": int(missed["missed_by_rzone"].sum()),
        "Misses with fragility data": int(
            missed.loc[missed["missed_by_rzone"], "max_prior_noncore_percentile"]
            .notna()
            .sum()
        ),
    }
).to_frame("count")

# %% [markdown]
# Bank fragility explains a subset of crises missed by borrower-side R-Zones,
# but not all of them. This motivates treating balance-sheet variables as a
# complement rather than a replacement.

# %% [markdown]
# ## Step 9. Check replication tolerances and locate the final artifacts
#
# Published values are stored separately from the estimators. The validation
# report compares generated results with 586 transcribed benchmarks using
# documented absolute tolerances.

# %%
validation = pd.read_csv(OUTPUT_DIR / "replication_validation.csv")
validation.groupby("exhibit")["within_tolerance"].agg(
    comparisons="size", passed="sum"
).assign(pass_rate=lambda frame: frame["passed"] / frame["comparisons"])

# %%
failed = validation.loc[~validation["within_tolerance"]]
print(f"Benchmarks outside tolerance: {len(failed)}")

# %% [markdown]
# ## Where to go next
#
# The primary reader-facing outputs are:
#
# - `reports/final_report.pdf`: the complete narrative LaTeX report;
# - `_output/figure1_event_history_updated.pdf`: the updated event histories;
# - `reports/table3_updated_preview.pdf` and `table4_updated_preview.pdf`:
#   detailed updated tables;
# - `_output/post_2016_rzone_tracker.csv`: country-year monitoring data;
# - `_output/replication_validation.csv`: every benchmark, tolerance, and result.
#
# The central lesson from the tour is that the replication recovers the paper's
# qualitative joint-boom result while making the data limitations explicit.
# Coverage improves considerably after the historical period, recent R-Zones
# are uncommon, bank fragility and memory add some information, and the absence
# of new in-sample systemic onsets makes strong modern validation premature.

# %%
artifacts = [
    BASE_DIR / "reports/final_report.pdf",
    OUTPUT_DIR / "figure1_event_history_updated.pdf",
    BASE_DIR / "reports/table3_updated_preview.pdf",
    BASE_DIR / "reports/table4_updated_preview.pdf",
    OUTPUT_DIR / "post_2016_rzone_tracker.csv",
    OUTPUT_DIR / "replication_validation.csv",
]
pd.DataFrame(
    {
        "artifact": [str(path) for path in artifacts],
        "exists": [path.exists() for path in artifacts],
    }
)
