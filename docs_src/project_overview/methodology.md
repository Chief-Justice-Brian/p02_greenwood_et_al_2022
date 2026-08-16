# Methodology

## Pipeline

The repository separates acquisition, cleaning, feature construction, and
analysis. `doit` records the file dependencies between these stages:

1. `pull` downloads each public source into `_data/` without changing it.
2. `tidy` creates source-consistent annual changes and the unified panel.
3. `analysis` writes replication, extension, and diagnostic artifacts to
   `_output/`.

`doit analysis:post_publication` is a separate post-publication calculation. It never
overwrites the historical replication artifacts.

The final shareable dataset is `_data/rzone_analysis_panel.parquet`. Its key is
`country_iso3, year`; duplicate keys are rejected during construction.

## Historical sample

The country universe is the 42-country GHSS sample in `country_sample.py`.
Following the paper, the common forecasting sample contains years 1950–2012
for which:

- business credit growth and real equity growth are both observed, or
  household credit growth and real house-price growth are both observed; and
- the BVX crisis-onset indicator is defined in every year from `t+1` through
  `t+4`.

Stopping forecast origins in 2012 preserves the same sample at horizons one
through four. Rows from 2013–2016 remain in the panel as historical outcomes,
and rows after 2016 are marked `is_post_2016`.

## Splicing and missing data

The central splicing rule comes from GHSS footnote 5: calculate a three-year
change separately inside each source and splice the resulting changes. Never
splice differently based index levels first. A change at year `t` requires
observations from the same source at `t` and exactly `t-3`.

Source priorities are:

- credit: IMF GDD, then BIS, then JST;
- equity: IMF, then JST, then OECD;
- house prices: BIS, then OECD, then JST;
- CPI and nominal GDP: WDI, supplemented by rebased JST levels.

Missing crisis observations are not converted to zero outside a chronology's
documented coverage. Similarly, an unavailable future crisis window remains
missing instead of being interpreted as “no crisis.” Every spliced predictor
retains a source column.

## Variables

Debt-to-GDP growth is the three-year percentage-point change. Real asset-price
growth is 100 times the three-year log difference. The forecast outcome for
horizon `h` is one if a BVX crisis begins in any year from `t+1` through `t+h`.

Debt quintiles and price terciles use the pooled historical forecasting sample,
not country-specific distributions. High debt growth is above Q80; high price
growth is above T66.7; their product is the sector R-zone. These historical
cutoffs remain fixed when classifying post-2016 observations.

## Estimation

The baseline linear probability models include country fixed effects and use
Driscoll–Kraay covariance estimates with bandwidths 0, 3, 5, and 6 at horizons
one through four. The implementation reports conventional Driscoll–Kraay
p-values; it does not yet implement the paper's additional Kiefer–Vogelsang
finite-sample p-value correction.

Fragility extensions use JST `noncore`, `ltd`, and `lev` separately, with Q80
high-fragility indicators. Each extended model is compared with the GHSS model
on the exact same complete-case sample. The dynamic comparison adds current
and lagged crisis outcomes plus lagged GHSS predictors. Reported AUC values are
in-sample diagnostics, not claims of out-of-sample performance.

## Post-publication exhibit update

Updated Tables 1, 3, and 4 and Figures 1 and 3 use predictor pairs through
2025. R-Zone indicators continue to use the pooled historical Q80 debt and
T66.7 price thresholds; the update does not re-estimate the definition using
later observations. Updated Table 1 nevertheless reports expanded-sample
quantiles descriptively and stores the frozen assignment threshold beside each
relevant gate in `table1_post_publication_quantiles.csv`.

The updated systemic-crisis onset variable combines BVX through 2016 with the
Laeven--Valencia (2026) IMF chronology from 2017 through 2025. The confirmed
post-2016 onsets in that source are Ghana (2017), Republic of Congo (2017), and
Lebanon (2019), none of which is in the 42-country GHSS sample. Events marked
borderline by the IMF are excluded. Thus zeros in 2017--2025 are documented
non-onsets for this sample, not missing observations recoded as zeros.

The end of a predictor series is not necessarily a valid forecast origin. A
horizon-$h$ outcome at year $t$ requires crisis coverage in every year from
$t+1$ through $t+h$. With crisis coverage through 2025, the final usable
forecast origins are 2024, 2023, 2022, and 2021 at horizons one through four.
These endpoints and the final usable year of every input are written to
`post_publication_data_coverage.csv` and repeated in the updated regression CSVs.

## Replication validation

Published values for every numeric cell in assigned Tables 1, 3, and 4 are
stored separately from the estimators in `paper_benchmarks.py` and
`exhibit_benchmarks.py`. Tests compare generated CSV results with those values
using documented absolute tolerances. Figure validation operates on the data
behind the graphics: Figure 1 checks exact key BVX event dates and published
R-Zone event counts/rates, while Figure 3 checks historical coverage, peaks,
peak timing, and window maxima from the published annual series. It does not
compare image pixels.

Tolerances are tighter for summary statistics and model fit, and wider for
sparse probability cells and interaction coefficients that are especially
sensitive to the unavailable GFD/Bloomberg equity series. The full audit is
written to `_output/replication_validation.csv`; any value outside its stated
tolerance fails `doit analysis:validation` and the exhibit-level pytest suite.

## Known replication differences

The project cannot use the paper's paid Global Financial Data and Bloomberg
equity series. IMF/JST/OECD substitutions reproduce the historical percentile
thresholds closely, but they change some R-zone assignments and therefore some
conditional crisis frequencies and regression coefficients. Those differences
are reported in `_output/rzone_validation.csv` and are not tuned away.
