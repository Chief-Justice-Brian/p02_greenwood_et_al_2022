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

Fragility extensions use JST `noncore`, `ltd`, and `lev` separately. Noncore
and loans-to-deposits flag values above their in-sample Q80; `lev` is a capital
ratio, so its flag marks thin capital below the Q20. The noncore cutoff is also
swept across candidate quantiles with Q80 frozen ex ante as the baseline. Each
extended model is compared with the GHSS model on the exact same complete-case
sample. The dynamic comparison adds current
and lagged crisis outcomes plus lagged GHSS predictors. Reported AUC values are
in-sample diagnostics, not claims of out-of-sample performance.

## Post-publication exhibit update

Updated Tables 1, 3, and 4 and Figures 1 and 3 use predictor pairs through
2025. R-Zone indicators continue to use the pooled historical Q80 debt and
T66.7 price thresholds; the update does not re-estimate the definition using
later observations. Updated Table 1 nevertheless reports expanded-sample
quantiles descriptively and stores the frozen assignment threshold beside each
relevant gate in `table1_post_publication_quantiles.csv`.

The post-publication onset variable `crisis_bvx_extended` combines BVX
through 2016 with our extension of BVX's stated criteria (a bank equity
index decline of 30% or more plus narrative evidence of widespread bank
failures or a banking panic) for 2017--2025, so the outcome is judged by one
rule in every year of the panel. A narrative screen of the sample yields
three candidate episodes: the United States 2023 (equity criterion computed
from pulled data), Switzerland 2023 (a single institution; fails the
widespread-failures arm), and Russia 2022 (excluded under BVX's own war
convention). Zeros elsewhere in 2017--2025 reflect the absence of any
narrative candidate, not missing observations recoded as zeros.

### Bank equity index selection

The equity criterion uses the CRSP value-weighted Banks portfolio from the
Ken French Data Library. Validated against the paper's own US bank equity
series (`Rtot_real` in the BVX replication kit), its post-war annual real
returns correlate at roughly 0.87 with crisis-year gaps of about three
percentage points, so it functions as a continuation of the paper's series
rather than a substitute. The large-cap KBW benchmark was considered and
rejected: it is tilted toward exactly the regional banks that crashed in
2023 and diverges from BVX's broad value-weighted construction. The choice
is decisive for the 2023 call: the broad value-weighted index's 2023
trailing-peak drawdown stops short of the 30% bar that large-cap benchmarks
crossed, while BVX's own chronology contains a precedent in the other
direction (the US in 1984 is counted as a crisis on narrative evidence with
a smaller aggregate equity decline). The pipeline applies the mechanical
rule to the paper-faithful index and the report presents 2023 as a
documented borderline call.

The end of a predictor series is not necessarily a valid forecast origin. A
horizon-$h$ outcome at year $t$ requires crisis coverage in every year from
$t+1$ through $t+h$. With crisis coverage through 2025, the final usable
forecast origins are 2024, 2023, 2022, and 2021 at horizons one through four.
These endpoints and the final usable year of every input are written to
`post_publication_data_coverage.csv` and repeated in the updated regression CSVs.

## Replication validation

Published values for every numeric cell in assigned Tables 1, 3, and 4 are
stored separately from the estimators in `paper_benchmarks.py` and
`exhibit_benchmarks.py`. The validation bounds are calculated from the paper,
not from the reconstruction errors:

- Table 1 mean and standard-deviation bounds equal 10% of the row's published
  standard deviation. Quantile bounds equal 10% of the published central
  quantile span, and observation counts allow 15% of published N.
- Table 3 distribution and crisis-frequency bounds are 95% Wilson half-widths
  implied by the published sector N, cell share, and probability. A
  difference-from-median bound combines the two relevant half-widths in
  quadrature.
- Table 4 coefficient bounds equal 1.5 standard errors, with standard errors
  inferred from the published coefficient and t-statistic. T-statistics and
  within-R2 use 25% relative bounds with a one-decimal reporting floor, and N
  allows 15%.
- Figure 1 allows 15% of each published event count and uses a Wilson bound for
  positive predictive value. Key BVX crisis dates remain exact.
- Figure 3 allows three country classifications out of the paper's 42-country
  sample and one year for peak timing. Coverage endpoints remain exact.

Figure validation operates on the data behind the graphics, not rendered
pixels. Normalized RMSE checks summarize each family of cell-specific bounds.
The resulting audit passes 545 of 586 comparisons. Table 1 and Figure 3 pass
completely; Figure 1 passes 46 of 47, Table 3 passes 264 of 273, and Table 4
passes 174 of 205. The failures are retained in
`_output/replication_validation.csv`: known exhibit-level misses are marked as
strict expected failures in pytest, while the validation task continues so the
report can present the audit rather than suppressing downstream outputs.

## Known replication differences

The project cannot use the paper's paid Global Financial Data and Bloomberg
equity series. IMF/JST/OECD substitutions reproduce the historical percentile
thresholds closely, but they change some R-zone assignments and therefore some
conditional crisis frequencies and regression coefficients. Those differences
are reported in `_output/rzone_validation.csv` and are not tuned away.
