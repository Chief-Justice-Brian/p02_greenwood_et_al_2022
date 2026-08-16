# Predictable Financial Crises

FINM 32900 Final Project · Brian Nguyen & Clara Duan

Replication and extension of **Greenwood, Hanson, Shleifer & Sørensen,
"Predictable Financial Crises," Journal of Finance 77(2), 2022** (working paper
version: HBS 20-130, which sets our table/figure numbering).

The paper shows that when 3-year credit growth and 3-year asset price growth
are *jointly* elevated (the "R-zone"), the probability of a financial crisis
within 3 years rises from ~7% to ~40%. We attempt to:

1. **Replicate** the assigned exhibits (Tables 1, 3, 4; Figures 1, 3) on a
   panel of 42 countries, 1950–2016, built entirely from free public sources.
2. **Extend** every exhibit through the present - create a re-runnable
   product, a **R-Zone** tracker, plus a new country-by-year classification
   table (2017–present, with distance to the threshold(s)) and a US timeline figure.
3. **Examine** the new out of sample data: was the warning light on in
   2020–2022, before the March 2023 banking stress? And does the dynamic
   (local projection) specification the paper asserts in footnote 10 beat the
   static baseline on AUC (memory specific model).

## Quick Start

Create and activate the conda environment, then let `doit` run everything:

```bash
conda env create -f environment.yml
conda activate p02_greenwood_et_al_2022
doit
```

No API keys or credentials are needed: every data source is free and pulled by
script (see below). An optional `.env` can override defaults — see
`.env.example`.

Run the unit tests alone with:

```bash
pytest
```

The main generated artifacts are:

- `_data/rzone_analysis_panel.parquet`: unified 42-country historical and
  post-2016 panel;
- `_output/table1_stats.csv` and `_output/table1_cutoffs.csv`: replication
  validation;
- `_output/table3_crisis_probabilities.csv`: Table 3 quantile cells, crisis
  frequencies, and differences from the median cell;
- `_output/figure1_event_history.pdf`: combined business and household
  historical R-zone event-history panels;
- `_output/table4_baseline_*.csv`: country-fixed-effects baseline models;
- `_output/post_2016_rzone_tracker.csv`: classifications using frozen
  historical cutoffs;
- `_output/fragility_*.csv`, `_output/dynamic_*.csv`, and
  `_output/missed_crisis_fragility.csv`: extension results.
- `_output/replication_validation.csv`: every published benchmark, replicated
  value, documented tolerance, numerical gap, and pass/fail result for Tables
  1, 3, and 4 and Figures 1 and 3.

The historical replication is kept separate from the post-publication update.
Build the post-publication exhibits with:

```bash
doit analysis:post_publication
doit compile_latex:table1_post_publication_preview \
     compile_latex:table3_post_publication_preview \
     compile_latex:table4_post_publication_preview \
     compile_latex:figure3_post_publication_preview
```

The post_publication task writes separately labelled Table 1, Table 3, Table 4,
Figure 1, and Figure 3 files (names containing `post_publication`) plus
`_output/post_publication_data_coverage.csv`. The `post_publication_*` files are
the rubric's "reproduce with updated numbers" deliverable: the same exhibits
recomputed with the data that arrived after the paper's sample ends. They
complement the historical replication files and do not supersede them.
Predictor and R-Zone series extend through
2025. Crisis outcomes use BVX through 2016 and Laeven--Valencia (2026) through
2025, so the last valid forecast origins are 2024, 2023, 2022, and 2021 for
horizons one through four. Historical R-Zone thresholds remain frozen.

The required original data-understanding exhibit is separate from every paper
replication. Build and open its captioned LaTeX report with:

```bash
doit analysis:data_overview compile_latex:data_overview
open reports/data_overview.pdf
```

It contains an original summary-statistics table comparing 1950–2012 with
2013–2025 and an original three-panel chart of predictor coverage and the joint
growth distributions behind the R-Zone classification.

Build the single narrative report containing the historical replication,
post-publication update, original data overview, and extensions with:

```bash
doit analysis:final_report compile_latex:final_report
open reports/final_report.pdf
```

The report contains no code listings. Its text explains the project's nature,
data sources, methods, successful replication results, remaining discrepancies,
modern-update limitations, and the interpretation of every included table and
figure.

An executable Jupyter notebook provides an HW-guide-style tour of the cleaned
panel and analysis. Build and open it with:

```bash
doit run_notebooks:01_predictable_financial_crises_project_tour.ipynb.py
open _output/01_predictable_financial_crises_project_tour.html
```

The notebook covers panel keys and coverage, source splicing, sample flags,
three-year growth, frozen quantile thresholds, R-Zone assignment, descriptive
cells, fixed-effects regressions, the update through 2025, extensions, and the
replication-tolerance audit.

Run the paper-benchmark validation gate directly with:

```bash
doit analysis:validation
pytest -q src/test_paper_exhibits.py
```

See `docs_src/project_overview/methodology.md` for the sample, splicing, and
missing-data rules.

## Data Sources

With one exception, every source in our pipeline is one the paper itself
used. In the authors' own words (pp. 7–8):

> "The International Monetary Fund's (IMF) Global Debt Database (Mbaye,
> Moreno-Badia, and Chae 2018) provides data on total credit outstanding —
> including both loans and debt securities — to nonfinancial businesses and
> households. [...] We supplement the IMF credit data using information from
> the JST (2017, 2019) MacroHistory database [...] We collect credit data for
> Thailand from the Bank of International Settlements' (BIS) Total Credit
> Statistics [...]
>
> Data on equity price indices are primarily from Global Financial Data
> (GFD). **Where suitable data is not available from GFD, we obtain equity
> price data from the IMF's International Financial Statistics database or
> the JST MacroHistory database** as augmented by Jordà et al. (2019). Using
> data on nominal price inflation from the World Bank's World Development
> Indicators and the MacroHistory database, we compute the inflation-adjusted
> change in equity prices. We obtain inflation-adjusted home price indices
> from the BIS Residential Property Price database [...] We again supplement
> the BIS data on real home prices with data from the JST MacroHistory
> database and the OECD's Housing Prices database."

Everything named there is free **except** GFD (a paid subscription) and Bloomberg.
For equities we therefore do exactly what the bolded sentence above
prescribes when GFD is unavailable — IFS + JST become our primary — and keep
every other source as the paper used it. The result is, to our knowledge,
the first end-to-end reconstruction of the paper that runs entirely on free
public sources.

The one source the paper never used is **OECD share prices**, which we add
for the post-2016 extension: no forward extension is possible with the
paper's own equity sources, since GFD and Bloomberg are paywalled and the
IMF stopped collecting share prices in 2017.

| Source | Role in the paper | Role here | Pull script |
|---|---|---|---|
| Baron–Verner–Xiong (2021), Harvard Dataverse | Baseline crisis chronology | Same | `src/pull_bvx_crises.py` |
| Laeven–Valencia (2026), IMF WP 26/94 | — | Systemic-crisis chronology update, 2017–2025 | encoded and documented in `src/post_publication_crisis_chronology.py` |
| IMF Global Debt Database | Credit: **primary** | Same | `src/pull_imf_gdd.py` |
| BIS Total Credit Statistics | Credit: used for Thailand | Credit supplement; carries the monitor forward | `src/pull_bis_total_credit.py` |
| IMF share price indices (former IFS) | Equity: their stated GFD fallback | Equity **primary**, 1950–2016 | `src/pull_imf_equity.py` |
| JST Macrohistory (R6) | Supplement for credit, equity, house prices, CPI/GDP; JST crisis dates | Same, early-sample fill | `src/pull_jst_macrohistory.py` |
| OECD share prices | — (our one addition) | Equity 2017–present + countries GFD-only in the paper | `src/pull_oecd_share_prices.py` |
| BIS Residential Property Prices | House prices: **primary** | Same | `src/pull_bis_property_prices.py` |
| OECD Analytical House Prices | House prices: supplement (their footnote 7) | Same | `src/pull_oecd_house_prices.py` |
| World Bank WDI | Inflation + GDP | Same | `src/pull_worldbank_wdi.py` |

Since our non-equity data sources are identical to the original paper (vintages aside), substitution risk is isolated to the equity column. This explains why our only deviation from Table 1 occurs in equity standard deviation (46.8 vs. 48.8)—an expected result given our reliance on free alternatives to the paper's paid equity sources.

## Formatting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and
formatting:

```bash
ruff format . && ruff check --select I --fix . && ruff check --fix .
```
