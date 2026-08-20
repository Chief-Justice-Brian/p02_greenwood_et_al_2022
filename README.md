# Predictable Financial Crises

FINM 32900 Final Project · Brian Nguyen & Clara Duan · Summer 2026

Replication and extension of Greenwood, Hanson, Shleifer & Sørensen, *"Predictable Financial Crises,"* Journal of Finance 77(2), 2022 (working paper: HBS 20-130).

The paper finds that when 3-year credit growth and 3-year asset-price growth are jointly elevated (the **R-zone**), the probability of a financial crisis within three years rises from roughly 7% to 40%.

This project:

1. **Replicates** Tables 1, 3, 4 and Figures 1, 3 for 42 countries over 1950–2016 using free public data.
2. **Extends** the R-zone and published exhibits through 2025 and produces a country-by-year R-zone tracker.
3. **Tests** the post-publication period, including the 2020–2022 warning signal and the paper's dynamic local-projection specification.

## Team Responsibilities

Both members committed throughout and integrated work through reviewed GitHub
pull requests.

### Brian Nguyen

* **Data engineering:** Built the data-pull pipeline and core data infrastructure.
* **Automation:** Developed the PyDoit workflow and project configuration.
* **Forward extension:** Extended the analysis beyond the paper's sample period, including the R-zone tracker.
* **Crisis extension:** Implemented the post-2016 crisis chronology using the BVX criteria and US bank-equity data.
* **Validation:** Developed the Table 1 replication validation tests.
* **Results:** Contributed the bulk of the fragility and dynamics extension analysis.

### Clara Duan

* **Data cleaning:** Built the cleaning pipeline for credit, equity, house-price, macro, and crisis data.
* **Replication exhibits:** Implemented Tables 1, 3, and 4 and Figures 1 and 3.
* **Regression analysis:** Developed the baseline fixed-effects regressions.
* **Validation:** Built the replication-tolerance audit and exhibit benchmarks.
* **Post-publication analysis:** Implemented the updated replication exhibits.
* **Documentation:** Developed the data overview and project-tour notebook.

### Both

* **Panel construction:** Built the final analysis panel and R-zone features.
* **Reporting:** Produced the final results and LaTeX report.
* **Testing:** Developed and maintained the shared test suite.

## Quick Start

Create the conda environment and run the full pipeline:

```bash
conda env create -f environment.yml
conda activate p02_greenwood_et_al_2022
doit
```

No API keys or credentials are required. Optional settings can be supplied through `.env`; see `.env.example`.

Run the test suite separately:

```bash
pytest
```

## Main Outputs

The pipeline produces:

* `_data/rzone_analysis_panel.parquet`: unified historical and post-2016 analysis panel.
* `_output/table1_stats.csv` and `_output/table1_cutoffs.csv`: Table 1 replication checks.
* `_output/table3_crisis_probabilities.csv`: Table 3 crisis probabilities.
* `_output/figure1_event_history.pdf`: Figure 1 event-history panels.
* `_output/table4_baseline_*.csv`: Table 4 baseline regressions.
* `_output/post_2016_rzone_tracker.csv`: post-2016 R-zone classifications using frozen historical thresholds.
* `_output/fragility_*.csv`, `_output/dynamic_*.csv`, `_output/missed_crisis_fragility.csv`: extension results.
* `_output/replication_validation.csv`: published benchmarks, replicated values, tolerances, and pass/fail results.

The historical replication and post-publication update are kept separate.

## Post-Publication Update

Build the updated exhibits with:

```bash
doit analysis:post_publication
```

Predictors and R-zone classifications extend through 2025, with historical thresholds frozen. Crisis outcomes use the published BVX criteria through 2016 and the same criteria applied to new public data for 2017–2025, so "crisis" means the same thing in every year of the panel. The candidate post-2016 episodes and their verdicts, including the documented 2023 US borderline call, are written to `_output/post_publication_crisis_screen.csv`.

Build the original data overview with:

```bash
doit analysis:data_overview compile_latex:data_overview
open reports/data_overview.pdf
```

Build the full report with:

```bash
doit analysis:final_report compile_latex:final_report
open reports/final_report.pdf
```

The final report combines the historical replication, post-publication update, original data overview, and extensions.

## Data

The project uses the paper's data sources wherever possible:

| Source                          | Role                                                       |
| ------------------------------- | ---------------------------------------------------------- |
| Baron–Verner–Xiong (2021)       | Crisis chronology through 2016                             |
| IMF Global Debt Database        | Credit                                                     |
| BIS Total Credit Statistics     | Credit supplement                                          |
| IMF share-price indices         | Equity                                                     |
| JST Macrohistory                | Credit, equity, house prices, inflation, GDP, crisis dates |
| BIS Residential Property Prices | House prices                                               |
| OECD Analytical House Prices    | House-price supplement                                     |
| World Bank WDI                  | Inflation and GDP                                          |
| OECD share prices               | Post-2016 equity extension                                 |
| CRSP US Banks portfolio         | Post-2016 bank-equity criterion                            |

The paper's primary equity sources, GFD and Bloomberg, are paywalled. We therefore use the paper's stated fallback sources (IMF/IFS and JST) for the historical equity series and add OECD share prices for the forward extension.

## Validation and Documentation

Run the paper-benchmark validation gate with:

```bash
doit analysis:validation
pytest -q src/test_paper_exhibits.py
```

See `docs_src/project_overview/methodology.md` for sample construction, data splicing, and missing-data rules.

An executable project-tour notebook is also provided:

```bash
doit run_notebooks:01_predictable_financial_crises_project_tour.ipynb.py
open _output/01_predictable_financial_crises_project_tour.html
```

## Formatting

The project uses Ruff:

```bash
ruff format . && ruff check --select I --fix . && ruff check --fix .
```
