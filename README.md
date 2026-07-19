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
| IMF Global Debt Database | Credit: **primary** | Same | `src/pull_imf_gdd.py` |
| BIS Total Credit Statistics | Credit: used for Thailand | Credit supplement; carries the monitor forward | `src/pull_bis_total_credit.py` |
| IMF share price indices (former IFS) | Equity: their stated GFD fallback | Equity **primary**, 1950–2016 | `src/pull_imf_equity.py` |
| JST Macrohistory (R6) | Supplement for credit, equity, house prices, CPI/GDP; JST crisis dates | Same, early-sample fill | `src/pull_jst_macrohistory.py` |
| OECD share prices | — (our one addition) | Equity 2017–present + countries GFD-only in the paper | `src/pull_oecd_share_prices.py` |
| BIS Residential Property Prices | House prices: **primary** | Same | `src/pull_bis_property_prices.py` |
| OECD Analytical House Prices | House prices: supplement (their footnote 7) | Same | `src/pull_oecd_house_prices.py` |
| World Bank WDI | Inflation + GDP | Same | `src/pull_worldbank_wdi.py` |

Since our non-equity data sources are identical to the original paper (vintages aside), substitution risk is isolated to the equity column. This explains why our only deviation from Table 1 occurs in equity standard deviation (45.7 vs. 48.8)—a expected result given our reliance on alternative paid data sources.

## Formatting

This project uses [Ruff](https://docs.astral.sh/ruff/) for linting and
formatting:

```bash
ruff format . && ruff check --select I --fix . && ruff check --fix .
```
