# Data Sources

| Dataset | Frequency | Use | Cleaned output |
|---|---:|---|---|
| BVX replication kit | Annual | Crisis onsets, bank crashes, failures, panics, GDP growth | `crisis_panel.parquet` |
| CRSP US Banks portfolio (Ken French Data Library) | Daily | Bank equity criterion for the BVX-criteria extension | `us_bank_equity.parquet` |
| IMF Global Debt Database | Annual | Primary business, household, and private credit | `credit_panel.parquet` |
| BIS Total Credit | Quarterly | Credit supplement and modern extension | `credit_panel.parquet` |
| JST MacroHistory R6 | Annual | Early credit, prices, CPI/GDP, crises, bank fragility | All tidy panels |
| IMF MFS/FMP | Annual | Primary free equity-price source | `equity_panel.parquet` |
| OECD share prices | Annual | Equity supplement and post-2016 extension | `equity_panel.parquet` |
| BIS Residential Property Prices | Quarterly | Primary real house-price source | `house_price_panel.parquet` |
| OECD Analytical House Prices | Annual/quarterly | House-price supplement | `house_price_panel.parquet` |
| World Bank WDI | Annual | CPI and current-price GDP | `macro_deflators.parquet` |

Raw API responses are normalized by the `pull_*.py` scripts and saved as
Parquet. The pull scripts document the requested series codes, source URLs,
units, and update cadence.

The main limitation is equity coverage. GHSS primarily use paid GFD data, with
Bloomberg for a few countries. This public reproduction substitutes IMF, JST,
and OECD series, so a close but non-exact replication is expected.

The BVX chronology ends in 2016, so the historical replication never labels
later years as crisis-free. For 2017--2025 the outcome series
`crisis_bvx_extended` continues BVX's own criteria (a 30%+ bank equity index
decline plus narrative evidence of widespread failures or panics) using the
CRSP bank portfolio and a documented narrative screen, so "crisis" is judged
by the same rule in every year of the panel. The IMF's Laeven--Valencia
chronology is not used as a data input: it applies a different,
intervention-based definition (the two rules disagree even about the same
historical years -- the US in 1984 is a crisis under BVX but not under
Laeven--Valencia), and the IMF publishes it only as tables inside a working
paper, which fails this project's published-datasets-only policy. It is cited
in the final report as corroborating context. The 2023 CSV remains a
predictor case-study slice, not a bank-level SVB dataset.
