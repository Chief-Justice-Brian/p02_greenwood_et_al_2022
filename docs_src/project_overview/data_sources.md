# Data Sources

| Dataset | Frequency | Use | Cleaned output |
|---|---:|---|---|
| BVX replication kit | Annual | Crisis onsets, bank crashes, failures, panics, GDP growth | `crisis_panel.parquet` |
| Laeven--Valencia (2026), IMF WP 26/94 | Annual | Updated systemic-crisis onsets, 2017--2025 | Added at analysis time by `updated_crisis_chronology.py` |
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
later years as crisis-free. The separately labelled updated exhibits append the
Laeven--Valencia (2026) systemic-crisis classifications for 2017--2025. Its
confirmed post-2016 onsets are outside the GHSS 42-country universe, making the
sample's zeros in those years documented non-onsets. Borderline cases are
excluded. The 2023 CSV remains a predictor case-study slice, not a bank-level
SVB dataset.
