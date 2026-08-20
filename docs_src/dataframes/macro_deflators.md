# Macro Deflators Panel

`_data/macro_deflators.parquet`, built by `src/clean_macro_deflators.py`.
One row per (country, year), keyed by `country_iso3` and `year`.

Continuous annual CPI and nominal-GDP panels for the sample countries.
World Bank WDI is the primary source; JST MacroHistory fills early
observations and gaps for the subset of sample countries it covers. Because
the two providers use different index bases and sometimes different
currency-unit scales, JST levels are rebased country by country using the
median ratio during overlap years before they are used as a supplement.
