# Cleaned Equity Panel

`_data/equity_panel.parquet`, built by `src/clean_equity_panel.py`.
One row per (country, year), keyed by `country_iso3` and `year`.

Spliced three-year real equity-price growth. Sources in splice priority: IMF
share price indices, then JST MacroHistory, then OECD share prices (the
project's post-2016 fallback, since the IMF stopped collecting share
prices). IMF and OECD nominal indices are deflated with the macro CPI panel;
JST growth uses its own capital-gain and CPI series so its native scale
stays consistent.
