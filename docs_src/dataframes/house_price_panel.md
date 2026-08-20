# Cleaned House Price Panel

`_data/house_price_panel.parquet`, built by `src/clean_house_price_panel.py`.
One row per (country, year), keyed by `country_iso3` and `year`.

Spliced three-year real house-price growth. Sources in splice priority: BIS
real residential property prices, then OECD analytical house prices, then
JST MacroHistory.
