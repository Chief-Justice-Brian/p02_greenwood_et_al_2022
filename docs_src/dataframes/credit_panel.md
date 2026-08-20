# Cleaned Credit Panel

`_data/credit_panel.parquet`, built by `src/clean_credit_panel.py`.
One row per (country, year), keyed by `country_iso3` and `year`.

Three-year changes are computed inside each provider before splicing, as the
paper's footnote 6 requires. Sources in priority: IMF Global Debt
Database, then BIS Total Credit Statistics (which also carries the extension
forward), then JST MacroHistory supplying early history for the subset of
sample countries it covers.

Four measures are produced: the business, household, and total private
debt-to-GDP three-year changes (percentage points), and the three-year log
change in real total private debt (x100). A source label sits beside every
measure so each observation's provider is auditable.
