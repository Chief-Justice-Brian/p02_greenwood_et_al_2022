# R-Zone Analysis Panel

`_data/rzone_analysis_panel.parquet`, built by `src/build_analysis_panel.py`.
One row per (country, year), keyed by `country_iso3` and `year`.

The panel splices credit, equity, and house-price series from the project's
public sources, computes three-year debt and real price growth for the
business and household sectors, assigns R-zone indicators from frozen
historical quantile thresholds, and attaches forward crisis windows from the
extended BVX chronology. Sample flags mark which forecast origins enter the
historical replication versus the post-publication update.

See `docs_src/project_overview/methodology.md` for the sample construction,
splicing, and missing-data rules, and each `src/pull_*.py` module docstring
for source-level documentation.
