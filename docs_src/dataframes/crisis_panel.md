# Crisis Chronology Panel

`_data/crisis_panel.parquet`, built by `src/clean_crisis_chronologies.py`.
One row per (country, year), keyed by `country_iso3` and `year`.

BVX's revised chronology is the baseline crisis outcome; JST's crisis
indicator and the Reinhart-Rogoff indicator carried in the BVX kit are kept
for Table 1's chronology comparison, alongside BVX's bank equity crash, bank
failure, and panic flags and real GDP growth. Two support conventions: bank
equity crashes are recorded only in event years, so the indicator is
zero-filled elsewhere; the Reinhart-Rogoff chronology is defined only for
its covered countries and only through 2010, and stays missing outside that
support.
