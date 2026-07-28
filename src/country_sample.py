"""The 42-country sample of Greenwood-Hanson-Shleifer-Sorensen (2022) + code maps.

The paper's baseline sample (HBS WP 20-130) is an unbalanced panel of 42
countries, 1950-2016. The paper enumerates all 42 by name in the notes to
Table 5, robustness rows (xi) and (xii), which partition the sample into the
26 countries the World Bank classified as high-income in 1995 and the 16 it
classified as low- or medium-income. That text enumeration is the source for
``GHSS_COUNTRIES`.

Definitions:

``GHSS_COUNTRIES``
    ISO3 -> country name, for the 42 countries the paper names and no others.
    Sets the country grid of the analysis panel, so it fixes the row set
    behind Table 1.
``BIS_ISO2_TO_ISO3``
    BIS 2-letter country code -> ISO3, for those same 42 and nothing else.
    Puts BIS total credit and BIS property prices onto the panel, via
    ``clean_credit_panel`` and ``clean_house_price_panel``.

Mapping through ``BIS_ISO2_TO_ISO3`` and dropping NaN is also the filter that
removes BIS aggregates (``4T``, ``5A``, ``5R``, ``G2``, ``XM``, ...), which are
deliberately absent from it. The same mechanism is why an omission is
dangerous: a sample country missing from the map loses its BIS data silently
instead of failing.

Not stated by the paper: GHSS never say which countries they excluded or why,
so any claim about the exclusions is ours, not theirs. BVX's chronology covers
more countries than GHSS's 42, and comparing the two lists is what identifies
the difference (Egypt, the Philippines, Taiwan and Venezuela). That comparison
belongs in a test against the loaded BVX file rather than in prose here, where
nothing checks it.
"""

# Named by the paper in its Table 5 notes, rows (xi) and (xii)
GHSS_COUNTRIES = {
    "ARG": "Argentina",
    "AUS": "Australia",
    "AUT": "Austria",
    "BEL": "Belgium",
    "BRA": "Brazil",
    "CAN": "Canada",
    "CHE": "Switzerland",
    "CHL": "Chile",
    "COL": "Colombia",
    "CZE": "Czech Republic",
    "DEU": "Germany",
    "DNK": "Denmark",
    "ESP": "Spain",
    "FIN": "Finland",
    "FRA": "France",
    "GBR": "United Kingdom",
    "GRC": "Greece",
    "HKG": "Hong Kong",
    "HUN": "Hungary",
    "IDN": "Indonesia",
    "IND": "India",
    "IRL": "Ireland",
    "ISL": "Iceland",
    "ISR": "Israel",
    "ITA": "Italy",
    "JPN": "Japan",
    "KOR": "Korea",
    "LUX": "Luxembourg",
    "MEX": "Mexico",
    "MYS": "Malaysia",
    "NLD": "Netherlands",
    "NOR": "Norway",
    "NZL": "New Zealand",
    "PER": "Peru",
    "PRT": "Portugal",
    "RUS": "Russia",
    "SGP": "Singapore",
    "SWE": "Sweden",
    "THA": "Thailand",
    "TUR": "Turkey",
    "USA": "United States",
    "ZAF": "South Africa",
}

# Keys must stay in sync with GHSS_COUNTRIES -- see the module docstring
BIS_ISO2_TO_ISO3 = {
    "AR": "ARG",
    "AT": "AUT",
    "AU": "AUS",
    "BE": "BEL",
    "BR": "BRA",
    "CA": "CAN",
    "CH": "CHE",
    "CL": "CHL",
    "CO": "COL",
    "CZ": "CZE",
    "DE": "DEU",
    "DK": "DNK",
    "ES": "ESP",
    "FI": "FIN",
    "FR": "FRA",
    "GB": "GBR",
    "GR": "GRC",
    "HK": "HKG",
    "HU": "HUN",
    "ID": "IDN",
    "IE": "IRL",
    "IL": "ISR",
    "IN": "IND",
    "IS": "ISL",
    "IT": "ITA",
    "JP": "JPN",
    "KR": "KOR",
    "LU": "LUX",
    "MX": "MEX",
    "MY": "MYS",
    "NL": "NLD",
    "NO": "NOR",
    "NZ": "NZL",
    "PE": "PER",
    "PT": "PRT",
    "RU": "RUS",
    "SE": "SWE",
    "SG": "SGP",
    "TH": "THA",
    "TR": "TUR",
    "US": "USA",
    "ZA": "ZAF",
}
