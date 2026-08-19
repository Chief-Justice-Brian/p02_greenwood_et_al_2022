"""Country lists and BIS code mappings for the GHSS 42-country sample.

The Greenwood-Hanson-Shleifer-Sorensen (2022) baseline sample is an
unbalanced panel of 42 countries, 1950-2016. The paper lists these countries
in the notes to Table 5, rows (xi) and (xii); that enumeration defines
``GHSS_COUNTRIES``.

``GHSS_COUNTRIES``
    ISO3 -> country name for the 42 countries in the paper's sample. This
    defines the country grid used throughout the analysis panel.

``BIS_ISO2_TO_ISO3``
    BIS 2-letter country code -> ISO3 for those same countries. This mapping
    is used to merge BIS credit and house-price data onto the analysis panel.

The BIS mapping intentionally excludes aggregate codes. Because unmatched
BIS codes are dropped during mapping, omitting a sample country would
silently remove its BIS observations.
"""

# 42 countries listed by the paper in the Table 5 notes, rows (xi) and (xii)
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

# BIS codes for the same 42 countries; excludes BIS aggregate codes
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