"""Build Figure 1 (event history), R-zone timelines, and the post-2016 tracker.

``plot_event_history`` and ``combine_event_history_panels`` produce the
assigned Figure 1 replication (``figure1_event_history.png``/``.pdf``);
``rzone_summary`` writes ``rzone_validation.csv`` against the paper
benchmarks; ``post_2016_tracker`` writes the country-by-year classification
with distance-to-threshold columns.
"""

from pathlib import Path
from textwrap import fill

import matplotlib.pyplot as plt
import pandas as pd

from build_analysis_panel import load_analysis_panel
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))

# Top-to-bottom order used in the paper's event-history figure. Keeping the
# order fixed makes the business and household panels directly comparable.
EVENT_HISTORY_ORDER = [
    "USA",
    "GBR",
    "THA",
    "TUR",
    "CHE",
    "SWE",
    "ESP",
    "ZAF",
    "SGP",
    "RUS",
    "PRT",
    "PER",
    "NOR",
    "NZL",
    "NLD",
    "MEX",
    "MYS",
    "LUX",
    "KOR",
    "JPN",
    "ITA",
    "ISR",
    "IRL",
    "IDN",
    "IND",
    "ISL",
    "HUN",
    "HKG",
    "GRC",
    "DEU",
    "FRA",
    "FIN",
    "DNK",
    "CZE",
    "COL",
    "CHL",
    "CAN",
    "BRA",
    "BEL",
    "AUT",
    "AUS",
    "ARG",
]


def quantile_cell_results(panel):
    """Compute raw Table 3 cell sizes and crisis frequencies.

    :param panel: country-year analysis panel from load_analysis_panel.
    :returns: one row per sector x horizon x price-tercile x debt-quintile cell
        with its count, share of the sector sample, and crisis frequency in
        percent.
    """
    records = []
    for sector in ["business", "household"]:
        sample = panel[panel[f"in_{sector}_sample"]].copy()
        debt = f"{sector}_debt_quintile"
        price = f"{sector}_price_tercile"
        for horizon in [1, 2, 3, 4]:
            outcome = f"crisis_next_{horizon}y"
            grouped = sample.groupby([price, debt], observed=True)[outcome]
            for (price_tercile, debt_quintile), values in grouped:
                records.append(
                    {
                        "sector": sector,
                        "horizon": horizon,
                        "price_tercile": int(price_tercile),
                        "debt_quintile": int(debt_quintile),
                        "n": values.count(),
                        "distribution_pct": 100.0 * values.count() / len(sample),
                        "crisis_frequency_pct": 100.0 * values.mean(),
                    }
                )
    return pd.DataFrame(records)


def rzone_summary(panel):
    """Compare each sector's R-zone frequency and 3-year crisis rate to the paper.

    :param panel: country-year analysis panel from load_analysis_panel.
    :returns: one row per sector with our values next to the published
        benchmarks.
    """
    rows = []
    paper_frequency = {"business": 6.0, "household": 10.3}
    paper_crisis_3y = {"business": 45.3, "household": 36.8}
    for sector in ["business", "household"]:
        sample = panel[panel[f"in_{sector}_sample"]]
        in_rzone = sample[f"rzone_{sector}"].eq(1)
        rows.append(
            {
                "sector": sector,
                "n": len(sample),
                "rzone_n": int(in_rzone.sum()),
                "rzone_frequency_pct": 100.0 * in_rzone.mean(),
                "paper_rzone_frequency_pct": paper_frequency[sector],
                "crisis_within_3y_given_rzone_pct": 100.0
                * sample.loc[in_rzone, "crisis_next_3y"].mean(),
                "paper_crisis_within_3y_given_rzone_pct": paper_crisis_3y[sector],
            }
        )
    return pd.DataFrame(rows)


def post_2016_tracker(panel):
    """Build the post-2016 country-by-year R-zone tracker.

    Adds distance-to-cutoff columns (predictor minus frozen historical cutoff)
    so borderline calls are visible; drops rows where both sector flags are
    missing.

    :param panel: country-year analysis panel from load_analysis_panel.
    """
    columns = [
        "country_iso3",
        "country",
        "year",
        "delta3_business_debt_gdp",
        "delta3_log_real_equity",
        "high_business_debt_growth",
        "high_business_price_growth",
        "rzone_business",
        "delta3_household_debt_gdp",
        "delta3_log_real_house_price",
        "high_household_debt_growth",
        "high_household_price_growth",
        "rzone_household",
        "rzone_either",
    ]
    tracker = panel[panel["is_post_2016"]][columns].copy()
    tracker["business_debt_distance"] = (
        tracker["delta3_business_debt_gdp"] - panel["business_debt_q80_cutoff"].iloc[0]
    )
    tracker["business_price_distance"] = (
        tracker["delta3_log_real_equity"] - panel["business_price_t667_cutoff"].iloc[0]
    )
    tracker["household_debt_distance"] = (
        tracker["delta3_household_debt_gdp"]
        - panel["household_debt_q80_cutoff"].iloc[0]
    )
    tracker["household_price_distance"] = (
        tracker["delta3_log_real_house_price"]
        - panel["household_price_t667_cutoff"].iloc[0]
    )
    return tracker.dropna(
        subset=["rzone_business", "rzone_household"], how="all"
    ).reset_index(drop=True)


def _sector_pair_mask(panel, sector):
    """Mark rows where both of the sector's predictors are observed.

    :param panel: country-year frame holding the delta3 debt and price columns.
    :param sector: "business" or "household".
    """
    price_column = (
        "delta3_log_real_equity"
        if sector == "business"
        else "delta3_log_real_house_price"
    )
    return panel[f"delta3_{sector}_debt_gdp"].notna() & panel[price_column].notna()


def plot_event_history(
    panel,
    sector,
    output_path,
    end_year=2016,
    crisis_column="crisis_bvx",
    crisis_label="BVX (2019) crisis",
):
    """Plot one publication-style sector R-zone and crisis event history.

    :param panel: country-year analysis panel from load_analysis_panel.
    :param sector: "business" or "household".
    :param output_path: file path the finished panel image is saved to.
    :param end_year: last plotted year; coverage lines and axis ticks adjust
        to it.
    :param crisis_column: 0/1 panel column whose onsets are drawn as dots.
    :param crisis_label: legend text for the crisis dots.
    :raises ValueError: if sector is not "business" or "household".
    """
    if sector not in {"business", "household"}:
        raise ValueError("sector must be 'business' or 'household'")

    historical = panel[panel["year"].between(1950, end_year)].copy()
    country_names = (
        historical[["country_iso3", "country"]]
        .drop_duplicates("country_iso3")
        .set_index("country_iso3")["country"]
        .to_dict()
    )
    countries = [country for country in EVENT_HISTORY_ORDER if country in country_names]
    positions = {country: position for position, country in enumerate(countries)}

    pair_mask = _sector_pair_mask(historical, sector)
    coverage = (
        historical[pair_mask]
        .groupby("country_iso3")["year"]
        .agg(start="min", end="max")
    )

    fig, ax = plt.subplots(figsize=(14, 11.5))
    ax.set_facecolor("#EBEBEB")

    # A light guide runs through every country. The black segment records the
    # years for which both sector predictors exist; a dotted leader connects
    # the series to the country label on the right.
    for country in countries:
        y = positions[country]
        ax.hlines(y, 1938, end_year, color="white", linewidth=0.9, zorder=0)
        if country in coverage.index:
            start = int(coverage.loc[country, "start"])
            end = int(coverage.loc[country, "end"])
            ax.hlines(y, start, end, color="black", linewidth=1.15, zorder=1)
            ax.hlines(
                y,
                end,
                end_year + 2.2,
                color="black",
                linewidth=0.85,
                linestyles=":",
                zorder=1,
            )
        else:
            ax.hlines(
                y,
                1950,
                end_year + 2.2,
                color="#888888",
                linewidth=0.8,
                linestyles=":",
                zorder=1,
            )
        ax.text(
            end_year + 2.7,
            y,
            country_names[country],
            ha="left",
            va="center",
            fontsize=7.2,
            color="black" if country in coverage.index else "#777777",
        )

    rzones = historical[
        historical[f"rzone_{sector}"].eq(1) & historical["country_iso3"].isin(countries)
    ]
    crises = historical[
        historical[crisis_column].eq(1) & historical["country_iso3"].isin(countries)
    ]

    # "J-Zone" is not a typo: the paper's published Figure 1 legend reads
    # "J-Zone (Bus.)" / "J-Zone (HH.)" (WP 20-130 pp. 55-56), even though the
    # text calls it the R-Zone. We replicate the exhibit as published.
    rzone_label = "J-Zone (Bus.)" if sector == "business" else "J-Zone (HH.)"
    ax.scatter(
        crises["year"],
        crises["country_iso3"].map(positions),
        marker="o",
        s=39,
        facecolor="#F8766D",
        edgecolor="white",
        linewidth=0.45,
        zorder=4,
        label=crisis_label,
    )
    ax.scatter(
        rzones["year"],
        rzones["country_iso3"].map(positions),
        marker="x",
        s=43,
        color="#00BFC4",
        linewidth=1.55,
        zorder=5,
        label=rzone_label,
    )

    panel_letter = "A" if sector == "business" else "B"
    sector_title = "Business R-Zone" if sector == "business" else "Household R-Zone"
    ax.set_title(
        f"Panel {panel_letter}: {sector_title}",
        loc="left",
        fontsize=12,
        fontweight="bold",
        pad=12,
    )
    ax.set_xlim(1937, end_year + 14)
    ax.set_ylim(-0.8, len(countries) - 0.2)
    ax.invert_yaxis()
    ax.set_xlabel("Year", fontsize=9)
    ax.set_yticks([])
    ticks = [1940, 1960, 1980, 2000]
    if end_year >= 2020:
        ticks.append(2020)
    ax.set_xticks(ticks)
    ax.tick_params(axis="x", labelsize=8, length=0)
    ax.grid(axis="x", color="white", linewidth=0.8)
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.48, -0.07),
        ncol=2,
        frameon=False,
        fontsize=8,
        handletextpad=0.45,
        columnspacing=1.1,
    )
    fig.subplots_adjust(left=0.04, right=0.98, top=0.94, bottom=0.1)
    fig.savefig(output_path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def combine_event_history_panels(
    business_path,
    household_path,
    output_paths,
    title="Figure 1: Event History",
    caption=None,
):
    """Combine the two saved sector panel images into the paper-style Figure 1.

    :param business_path: saved image file of the business-sector panel
        (Panel A).
    :param household_path: saved image file of the household-sector panel
        (Panel B).
    :param output_paths: every path the combined figure is saved to (e.g.
        .png and .pdf).
    :param title: heading text drawn above the combined panels.
    :param caption: caption text under the title; None uses the replication
        default.
    """
    business_image = plt.imread(business_path)
    household_image = plt.imread(household_path)

    fig = plt.figure(figsize=(15, 23), facecolor="white")
    grid = fig.add_gridspec(
        2,
        1,
        left=0.015,
        right=0.985,
        bottom=0.015,
        top=0.91,
        hspace=0.035,
    )
    for image, cell in [(business_image, grid[0]), (household_image, grid[1])]:
        axis = fig.add_subplot(cell)
        axis.imshow(image)
        axis.axis("off")

    fig.text(
        0.02,
        0.975,
        title,
        ha="left",
        va="top",
        fontsize=15,
        fontweight="bold",
    )
    if caption is None:
        caption = (
            "Panel A plots R-Zone events measured with business debt growth and "
            "equity price growth together with financial-crisis onsets from Baron, "
            "Verner, and Xiong (2019). Panel B presents the analogous event history "
            "using household debt growth and house-price growth. Black lines show "
            "the country-years for which both sector predictors are available in "
            "our public-data reconstruction."
        )
    fig.text(
        0.02,
        0.952,
        fill(caption, width=155),
        ha="left",
        va="top",
        fontsize=10,
        linespacing=1.35,
    )
    for output_path in output_paths:
        fig.savefig(output_path, dpi=180, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_analysis_panel(DATA_DIR)
    quantile_cell_results(panel).to_csv(
        OUTPUT_DIR / "table3_cell_frequencies_raw.csv", index=False
    )
    rzone_summary(panel).to_csv(OUTPUT_DIR / "rzone_validation.csv", index=False)
    post_2016_tracker(panel).to_csv(
        OUTPUT_DIR / "post_2016_rzone_tracker.csv", index=False
    )
    plot_event_history(
        panel,
        "business",
        OUTPUT_DIR / "historical_business_rzone_timeline.png",
    )
    plot_event_history(
        panel,
        "household",
        OUTPUT_DIR / "historical_household_rzone_timeline.png",
    )
    combine_event_history_panels(
        OUTPUT_DIR / "historical_business_rzone_timeline.png",
        OUTPUT_DIR / "historical_household_rzone_timeline.png",
        [
            OUTPUT_DIR / "figure1_event_history.png",
            OUTPUT_DIR / "figure1_event_history.pdf",
        ],
    )
    print(
        "Saved descriptive results, Figure 1 event history and its separate "
        "business/household panels, and post-2016 tracker"
    )
