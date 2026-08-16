"""Create original data-understanding statistics and diagnostics.

These outputs are deliberately separate from the assigned replication
exhibits. They summarize coverage and the joint predictor distributions that
produce the sector R-Zone classifications.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from build_analysis_panel import load_analysis_panel
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))

SECTOR_COLUMNS = {
    "business": (
        "delta3_business_debt_gdp",
        "delta3_log_real_equity",
        "Business",
        "Real equity-price growth",
    ),
    "household": (
        "delta3_household_debt_gdp",
        "delta3_log_real_house_price",
        "Household",
        "Real house-price growth",
    ),
}

PERIODS = {
    "Historical estimation years": (1950, 2012),
    "Later predictor observations": (2013, 2025),
}


def _pair_mask(panel: pd.DataFrame, sector: str) -> pd.Series:
    debt, price, _, _ = SECTOR_COLUMNS[sector]
    return panel[debt].notna() & panel[price].notna()


def calculate_data_overview(panel: pd.DataFrame) -> pd.DataFrame:
    """Summarize complete predictor pairs by sector and analysis period."""
    rows = []
    country_count = panel["country_iso3"].nunique()
    for sector, (debt, price, sector_label, price_label) in SECTOR_COLUMNS.items():
        for period, (start_year, end_year) in PERIODS.items():
            sample = panel.loc[
                _pair_mask(panel, sector) & panel["year"].between(start_year, end_year),
                ["country_iso3", "year", debt, price, f"rzone_{sector}"],
            ].copy()
            possible = country_count * (end_year - start_year + 1)
            rows.append(
                {
                    "sector": sector,
                    "sector_label": sector_label,
                    "price_label": price_label,
                    "period": period,
                    "start_year": start_year,
                    "end_year": end_year,
                    "country_years": len(sample),
                    "countries": sample["country_iso3"].nunique(),
                    "coverage_pct": 100.0 * len(sample) / possible,
                    "debt_growth_mean": sample[debt].mean(),
                    "debt_growth_sd": sample[debt].std(ddof=1),
                    "price_growth_mean": sample[price].mean(),
                    "price_growth_sd": sample[price].std(ddof=1),
                    "debt_price_correlation": sample[debt].corr(sample[price]),
                    "rzone_share_pct": 100.0 * sample[f"rzone_{sector}"].mean(),
                }
            )
    return pd.DataFrame(rows)


def build_latex_table(summary: pd.DataFrame) -> str:
    """Render the overview statistics as a compact LaTeX tabular fragment."""
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        r"Sector & Period & $N$ & Countries & Coverage & Debt growth & Price growth & Corr. / R-Zone \\",
        r" & & & & (\%) & Mean (SD) & Mean (SD) & $\rho$ / \% \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            " & ".join(
                [
                    row.sector_label,
                    f"{row.start_year}--{row.end_year}",
                    f"{row.country_years:,}",
                    f"{row.countries}",
                    f"{row.coverage_pct:.1f}",
                    f"{row.debt_growth_mean:.1f} ({row.debt_growth_sd:.1f})",
                    f"{row.price_growth_mean:.1f} ({row.price_growth_sd:.1f})",
                    f"{row.debt_price_correlation:.2f} / {row.rzone_share_pct:.1f}",
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def _annual_coverage(panel: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sector in SECTOR_COLUMNS:
        counts = (
            panel.loc[_pair_mask(panel, sector) & panel["year"].between(1950, 2025)]
            .groupby("year")["country_iso3"]
            .nunique()
        )
        rows.extend(
            {"year": int(year), "sector": sector, "countries": int(count)}
            for year, count in counts.items()
        )
    return pd.DataFrame(rows)


def plot_data_overview(panel: pd.DataFrame, output_stem: Path) -> None:
    """Plot annual coverage and later-period joint predictor distributions."""
    fig = plt.figure(figsize=(11.5, 8.2))
    grid = fig.add_gridspec(2, 2, height_ratios=[0.82, 1.2], hspace=0.38, wspace=0.26)
    coverage_ax = fig.add_subplot(grid[0, :])
    business_ax = fig.add_subplot(grid[1, 0])
    household_ax = fig.add_subplot(grid[1, 1])

    colors = {"business": "#E76F51", "household": "#009FB7"}
    coverage = _annual_coverage(panel)
    for sector in ["business", "household"]:
        subset = coverage.loc[coverage["sector"].eq(sector)]
        coverage_ax.plot(
            subset["year"],
            subset["countries"],
            color=colors[sector],
            linewidth=1.8,
            label=SECTOR_COLUMNS[sector][2],
        )
    coverage_ax.axvspan(2012.5, 2025.5, color="#E9ECEF", zorder=-1)
    coverage_ax.axvline(2012.5, color="#555555", linestyle="--", linewidth=1.0)
    coverage_ax.text(
        2013.3,
        2.0,
        "Later predictor observations",
        fontsize=8.5,
        color="#444444",
    )
    coverage_ax.set_xlim(1950, 2025)
    coverage_ax.set_ylim(0, 43)
    coverage_ax.set_ylabel("Countries with complete pair")
    coverage_ax.set_title("Panel A: Predictor coverage over time", loc="left")
    coverage_ax.legend(frameon=False, ncol=2, loc="upper left")
    coverage_ax.grid(alpha=0.22)

    for sector, ax in [("business", business_ax), ("household", household_ax)]:
        debt, price, sector_label, price_label = SECTOR_COLUMNS[sector]
        sample = panel.loc[
            _pair_mask(panel, sector) & panel["year"].between(2013, 2025)
        ].copy()
        in_rzone = sample[f"rzone_{sector}"].eq(1)
        ax.scatter(
            sample.loc[~in_rzone, debt],
            sample.loc[~in_rzone, price],
            s=17,
            alpha=0.38,
            color="#7A7A7A",
            edgecolors="none",
            label="Outside R-Zone",
        )
        ax.scatter(
            sample.loc[in_rzone, debt],
            sample.loc[in_rzone, price],
            s=35,
            alpha=0.9,
            color=colors[sector],
            edgecolors="white",
            linewidths=0.45,
            label="R-Zone",
            zorder=3,
        )
        debt_cutoff = float(panel[f"{sector}_debt_q80_cutoff"].dropna().iloc[0])
        price_cutoff = float(panel[f"{sector}_price_t667_cutoff"].dropna().iloc[0])
        ax.axvline(debt_cutoff, color="#333333", linestyle="--", linewidth=1.05)
        ax.axhline(price_cutoff, color="#333333", linestyle="--", linewidth=1.05)
        ax.set_xlim(sample[debt].quantile(0.01), sample[debt].quantile(0.99))
        ax.set_ylim(sample[price].quantile(0.01), sample[price].quantile(0.99))
        ax.set_xlabel(r"$\Delta_3$ debt / GDP (percentage points)")
        ax.set_ylabel(rf"$\Delta_3$ {price_label} (%)")
        panel_letter = "B" if sector == "business" else "C"
        ax.set_title(
            f"Panel {panel_letter}: {sector_label} predictors, 2013--2025",
            loc="left",
        )
        ax.grid(alpha=0.18)
        ax.text(
            0.97,
            0.95,
            "R-Zone",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color=colors[sector],
            fontsize=9,
            fontweight="bold",
        )
    business_ax.legend(frameon=False, loc="lower left", fontsize=8)

    fig.savefig(output_stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(
        output_stem.with_suffix(".jpg"),
        dpi=300,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_analysis_panel(DATA_DIR)
    summary = calculate_data_overview(panel)
    summary.to_csv(OUTPUT_DIR / "data_overview_summary.csv", index=False)
    (OUTPUT_DIR / "data_overview_summary.tex").write_text(
        build_latex_table(summary), encoding="utf-8"
    )
    plot_data_overview(panel, OUTPUT_DIR / "data_overview_figure")
    print("Saved original data-overview table and diagnostic figure")


if __name__ == "__main__":
    main()
