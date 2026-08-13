"""Create Figure 3: annual fraction of countries in each historical R-Zone."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
START_YEAR = 1950
END_YEAR = 2012


def annual_rzone_fraction(
    panel: pd.DataFrame,
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    historical_sample: bool = True,
) -> pd.DataFrame:
    """Return annual R-Zone shares using each sector's available-country count."""
    years = pd.DataFrame({"year": range(start_year, end_year + 1)})
    output = years
    for sector in ["business", "household"]:
        eligible = (
            panel[f"in_{sector}_sample"]
            if historical_sample
            else panel[f"rzone_{sector}"].notna()
        )
        sample = panel.loc[
            eligible
            & panel["year"].between(start_year, end_year)
            & panel[f"rzone_{sector}"].notna(),
            ["year", f"rzone_{sector}"],
        ]
        annual = (
            sample.groupby("year")[f"rzone_{sector}"]
            .agg(["mean", "count"])
            .reset_index()
            .rename(
                columns={
                    "mean": f"{sector}_fraction",
                    "count": f"{sector}_n",
                }
            )
        )
        annual[f"{sector}_pct"] = 100.0 * annual[f"{sector}_fraction"]
        output = output.merge(annual, on="year", how="left")
    return output


def plot_figure3(
    annual: pd.DataFrame, png_path: Path, pdf_path: Path, jpg_path: Path
) -> None:
    plt.style.use("ggplot")
    fig, ax = plt.subplots(figsize=(9.2, 5.3))
    ax.plot(
        annual["year"],
        annual["business_pct"],
        color="#F8766D",
        linewidth=1.7,
        label="Business",
    )
    ax.plot(
        annual["year"],
        annual["household_pct"],
        color="#00BFC4",
        linewidth=1.7,
        label="Household",
    )
    ax.set_xlim(annual["year"].min() - 1, annual["year"].max() + 1)
    ax.set_ylim(bottom=-1)
    ax.set_xlabel("")
    ax.set_ylabel("% of countries in R-Zone")
    ax.grid(True, color="white", linewidth=1.0)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=2,
        frameon=False,
    )
    fig.subplots_adjust(left=0.11, right=0.98, top=0.98, bottom=0.23)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_path, dpi=300, bbox_inches="tight")
    fig.savefig(pdf_path, bbox_inches="tight")
    fig.savefig(jpg_path, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    panel = pd.read_parquet(DATA_DIR / "rzone_analysis_panel.parquet")
    annual = annual_rzone_fraction(panel)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    annual.to_csv(OUTPUT_DIR / "figure3_global_rzone.csv", index=False)
    plot_figure3(
        annual,
        OUTPUT_DIR / "figure3_fraction_countries_rzone.png",
        OUTPUT_DIR / "figure3_fraction_countries_rzone.pdf",
        OUTPUT_DIR / "figure3_fraction_countries_rzone.jpg",
    )
    print("Saved Figure 3 annual R-Zone series and publication graphics")


if __name__ == "__main__":
    main()
