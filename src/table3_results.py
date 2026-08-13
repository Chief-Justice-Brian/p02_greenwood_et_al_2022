"""Build the publication-style Table 3 replication.

For every price-tercile/debt-quintile cell, this script reports the share of
observations, the crisis frequency at horizons one through four, and the
difference from the median cell (price tercile 2, debt quintile 3).  Difference
p-values use Driscoll--Kraay standard errors with the horizon-specific lags used
throughout the replication.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from linearmodels.panel import PooledOLS

from baseline_regressions import DK_BANDWIDTH


ROOT = Path(__file__).resolve().parents[1]
PANEL_PATH = ROOT / "_data" / "rzone_analysis_panel.parquet"
RESULTS_PATH = ROOT / "_output" / "table3_crisis_probabilities.csv"
LATEX_PATH = ROOT / "_output" / "table3_replication.tex"

SECTOR_COLUMNS = {
    "business": ("business_price_tercile", "business_debt_quintile"),
    "household": ("household_price_tercile", "household_debt_quintile"),
}


def _difference_from_median(
    sample: pd.DataFrame,
    outcome: str,
    price_col: str,
    debt_col: str,
    price_tercile: int,
    debt_quintile: int,
    bandwidth: int,
) -> tuple[float, float, float]:
    """Return target-minus-median difference, t-statistic, and p-value."""
    if price_tercile == 2 and debt_quintile == 3:
        return 0.0, np.nan, np.nan

    target = (sample[price_col] == price_tercile) & (
        sample[debt_col] == debt_quintile
    )
    median = (sample[price_col] == 2) & (sample[debt_col] == 3)
    comparison = sample.loc[
        target | median, ["country_iso3", "year", outcome]
    ].dropna()
    comparison = comparison.copy()
    comparison["constant"] = 1.0
    comparison["target"] = (
        (sample.loc[comparison.index, price_col] == price_tercile)
        & (sample.loc[comparison.index, debt_col] == debt_quintile)
    ).astype(float)
    comparison = comparison.set_index(["country_iso3", "year"])

    result = PooledOLS(
        comparison[outcome].astype(float), comparison[["constant", "target"]]
    ).fit(cov_type="driscoll-kraay", bandwidth=bandwidth)
    return (
        100.0 * float(result.params["target"]),
        float(result.tstats["target"]),
        float(result.pvalues["target"]),
    )


def calculate_table3(panel: pd.DataFrame) -> pd.DataFrame:
    """Calculate every cell needed by Panels A--D."""
    rows: list[dict[str, float | int | str]] = []
    for sector, (price_col, debt_col) in SECTOR_COLUMNS.items():
        sample = panel.loc[panel[f"in_{sector}_sample"]].copy()
        total = len(sample)
        for horizon in range(1, 5):
            outcome = f"crisis_next_{horizon}y"
            for price_tercile in range(1, 4):
                for debt_quintile in range(1, 6):
                    cell_mask = (sample[price_col] == price_tercile) & (
                        sample[debt_col] == debt_quintile
                    )
                    cell = sample.loc[cell_mask]
                    difference, t_stat, p_value = _difference_from_median(
                        sample,
                        outcome,
                        price_col,
                        debt_col,
                        price_tercile,
                        debt_quintile,
                        DK_BANDWIDTH[horizon],
                    )
                    rows.append(
                        {
                            "sector": sector,
                            "horizon": horizon,
                            "price_tercile": price_tercile,
                            "debt_quintile": debt_quintile,
                            "n": len(cell),
                            "distribution_pct": 100.0 * len(cell) / total,
                            "crisis_frequency_pct": 100.0 * cell[outcome].mean(),
                            "difference_from_median_pct": difference,
                            "difference_t_stat": t_stat,
                            "difference_p_value": p_value,
                        }
                    )
    return pd.DataFrame(rows)


def _stars(p_value: float) -> str:
    if pd.isna(p_value):
        return ""
    if p_value < 0.01:
        return "***"
    if p_value < 0.05:
        return "**"
    if p_value < 0.10:
        return "*"
    return ""


def _number(value: float, p_value: float = np.nan, highlight: bool = False) -> str:
    rendered = f"{value:.1f}"
    stars = _stars(p_value)
    if stars:
        rendered += rf"\textsuperscript{{{stars}}}"
    if highlight:
        rendered = rf"\cellcolor{{gray!20}}\textbf{{{rendered}}}"
    return rendered


def _distribution_panel(data: pd.DataFrame, letter: str, sector: str) -> str:
    label = "Business Debt and Equity Prices" if sector == "business" else (
        "Household Debt and House Prices"
    )
    lines = [
        rf"\noindent\textbf{{Panel {letter}:}} \textit{{Distribution of Observations (\%) by Growth in {label}}}",
        r"\begin{center}",
        r"\begin{tabular}{rrrrrr}",
        r" & \multicolumn{5}{c}{Debt Quintile} \\",
        r"\addlinespace[2pt]",
        r"Price Tercile & 1 & 2 & 3 & 4 & 5 \\",
        r"\midrule",
    ]
    base = data[(data["sector"] == sector) & (data["horizon"] == 1)]
    for price in range(1, 4):
        cells = [
            _number(
                float(
                    base.loc[
                        (base["price_tercile"] == price)
                        & (base["debt_quintile"] == debt),
                        "distribution_pct",
                    ].iloc[0]
                )
            )
            for debt in range(1, 6)
        ]
        lines.append(f"{price} & " + " & ".join(cells) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{center}"])
    return "\n".join(lines)


def _probability_horizon(data: pd.DataFrame, sector: str, horizon: int) -> str:
    subset = data[(data["sector"] == sector) & (data["horizon"] == horizon)]
    year_label = "year" if horizon == 1 else "years"
    lines = [
        rf"\noindent\textbf{{{horizon}-{year_label} horizon}}",
        r"\begin{center}",
        r"\begin{tabular}{rrrrrr@{\hspace{1.8em}}rrrrr}",
        r" & \multicolumn{5}{c}{\textit{Crisis Frequency}} & \multicolumn{5}{c}{\textit{Diff. from Median}} \\",
        r" & \multicolumn{5}{c}{Debt Quintile} & \multicolumn{5}{c}{Debt Quintile} \\",
        r"\addlinespace[2pt]",
        r"Price Tercile & 1 & 2 & 3 & 4 & 5 & 1 & 2 & 3 & 4 & 5 \\",
        r"\midrule",
    ]
    for price in range(1, 4):
        frequencies: list[str] = []
        differences: list[str] = []
        for debt in range(1, 6):
            cell = subset[
                (subset["price_tercile"] == price)
                & (subset["debt_quintile"] == debt)
            ].iloc[0]
            highlight = price == 3 and debt == 5
            frequencies.append(
                _number(float(cell["crisis_frequency_pct"]), highlight=highlight)
            )
            differences.append(
                _number(
                    float(cell["difference_from_median_pct"]),
                    float(cell["difference_p_value"]),
                    highlight,
                )
            )
        lines.append(
            f"{price} & "
            + " & ".join(frequencies)
            + " & "
            + " & ".join(differences)
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{center}"])
    return "\n".join(lines)


def _probability_panel(data: pd.DataFrame, letter: str, sector: str) -> str:
    label = "Business Debt and Equity Prices" if sector == "business" else (
        "Household Debt and House Prices"
    )
    blocks = [
        rf"\noindent\textbf{{Panel {letter}:}} \textit{{Crisis Probabilities (\%) by Growth in {label}}}"
    ]
    for horizon in range(1, 5):
        blocks.append(_probability_horizon(data, sector, horizon))
    return "\n\n".join(blocks)


def build_latex(data: pd.DataFrame) -> str:
    """Render the four publication-style panels as a LaTeX fragment."""
    return "\n\n".join(
        [
            r"\begingroup\small\setlength{\tabcolsep}{3.5pt}\renewcommand{\arraystretch}{0.92}",
            _distribution_panel(data, "A", "business"),
            _probability_panel(data, "B", "business"),
            r"\newpage",
            _distribution_panel(data, "C", "household"),
            _probability_panel(data, "D", "household"),
            r"\endgroup",
        ]
    )


def main() -> None:
    panel = pd.read_parquet(PANEL_PATH)
    results = calculate_table3(panel)
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_PATH, index=False)
    LATEX_PATH.write_text(build_latex(results), encoding="utf-8")
    print(f"Wrote {len(results):,} Table 3 cells to {RESULTS_PATH}")
    print(f"Wrote formatted Table 3 to {LATEX_PATH}")


if __name__ == "__main__":
    main()
