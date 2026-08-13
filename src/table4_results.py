"""Format the baseline regression estimates as publication-style Table 4."""

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "_output"
COEFFICIENTS_PATH = OUTPUT_DIR / "table4_baseline_coefficients.csv"
MODELS_PATH = OUTPUT_DIR / "table4_baseline_models.csv"
LATEX_PATH = OUTPUT_DIR / "table4_replication.tex"

SPECIFICATIONS = ["high_debt_only", "high_price_only", "full", "rzone_only"]


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


def _estimate(value: float, p_value: float) -> str:
    return f"{value:.1f}" + (
        rf"\textsuperscript{{{_stars(p_value)}}}" if _stars(p_value) else ""
    )


def _coefficient_lookup(coefficients: pd.DataFrame) -> dict[tuple, pd.Series]:
    return {
        (row.sector, int(row.horizon), row.specification, row.variable): row
        for row in coefficients.itertuples(index=False)
    }


def _model_lookup(models: pd.DataFrame) -> dict[tuple, pd.Series]:
    return {
        (row.sector, int(row.horizon), row.specification): row
        for row in models.itertuples(index=False)
    }


def _sector_panel(
    coefficients: pd.DataFrame, models: pd.DataFrame, sector: str, letter: str
) -> str:
    coefficient_lookup = _coefficient_lookup(coefficients)
    model_lookup = _model_lookup(models)
    suffix = "Bus." if sector == "business" else "HH"
    variables = [
        (
            rf"High Debt Growth$^{{{suffix}}}$ ($\beta^{{(h)}}$)",
            f"high_{sector}_debt_growth",
        ),
        (
            rf"High Price Growth$^{{{suffix}}}$ ($\delta^{{(h)}}$)",
            f"high_{sector}_price_growth",
        ),
        (rf"R-Zone$^{{{suffix}}}$ ($\gamma^{{(h)}}$)", f"rzone_{sector}"),
    ]

    lines = [
        rf"\noindent\textit{{Panel {letter}: {sector.title()} Sector}}",
        r"\par\smallskip",
        r"\noindent\resizebox{\linewidth}{!}{%",
        r"\begin{tabular}{l*{16}{r}}",
        r"\toprule",
        " & "
        + " & ".join(
            [
                rf"\multicolumn{{4}}{{c}}{{Crisis within {h} "
                + ("year" if h == 1 else "years")
                + "}"
                for h in range(1, 5)
            ]
        )
        + r" \\",
        " & "
        + " & ".join(
            [rf"({h}.{spec})" for h in range(1, 5) for spec in range(1, 5)]
        )
        + r" \\",
        r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-13}\cmidrule(lr){14-17}",
    ]

    for label, variable in variables:
        estimates: list[str] = []
        t_stats: list[str] = []
        for horizon in range(1, 5):
            for specification in SPECIFICATIONS:
                key = (sector, horizon, specification, variable)
                if key in coefficient_lookup:
                    row = coefficient_lookup[key]
                    estimates.append(_estimate(row.coefficient_pp, row.p_value))
                    t_stats.append(rf"[{row.t_stat:.1f}]")
                else:
                    estimates.append("")
                    t_stats.append("")
        lines.append(label + " & " + " & ".join(estimates) + r" \\")
        lines.append(" & " + " & ".join(t_stats) + r" \\")
        lines.append(r"\addlinespace[2pt]")

    combined_values: list[str] = []
    combined_t_stats: list[str] = []
    r_squared: list[str] = []
    observations: list[str] = []
    for horizon in range(1, 5):
        for specification in SPECIFICATIONS:
            row = model_lookup[(sector, horizon, specification)]
            combined_values.append(
                _estimate(row.combined_effect_pp, row.combined_p_value)
            )
            combined_t_stats.append(rf"[{row.combined_t_stat:.1f}]")
            r_squared.append(f"{row.within_r2_pct:.1f}")
            observations.append(f"{int(row.n):,}")

    lines.extend(
        [
            r"\midrule",
            r"Sum of coefficients ($\beta^{(h)}+\delta^{(h)}+\gamma^{(h)}$) & "
            + " & ".join(combined_values)
            + r" \\",
            r"t-statistic ($\beta^{(h)}+\delta^{(h)}+\gamma^{(h)}$) & " + " & ".join(combined_t_stats) + r" \\",
            r"$R^2$ (within) & " + " & ".join(r_squared) + r" \\",
            r"$N$ & " + " & ".join(observations) + r" \\",
            r"\bottomrule",
            r"\end{tabular}%",
            r"}",
        ]
    )
    return "\n".join(lines)


def build_latex(coefficients: pd.DataFrame, models: pd.DataFrame) -> str:
    return "\n\n".join(
        [
            r"\begingroup\scriptsize\renewcommand{\arraystretch}{0.92}",
            _sector_panel(coefficients, models, "business", "A"),
            r"\vspace{1.0em}",
            _sector_panel(coefficients, models, "household", "B"),
            r"\endgroup",
        ]
    )


def main() -> None:
    coefficients = pd.read_csv(COEFFICIENTS_PATH)
    models = pd.read_csv(MODELS_PATH)
    LATEX_PATH.write_text(build_latex(coefficients, models), encoding="utf-8")
    print(f"Wrote formatted Table 4 to {LATEX_PATH}")


if __name__ == "__main__":
    main()
