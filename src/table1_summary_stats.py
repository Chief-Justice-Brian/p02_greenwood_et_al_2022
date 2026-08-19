"""Generate the Table 1 replication and divergence report."""

from pathlib import Path

import pandas as pd

import paper_benchmarks as bench
from build_analysis_panel import load_analysis_panel
from settings import config

DATA_DIR = Path(config("DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))

ROW_LABELS = {
    "crisis_bvx": "Baron, Verner and Xiong (2019) (\\%)",
    "crisis_jst": "Schularick and Taylor (2012) (\\%)",
    "crisis_rr": "Reinhart and Rogoff (2009) (\\%)",
    "bank_equity_crash": "Bank Equity Crash (\\%)",
    "bank_failures": "Bank Failures (\\%)",
    "banking_panic": "Panics (\\%)",
    "real_gdp_growth": "$\\Delta_1$ log real GDP (\\%)",
    "delta3_business_debt_gdp": "$\\Delta_3$ Business Debt / GDP (\\%)",
    "delta3_household_debt_gdp": "$\\Delta_3$ Household Debt / GDP (\\%)",
    "delta3_log_real_private_debt": "$\\Delta_3$ log real Debt (\\%)",
    "delta3_log_real_equity": "$\\Delta_3$ log real Equity Index (\\%)",
    "delta3_log_real_house_price": "$\\Delta_3$ log real House Price Index (\\%)",
}

PERCENT_ROWS = {
    "crisis_bvx",
    "crisis_jst",
    "crisis_rr",
    "bank_equity_crash",
    "bank_failures",
    "banking_panic",
}

BUSINESS_ROWS = {
    "delta3_business_debt_gdp",
    "delta3_log_real_equity",
}

HOUSEHOLD_ROWS = {
    "delta3_household_debt_gdp",
    "delta3_log_real_house_price",
}


def _row_sample(sample, column):
    """Use the same complete-pair sample rule as the corresponding paper row.

    :param sample: analysis-panel rows already restricted to the paper sample.
    :param column: Table 1 variable name deciding which sector filter applies.
    """
    if column in BUSINESS_ROWS:
        return sample[sample["in_business_sample"]]
    if column in HOUSEHOLD_ROWS:
        return sample[sample["in_household_sample"]]
    return sample


def calculate_table1(panel):
    """Compare replicated Table 1 counts, means, and SDs to the published rows.

    :param panel: full country-year analysis panel from load_analysis_panel.
    :returns: one row per Table 1 variable with replicated and paper values
        plus their differences; crisis-indicator rows are scaled to percent.
    """
    sample = panel[panel["in_paper_sample"]].copy()
    records = []
    for column in bench.TABLE1_PUBLISHED_ROWS:
        values = _row_sample(sample, column)[column].dropna().astype(float)
        scale = 100.0 if column in PERCENT_ROWS else 1.0
        records.append(
            {
                "variable": column,
                "label": ROW_LABELS[column],
                "replicated_n": len(values),
                "replicated_mean": scale * values.mean(),
                "replicated_sd": scale * values.std(ddof=1),
                "paper_n": bench.TABLE1_PUBLISHED_ROWS[column][0],
                "paper_mean": bench.TABLE1_PUBLISHED_ROWS[column][1],
                "paper_sd": bench.TABLE1_PUBLISHED_ROWS[column][2],
            }
        )
    result = pd.DataFrame(records)
    result["n_difference"] = result["replicated_n"] - result["paper_n"]
    result["mean_difference"] = result["replicated_mean"] - result["paper_mean"]
    result["sd_difference"] = result["replicated_sd"] - result["paper_sd"]
    return result


def calculate_quantile_comparison(panel):
    """Recompute every quantile displayed in the bottom of the paper's table.

    :param panel: full country-year analysis panel from load_analysis_panel.
    """
    sample = panel[panel["in_paper_sample"]]
    specifications = []

    debt_specs = [
        (
            "delta3_business_debt_gdp",
            "Business debt",
            sample.loc[sample["in_business_sample"], "delta3_business_debt_gdp"],
            bench.DEBT_GROWTH_QUANTILES["business"],
        ),
        (
            "delta3_household_debt_gdp",
            "Household debt",
            sample.loc[sample["in_household_sample"], "delta3_household_debt_gdp"],
            bench.DEBT_GROWTH_QUANTILES["household"],
        ),
        (
            "delta3_log_real_private_debt",
            "Real private debt",
            sample["delta3_log_real_private_debt"],
            bench.PRIVATE_DEBT_LOG_QUANTILES,
        ),
    ]
    for variable, label, values, published_quantiles in debt_specs:
        for quantile, published in published_quantiles.items():
            specifications.append(
                (variable, label, values, quantile, published, quantile == 0.8)
            )

    price_specs = [
        (
            "delta3_log_real_equity",
            "Equity growth",
            sample.loc[sample["in_business_sample"], "delta3_log_real_equity"],
            bench.PRICE_GROWTH_TERCILES["equity"],
        ),
        (
            "delta3_log_real_house_price",
            "House-price growth",
            sample.loc[sample["in_household_sample"], "delta3_log_real_house_price"],
            bench.PRICE_GROWTH_TERCILES["house_price"],
        ),
    ]
    for variable, label, values, published_quantiles in price_specs:
        for quantile, published in published_quantiles.items():
            specifications.append(
                (variable, label, values, quantile, published, quantile == 2 / 3)
            )

    rows = []
    for variable, label, values, quantile, paper, is_rzone_gate in specifications:
        replicated = values.dropna().quantile(quantile)
        rows.append(
            {
                "variable": variable,
                "label": label,
                "quantile": quantile,
                "quantile_label": (
                    f"Q{quantile * 100:.1f}"
                    if quantile in {1 / 3, 2 / 3}
                    else f"Q{int(quantile * 100)}"
                ),
                "replicated": replicated,
                "paper": paper,
                "difference": replicated - paper,
                "is_rzone_gate": is_rzone_gate,
            }
        )
    return pd.DataFrame(rows)


def calculate_cutoff_comparison(panel):
    """Compatibility view containing only the four R-zone assignment gates.

    :param panel: full country-year analysis panel from load_analysis_panel.
    """
    return calculate_quantile_comparison(panel).query("is_rzone_gate").copy()


def _latex_fragment(frame, columns, headers, float_format="%.2f"):
    """Render the selected frame columns as a booktabs LaTeX table string.

    :param frame: DataFrame holding the rows to render.
    :param columns: frame columns to keep, in display order.
    :param headers: replacement column headers, aligned with columns.
    :param float_format: printf-style format applied to float cells.
    """
    display = frame[columns].copy()
    display.columns = headers
    return display.to_latex(
        index=False,
        escape=False,
        na_rep="--",
        float_format=float_format,
        column_format="l" + "r" * (len(columns) - 1),
    )


def _format_number(value):
    """Format a value to two decimals, rendering NaN as an empty string.

    :param value: numeric scalar to render; may be NaN or None for a blank cell.
    """
    if pd.isna(value):
        return ""
    return f"{value:.2f}"


def _summary_lookup(summary):
    """Index the summary frame by variable name as a dict of row dicts.

    :param summary: calculate_table1 output, one row per Table 1 variable.
    """
    return summary.set_index("variable").to_dict(orient="index")


def _quantile_lookup(quantiles):
    """Map (variable, quantile) pairs to their replicated cutoff values.

    :param quantiles: calculate_quantile_comparison output, one row per cutoff.
    """
    return {
        (row.variable, row.quantile): row.replicated
        for row in quantiles.itertuples(index=False)
    }


def _data_row(summary_rows, quantile_rows, variable, quantiles=()):
    """Build one LaTeX body row: label, N, mean, SD, then optional quantile cells.

    :param summary_rows: variable -> summary record, from _summary_lookup.
    :param quantile_rows: (variable, quantile) -> cutoff, from _quantile_lookup.
    :param variable: panel column name of the Table 1 row being rendered.
    :param quantiles: quantile levels to append; None entries render as empty
        cells.
    """
    row = summary_rows[variable]
    values = [
        ROW_LABELS[variable],
        f"{int(row['replicated_n'])}",
        _format_number(row["replicated_mean"]),
        _format_number(row["replicated_sd"]),
    ]
    values.extend(
        _format_number(quantile_rows.get((variable, quantile)))
        for quantile in quantiles
    )
    values.extend([""] * (8 - len(values)))
    return " & ".join(values) + r" \\"


def build_paper_style_table(summary, quantiles):
    """Render the replication using the grouped layout of GHSS Table 1.

    :param summary: output of calculate_table1.
    :param quantiles: output of calculate_quantile_comparison.
    """
    summary_rows = _summary_lookup(summary)
    quantile_rows = _quantile_lookup(quantiles)
    lines = [
        r"\begin{tabular}{rrrr@{\hspace{1.4em}}rrrr}",
        r"\toprule",
        r" & $N$ & Mean & SD & & & & \\",
        r"\cmidrule(lr){2-4}",
        r"\multicolumn{1}{r}{\underline{Financial Crisis Indicators:}}"
        r" & & & & & & & \\",
    ]
    for variable in ["crisis_bvx", "crisis_jst", "crisis_rr"]:
        lines.append(_data_row(summary_rows, quantile_rows, variable))

    lines.extend(
        [
            r"\addlinespace",
            r"\multicolumn{1}{r}{\underline{Crashes, Failures and Panics:}}"
            r" & & & & & & & \\",
        ]
    )
    for variable in ["bank_equity_crash", "bank_failures", "banking_panic"]:
        lines.append(_data_row(summary_rows, quantile_rows, variable))

    lines.extend(
        [
            r"\addlinespace",
            r"\multicolumn{1}{r}{\underline{GDP:}} & & & & & & & \\",
            _data_row(summary_rows, quantile_rows, "real_gdp_growth"),
            r"\addlinespace",
            r" & & & & \multicolumn{4}{c}{Quantiles} \\",
            r"\multicolumn{1}{r}{\underline{Debt Growth:}}"
            r" & & & & Q20 & Q40 & Q60 & Q80 \\",
            r"\cmidrule(lr){5-8}",
        ]
    )
    for variable in [
        "delta3_business_debt_gdp",
        "delta3_household_debt_gdp",
        "delta3_log_real_private_debt",
    ]:
        lines.append(
            _data_row(summary_rows, quantile_rows, variable, (0.2, 0.4, 0.6, 0.8))
        )

    lines.extend(
        [
            r"\addlinespace",
            r"\multicolumn{1}{r}{\underline{Price Growth:}}"
            r" & & & & & Q33.3 & Q66.7 & \\",
            r"\cmidrule(lr){5-8}",
        ]
    )
    for variable in ["delta3_log_real_equity", "delta3_log_real_house_price"]:
        # Empty cells place the two price terciles beneath the first and third
        # of the four quantile columns, matching the original table's layout.
        values = _data_row(
            summary_rows, quantile_rows, variable, (None, 1 / 3, 2 / 3, None)
        )
        lines.append(values)

    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    panel = load_analysis_panel(DATA_DIR)
    summary = calculate_table1(panel)
    quantiles = calculate_quantile_comparison(panel)
    summary.to_csv(OUTPUT_DIR / "table1_stats.csv", index=False)
    quantiles.to_csv(OUTPUT_DIR / "table1_cutoffs.csv", index=False)

    summary_tex = build_paper_style_table(summary, quantiles)
    (OUTPUT_DIR / "table1_summary_stats.tex").write_text(summary_tex)

    comparison = summary[
        ["label", "replicated_mean", "paper_mean", "mean_difference"]
    ].copy()
    comparison_tex = _latex_fragment(
        comparison,
        ["label", "replicated_mean", "paper_mean", "mean_difference"],
        ["Variable", "Replication", "Paper", "Difference"],
    )
    quantile_tex = _latex_fragment(
        quantiles,
        ["label", "quantile_label", "replicated", "paper", "difference"],
        ["Variable", "Quantile", "Replication", "Paper", "Difference"],
    )
    (OUTPUT_DIR / "table1_comparison.tex").write_text(
        comparison_tex + "\n\\par\\medskip\n" + quantile_tex
    )
    print(f"Saved Table 1 validation artifacts to {OUTPUT_DIR}")
