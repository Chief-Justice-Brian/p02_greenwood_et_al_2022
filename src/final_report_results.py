"""Create compact LaTeX fragments for the project's final narrative report."""

from pathlib import Path

import pandas as pd

from settings import config

OUTPUT_DIR = Path(config("OUTPUT_DIR"))


def _stars(p_value: float) -> str:
    """Return significance stars at the 1/5/10 percent levels.

    :param p_value: two-sided p-value; NaN yields no stars.
    :returns: the significance stars.
    """
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
    """Format a point estimate with LaTeX superscript significance stars.

    :param value: point estimate, printed with one decimal place.
    :param p_value: two-sided p-value that determines the stars.
    """
    stars = _stars(p_value)
    return f"{value:.1f}" + (rf"\textsuperscript{{{stars}}}" if stars else "")


def validation_summary() -> str:
    """Build the LaTeX tabular of per-exhibit benchmark pass rates.

    Reads replication_validation.csv.

    :returns: the tabular as a string.
    """
    validation = pd.read_csv(OUTPUT_DIR / "replication_validation.csv")
    summary = (
        validation.groupby("exhibit", sort=False)["within_tolerance"]
        .agg(["sum", "count"])
        .reset_index()
    )
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        r"Exhibit & Benchmarks & Within tolerance & Pass rate (\%) \\",
        r"\midrule",
    ]
    for row in summary.itertuples(index=False):
        lines.append(
            f"{row.exhibit} & {row.count} & {row.sum} & "
            f"{100.0 * row.sum / row.count:.1f} \\\\"
        )
    total = len(validation)
    passed = int(validation["within_tolerance"].sum())
    failed = total - passed
    complete_exhibits = summary.loc[summary["sum"].eq(summary["count"]), "exhibit"]
    complete_text = " and ".join(complete_exhibits.tolist())
    lines.extend(
        [
            r"\midrule",
            f"Total & {total} & {passed} & {100.0 * passed / total:.1f} \\\\ ",
            r"\bottomrule",
            r"\end{tabular}",
            "",
            rf"\par\smallskip\noindent The paper-scaled audit passes {passed} of {total} comparisons and flags {failed}. {complete_text} pass completely. The remaining misses identify where the public-data reconstruction does not recover the paper's cell-level probabilities, inferential statistics, or model fit within the paper-derived bounds; they are reported rather than used to widen the tolerances.",
        ]
    )
    return "\n".join(lines) + "\n"


def fragility_summary() -> str:
    """Build the LaTeX tabular of horizon-3 fragility interaction estimates.

    Shows the high-fragility and R-Zone x fragility coefficients plus the
    baseline and extended within-R2 for each sector and measure.
    """
    coefficients = pd.read_csv(OUTPUT_DIR / "fragility_regression_coefficients.csv")
    models = pd.read_csv(OUTPUT_DIR / "fragility_regression_models.csv")
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Sector & Measure & High fragility & R-Zone $\times$ fragility & $R^2_0$ & $R^2_1$ & $N$ \\",
        r"\midrule",
    ]
    for sector in ["business", "household"]:
        for measure in ["noncore", "ltd", "lev"]:
            subset = coefficients.loc[
                coefficients["sector"].eq(sector)
                & coefficients["fragility_measure"].eq(measure)
                & coefficients["horizon"].eq(3)
                & coefficients["specification"].eq("fragility_extension")
            ]
            high = subset.loc[subset["variable"].eq(f"high_{measure}")].iloc[0]
            interaction = subset.loc[
                subset["variable"].eq(f"rzone_{sector}_x_high_{measure}")
            ].iloc[0]
            model_subset = models.loc[
                models["sector"].eq(sector)
                & models["fragility_measure"].eq(measure)
                & models["horizon"].eq(3)
            ]
            base = model_subset.loc[
                model_subset["specification"].eq("ghss_same_sample")
            ].iloc[0]
            extended = model_subset.loc[
                model_subset["specification"].eq("fragility_extension")
            ].iloc[0]
            lines.append(
                " & ".join(
                    [
                        sector.title(),
                        {
                            "noncore": "Noncore",
                            "ltd": "Loans/deposits",
                            "lev": "Leverage",
                        }[measure],
                        _estimate(high["coefficient_pp"], high["p_value"]),
                        _estimate(
                            interaction["coefficient_pp"], interaction["p_value"]
                        ),
                        f"{base['within_r2_pct']:.1f}",
                        f"{extended['within_r2_pct']:.1f}",
                        f"{int(extended['n']):,}",
                    ]
                )
                + r" \\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def fragility_continuous_summary() -> str:
    """Build the LaTeX tabular of horizon-3 continuous fragility estimates.

    Shows the standardized-level and R-Zone x level coefficients plus the
    baseline and extended within-R2 for each sector and measure, mirroring
    the flag-based table so the two are read side by side.
    """
    coefficients = pd.read_csv(OUTPUT_DIR / "fragility_continuous_coefficients.csv")
    models = pd.read_csv(OUTPUT_DIR / "fragility_continuous_models.csv")
    lines = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Sector & Measure & Fragility level & R-Zone $\times$ level & $R^2_0$ & $R^2_1$ & $N$ \\",
        r"\midrule",
    ]
    for sector in ["business", "household"]:
        for measure in ["noncore", "ltd", "lev"]:
            subset = coefficients.loc[
                coefficients["sector"].eq(sector)
                & coefficients["fragility_measure"].eq(measure)
                & coefficients["horizon"].eq(3)
                & coefficients["specification"].eq("continuous_extension")
            ]
            level = subset.loc[subset["variable"].eq(f"{measure}_fragility_z")].iloc[0]
            interaction = subset.loc[
                subset["variable"].eq(f"rzone_{sector}_x_{measure}_fragility_z")
            ].iloc[0]
            model_subset = models.loc[
                models["sector"].eq(sector)
                & models["fragility_measure"].eq(measure)
                & models["horizon"].eq(3)
            ]
            base = model_subset.loc[
                model_subset["specification"].eq("ghss_same_sample")
            ].iloc[0]
            extended = model_subset.loc[
                model_subset["specification"].eq("continuous_extension")
            ].iloc[0]
            lines.append(
                " & ".join(
                    [
                        sector.title(),
                        {
                            "noncore": "Noncore",
                            "ltd": "Loans/deposits",
                            "lev": "Leverage",
                        }[measure],
                        _estimate(level["coefficient_pp"], level["p_value"]),
                        _estimate(
                            interaction["coefficient_pp"], interaction["p_value"]
                        ),
                        f"{base['within_r2_pct']:.1f}",
                        f"{extended['within_r2_pct']:.1f}",
                        f"{int(extended['n']):,}",
                    ]
                )
                + r" \\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def fragility_form_checks_summary() -> str:
    """Build the LaTeX tabular of the two noncore functional-form checks.

    Shows the level and 3-year-change coefficients from their joint race and
    the quadratic term from the curvature check, per sector and horizon.
    """
    coefficients = pd.read_csv(OUTPUT_DIR / "fragility_form_checks_coefficients.csv")
    lines = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Sector & Horizon & Level & 3-year change & Quadratic \\",
        r"\midrule",
    ]
    for sector in ["business", "household"]:
        for horizon in [1, 2, 3, 4]:
            race = coefficients.loc[
                coefficients["sector"].eq(sector)
                & coefficients["horizon"].eq(horizon)
                & coefficients["specification"].eq("level_and_change")
            ]
            level = race.loc[race["variable"].eq("noncore_fragility_z")].iloc[0]
            change = race.loc[race["variable"].eq("delta3_noncore_fragility_z")].iloc[0]
            quadratic = coefficients.loc[
                coefficients["sector"].eq(sector)
                & coefficients["horizon"].eq(horizon)
                & coefficients["specification"].eq("level_quadratic")
                & coefficients["variable"].eq("noncore_fragility_z_sq")
            ].iloc[0]
            lines.append(
                " & ".join(
                    [
                        sector.title() if horizon == 1 else "",
                        str(horizon),
                        _estimate(level["coefficient_pp"], level["p_value"]),
                        _estimate(change["coefficient_pp"], change["p_value"]),
                        _estimate(quadratic["coefficient_pp"], quadratic["p_value"]),
                    ]
                )
                + r" \\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def rzone_exit_summary() -> str:
    """Build the LaTeX tabular of post-exit R-Zone risk estimates.

    Shows the current-zone and pooled recent-exit coefficients plus the
    mutually exclusive years-since-exit buckets on all origins, per sector
    and horizon, beside the pooled recent-exit coefficient re-estimated on
    origins with no BVX crisis in the current or three prior years.
    """
    coefficients = pd.read_csv(OUTPUT_DIR / "rzone_exit_coefficients.csv")
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Sector & Horizon & In zone & Recent exit & Exit $-1$y & Exit $-2$y & Exit $-3$y & No-crisis exit \\",
        r"\midrule",
    ]
    for sector in ["business", "household"]:
        zone = f"rzone_{sector}"
        for horizon in [1, 2, 3, 4]:
            subset = coefficients.loc[
                coefficients["sector"].eq(sector) & coefficients["horizon"].eq(horizon)
            ]
            pooled = subset.loc[
                subset["sample"].eq("all_origins")
                & subset["specification"].eq("recent_exit")
            ]
            in_zone = pooled.loc[pooled["variable"].eq(zone)].iloc[0]
            recent = pooled.loc[pooled["variable"].eq(f"{zone}_recent_exit")].iloc[0]
            by_year = subset.loc[
                subset["sample"].eq("all_origins")
                & subset["specification"].eq("exit_by_year")
            ]
            buckets = [
                by_year.loc[by_year["variable"].eq(f"{zone}_exited_{k}y")].iloc[0]
                for k in [1, 2, 3]
            ]
            clean = subset.loc[
                subset["sample"].eq("no_recent_crisis")
                & subset["specification"].eq("recent_exit")
                & subset["variable"].eq(f"{zone}_recent_exit")
            ].iloc[0]
            lines.append(
                " & ".join(
                    [
                        sector.title() if horizon == 1 else "",
                        str(horizon),
                        _estimate(in_zone["coefficient_pp"], in_zone["p_value"]),
                        _estimate(recent["coefficient_pp"], recent["p_value"]),
                        *[
                            _estimate(bucket["coefficient_pp"], bucket["p_value"])
                            for bucket in buckets
                        ],
                        _estimate(clean["coefficient_pp"], clean["p_value"]),
                    ]
                )
                + r" \\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def footnote10_verdict() -> str:
    """Build the LaTeX tabular of the precommitted footnote-10 verdict.

    Compares the static and dynamic R-Zone coefficients per sector and
    horizon and marks whether the dynamic estimate sits within one static
    standard error, the tightest of the three precommitted criteria.
    """
    coefficients = pd.read_csv(OUTPUT_DIR / "dynamic_regression_coefficients.csv")
    rzone_rows = coefficients.loc[coefficients["variable"].str.startswith("rzone")]
    lines = [
        r"\begin{tabular}{lrrrc}",
        r"\toprule",
        r"Sector & Horizon & Static $\gamma^{(h)}$ & Dynamic $\gamma^{(h)}$ & Within 1 s.e. \\",
        r"\midrule",
    ]
    for sector in ["business", "household"]:
        for horizon in [1, 2, 3, 4]:
            subset = rzone_rows.loc[
                rzone_rows["sector"].eq(sector) & rzone_rows["horizon"].eq(horizon)
            ]
            static = subset.loc[subset["specification"].eq("ghss_same_sample")].iloc[0]
            dynamic = subset.loc[subset["specification"].eq("autoregressive")].iloc[0]
            within_one_se = (
                abs(dynamic["coefficient_pp"] - static["coefficient_pp"])
                <= static["std_error_pp"]
            )
            lines.append(
                " & ".join(
                    [
                        sector.title() if horizon == 1 else "",
                        str(horizon),
                        _estimate(static["coefficient_pp"], static["p_value"]),
                        _estimate(dynamic["coefficient_pp"], dynamic["p_value"]),
                        "Yes" if within_one_se else "No",
                    ]
                )
                + r" \\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def dynamic_summary() -> str:
    """Build the LaTeX tabular racing the static and autoregressive models.

    Reports the R-Zone coefficient, within-R2, and in-sample AUC for both
    specifications at every sector and horizon.
    """
    coefficients = pd.read_csv(OUTPUT_DIR / "dynamic_regression_coefficients.csv")
    models = pd.read_csv(OUTPUT_DIR / "dynamic_regression_models.csv")
    lines = [
        r"\begin{tabular}{lrrrrrrr}",
        r"\toprule",
        r"Sector / horizon & Static R-Zone & Dynamic R-Zone & $R^2_0$ & $R^2_1$ & AUC$_0$ & AUC$_1$ & $N$ \\",
        r"\midrule",
    ]
    for sector in ["business", "household"]:
        for horizon in range(1, 5):
            coefficient_subset = coefficients.loc[
                coefficients["sector"].eq(sector)
                & coefficients["horizon"].eq(horizon)
                & coefficients["variable"].eq(f"rzone_{sector}")
            ]
            static_coefficient = coefficient_subset.loc[
                coefficient_subset["specification"].eq("ghss_same_sample")
            ].iloc[0]
            dynamic_coefficient = coefficient_subset.loc[
                coefficient_subset["specification"].eq("autoregressive")
            ].iloc[0]
            model_subset = models.loc[
                models["sector"].eq(sector) & models["horizon"].eq(horizon)
            ]
            static = model_subset.loc[
                model_subset["specification"].eq("ghss_same_sample")
            ].iloc[0]
            dynamic = model_subset.loc[
                model_subset["specification"].eq("autoregressive")
            ].iloc[0]
            lines.append(
                " & ".join(
                    [
                        f"{sector.title()} / {horizon}y",
                        _estimate(
                            static_coefficient["coefficient_pp"],
                            static_coefficient["p_value"],
                        ),
                        _estimate(
                            dynamic_coefficient["coefficient_pp"],
                            dynamic_coefficient["p_value"],
                        ),
                        f"{static['within_r2_pct']:.1f}",
                        f"{dynamic['within_r2_pct']:.1f}",
                        f"{static['in_sample_auc']:.3f}",
                        f"{dynamic['in_sample_auc']:.3f}",
                        f"{int(dynamic['n']):,}",
                    ]
                )
                + r" \\"
            )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def missed_crises() -> str:
    """Build the LaTeX longtable of crises neither R-Zone preceded.

    Lists each missed crisis with its prior-three-year bank-fragility
    percentiles; dashes mark unavailable JST banking data.
    """
    data = pd.read_csv(OUTPUT_DIR / "missed_crisis_fragility.csv")
    data = data.loc[data["missed_by_rzone"]].copy()
    lines = [
        r"\begin{longtable}{lrrrr}",
        r"\caption{\textbf{Missed crises and prior bank-fragility percentiles.} The table lists every crisis not preceded by either sector's R-Zone and the maximum bank-fragility percentile during the prior three years. Values above 80 indicate unusually fragile banking conditions. The mixed pattern shows why fragility is a diagnostic complement rather than a complete replacement for the borrower-side signal; dashes indicate unavailable JST banking data.}\label{tab:missed-crises} \\",
        r"\toprule",
        r"Country & Crisis year & Noncore pct. & Loans/deposits pct. & Leverage pct. \\",
        r"\midrule",
        r"\endfirsthead",
        r"\toprule",
        r"Country & Crisis year & Noncore pct. & Loans/deposits pct. & Leverage pct. \\",
        r"\midrule",
        r"\endhead",
    ]

    def percentile(value: float) -> str:
        """Format a percentile, printing a dash when the value is missing.

        :param value: percentile on the 0-100 scale, or NaN when unavailable.
        """
        return "--" if pd.isna(value) else f"{value:.1f}"

    for row in data.itertuples(index=False):
        country = row.country.replace("&", r"\&")
        lines.append(
            f"{country} & {row.crisis_year} & "
            f"{percentile(row.prior_noncore_fragility_pct)} & "
            f"{percentile(row.prior_ltd_fragility_pct)} & "
            f"{percentile(row.prior_lev_fragility_pct)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{longtable}"])
    return "\n".join(lines) + "\n"


def tracker_summary() -> str:
    """Build the LaTeX tabular of post-2016 annual R-Zone counts."""
    tracker = pd.read_csv(OUTPUT_DIR / "post_2016_rzone_tracker.csv")
    rows = []
    for year in range(2017, int(tracker["year"].max()) + 1):
        year_data = tracker.loc[tracker["year"].eq(year)]
        rows.append(
            {
                "year": year,
                "business_n": year_data["rzone_business"].notna().sum(),
                "business_rzones": year_data["rzone_business"].sum(min_count=1),
                "household_n": year_data["rzone_household"].notna().sum(),
                "household_rzones": year_data["rzone_household"].sum(min_count=1),
                "either_rzones": year_data["rzone_either"].sum(min_count=1),
            }
        )
    lines = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Year & Business $N$ & Business R-Zones & Household $N$ & Household R-Zones & Either \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['year']} & {row['business_n']} & {int(row['business_rzones'])} & "
            f"{row['household_n']} & {int(row['household_rzones'])} & "
            f"{int(row['either_rzones'])} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def usa_case_study() -> str:
    """Build the LaTeX tabular of the US 2019-2023 predictor and R-Zone path."""
    data = pd.read_csv(OUTPUT_DIR / "usa_2023_case_study.csv")
    lines = [
        r"\begin{tabular}{rrrrrrr}",
        r"\toprule",
        r"Year & Business debt & Equity & Bus. R-Zone & Household debt & House price & HH R-Zone \\",
        r"\midrule",
    ]
    for row in data.itertuples(index=False):
        lines.append(
            f"{row.year} & {row.delta3_business_debt_gdp:.1f} & "
            f"{row.delta3_log_real_equity:.1f} & {int(row.rzone_business)} & "
            f"{row.delta3_household_debt_gdp:.1f} & "
            f"{row.delta3_log_real_house_price:.1f} & "
            f"{int(row.rzone_household)} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    return "\n".join(lines) + "\n"


def main() -> None:
    """Write every final-report LaTeX fragment to OUTPUT_DIR."""
    fragments = {
        "report_validation_summary.tex": validation_summary(),
        "report_fragility_summary.tex": fragility_summary(),
        "report_fragility_continuous.tex": fragility_continuous_summary(),
        "report_fragility_form_checks.tex": fragility_form_checks_summary(),
        "report_rzone_exit.tex": rzone_exit_summary(),
        "report_footnote10_verdict.tex": footnote10_verdict(),
        "report_dynamic_summary.tex": dynamic_summary(),
        "report_missed_crises.tex": missed_crises(),
        "report_tracker_summary.tex": tracker_summary(),
        "report_usa_case_study.tex": usa_case_study(),
    }
    for filename, content in fragments.items():
        (OUTPUT_DIR / filename).write_text(content, encoding="utf-8")
    print("Saved final-report LaTeX fragments")


if __name__ == "__main__":
    main()
