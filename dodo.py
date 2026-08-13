"""Run or update the project. This file uses the `doit` Python package. It works
like a Makefile, but is Python-based.

Pipeline stages (each stage only depends on the previous one):
1. config     -- create _data/_output directories
2. pull       -- download every raw data source (all free, no API keys):
                 BVX crisis chronology, JST Macrohistory, BIS total credit,
                 BIS property prices, IMF Global Debt Database, IMF share
                 prices, OECD share prices, OECD house prices, World Bank WDI

"""

#######################################
## Configuration and Helpers for PyDoit
#######################################
## Make sure the src folder is in the path
import sys

sys.path.insert(1, "./src/")

import shutil
import subprocess
from os import environ
from pathlib import Path

from settings import config

DOIT_CONFIG = {"backend": "sqlite3", "dep_file": "./.doit-db.sqlite"}


BASE_DIR = config("BASE_DIR")
DATA_DIR = config("DATA_DIR")
MANUAL_DATA_DIR = config("MANUAL_DATA_DIR")
OUTPUT_DIR = config("OUTPUT_DIR")
OS_TYPE = config("OS_TYPE")

## Helpers for handling Jupyter Notebook tasks
environ["PYDEVD_DISABLE_FILE_VALIDATION"] = "1"
environ["JUPYTER_DATA_DIR"] = str(OUTPUT_DIR / ".jupyter")
environ["JUPYTER_RUNTIME_DIR"] = str(OUTPUT_DIR / ".jupyter" / "runtime")
environ["IPYTHONDIR"] = str(OUTPUT_DIR / ".ipython")


# fmt: off
## Helper functions for automatic execution of Jupyter notebooks
def jupyter_execute_notebook(notebook_path):
    return f"jupyter nbconvert --execute --to notebook --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"
def jupyter_to_html(notebook_path, output_dir=OUTPUT_DIR):
    return f"jupyter nbconvert --to html --output-dir={output_dir} {notebook_path}"
def jupyter_to_md(notebook_path, output_dir=OUTPUT_DIR):
    """Requires jupytext"""
    return f"jupytext --to markdown --output-dir={output_dir} {notebook_path}"
def jupyter_clear_output(notebook_path):
    """Clear the output of a notebook"""
    return f"jupyter nbconvert --ClearOutputPreprocessor.enabled=True --ClearMetadataPreprocessor.enabled=True --inplace {notebook_path}"
# fmt: on


def mv(from_path, to_path):
    """Move a file to a folder"""
    from_path = Path(from_path)
    to_path = Path(to_path)
    to_path.mkdir(parents=True, exist_ok=True)
    if OS_TYPE == "nix":
        command = f"mv {from_path} {to_path}"
    else:
        command = f"move {from_path} {to_path}"
    return command


def copy_file(origin_path, destination_path, mkdir=True):
    """Create a Python action for copying a file."""

    def _copy_file():
        origin = Path(origin_path)
        dest = Path(destination_path)
        if mkdir:
            dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(origin, dest)

    return _copy_file


##################################
## Begin rest of PyDoit tasks here
##################################


def task_config():
    """Create empty directories for data and output if they don't exist"""
    return {
        "actions": ["python ./src/settings.py"],
        "targets": [DATA_DIR, OUTPUT_DIR],
        "file_dep": ["./src/settings.py"],
        "clean": [],
    }


# One entry per data source: script name -> the parquet files it produces.
# No raw data in the repo; everything regenerates from `doit`).
pull_tasks = {
    "bvx_crises": {
        "script": "pull_bvx_crises.py",
        "doc": "BVX (2021) banking-crisis chronology from Harvard Dataverse",
        "outputs": [
            "bvx_crisis_list.parquet",
            "bvx_annual_regdata.parquet",
        ],
    },
    "jst_macrohistory": {
        "script": "pull_jst_macrohistory.py",
        "doc": "Jorda-Schularick-Taylor Macrohistory panel (R6)",
        "outputs": ["jst_macrohistory.parquet"],
    },
    "bis_total_credit": {
        "script": "pull_bis_total_credit.py",
        "doc": "BIS total credit to households and non-financial corporates",
        "outputs": ["bis_total_credit.parquet"],
    },
    "bis_property_prices": {
        "script": "pull_bis_property_prices.py",
        "doc": "BIS selected residential property prices",
        "outputs": ["bis_property_prices.parquet"],
    },
    "imf_gdd": {
        "script": "pull_imf_gdd.py",
        "doc": "IMF Global Debt Database (household/corporate debt to GDP)",
        "outputs": ["imf_gdd.parquet"],
    },
    "imf_equity": {
        "script": "pull_imf_equity.py",
        "doc": "IMF (former IFS) share price indices, 1950-2016",
        "outputs": ["imf_equity.parquet"],
    },
    "oecd_share_prices": {
        "script": "pull_oecd_share_prices.py",
        "doc": "OECD share price indices, 1950-present",
        "outputs": ["oecd_share_prices.parquet"],
    },
    "oecd_house_prices": {
        "script": "pull_oecd_house_prices.py",
        "doc": "OECD analytical house price indicators",
        "outputs": ["oecd_house_prices.parquet"],
    },
    "worldbank_wdi": {
        "script": "pull_worldbank_wdi.py",
        "doc": "World Bank WDI: CPI and nominal GDP",
        "outputs": ["worldbank_wdi.parquet"],
    },
}


def task_pull():
    """Pull data from external sources"""
    for task_name, task_info in pull_tasks.items():
        script_path = f"./src/{task_info['script']}"
        yield {
            "name": task_name,
            "doc": task_info["doc"],
            "actions": [f"python {script_path}"],
            "targets": [DATA_DIR / output for output in task_info["outputs"]],
            "file_dep": ["./src/settings.py", script_path],
            "clean": [],
        }


TIDY_HELPER_MODULES = [
    "./src/settings.py",
    "./src/country_sample.py",
    "./src/panel_utils.py",
    "./src/rzone_features.py",
]

# The tidy-data stage: each entry cleans pulled sources into one tidy panel.
# input_data/outputs are files in DATA_DIR; doit chains the stages via these
# file dependencies (pull -> deflators -> equity -> ... -> analysis panel).
clean_tasks = {
    "macro_deflators": {
        "script": "clean_macro_deflators.py",
        "doc": "Continuous log-CPI deflator panel (WDI + JST chained)",
        "input_data": ["worldbank_wdi.parquet", "jst_macrohistory.parquet"],
        "outputs": ["macro_deflators.parquet"],
    },
    "crisis_chronologies": {
        "script": "clean_crisis_chronologies.py",
        "doc": "Tidy BVX/JST/RR crisis indicator panel",
        "input_data": ["bvx_annual_regdata.parquet", "jst_macrohistory.parquet"],
        "outputs": ["crisis_panel.parquet"],
    },
    "credit_panel": {
        "script": "clean_credit_panel.py",
        "doc": "Spliced 3-year credit growth (GDD -> BIS -> JST)",
        "input_data": [
            "imf_gdd.parquet",
            "bis_total_credit.parquet",
            "jst_macrohistory.parquet",
            "worldbank_wdi.parquet",
            "macro_deflators.parquet",
        ],
        "outputs": ["credit_panel.parquet"],
    },
    "equity_panel": {
        "script": "clean_equity_panel.py",
        "doc": "Spliced 3-year real equity growth (IMF -> JST -> OECD)",
        "input_data": [
            "imf_equity.parquet",
            "jst_macrohistory.parquet",
            "oecd_share_prices.parquet",
            "macro_deflators.parquet",
        ],
        "outputs": ["equity_panel.parquet"],
    },
    "house_price_panel": {
        "script": "clean_house_price_panel.py",
        "doc": "Spliced 3-year real house price growth (BIS -> OECD -> JST)",
        "input_data": [
            "bis_property_prices.parquet",
            "oecd_house_prices.parquet",
            "jst_macrohistory.parquet",
        ],
        "outputs": ["house_price_panel.parquet"],
    },
    "analysis_panel": {
        "script": "build_analysis_panel.py",
        "doc": "Shared 42-country analysis panel with paper-sample flag",
        "input_data": [
            "crisis_panel.parquet",
            "credit_panel.parquet",
            "equity_panel.parquet",
            "house_price_panel.parquet",
        ],
        "outputs": ["rzone_analysis_panel.parquet"],
    },
}


def task_tidy():
    """Clean pulled data into tidy panels and build the shared analysis panel"""
    for task_name, task_info in clean_tasks.items():
        script_path = f"./src/{task_info['script']}"
        yield {
            "name": task_name,
            "doc": task_info["doc"],
            "actions": [f"python {script_path}"],
            "targets": [DATA_DIR / output for output in task_info["outputs"]],
            "file_dep": [
                script_path,
                *TIDY_HELPER_MODULES,
                *[DATA_DIR / input_file for input_file in task_info["input_data"]],
            ],
            "clean": [],
        }


def task_analysis():
    """Build the replication exhibits from the analysis panel"""
    yield {
        "name": "table1",
        "doc": "Table 1 summary statistics + divergence report vs the paper",
        "actions": ["python ./src/table1_summary_stats.py"],
        "targets": [
            OUTPUT_DIR / "table1_stats.csv",
            OUTPUT_DIR / "table1_cutoffs.csv",
            OUTPUT_DIR / "table1_summary_stats.tex",
            OUTPUT_DIR / "table1_comparison.tex",
        ],
        "file_dep": [
            "./src/table1_summary_stats.py",
            "./src/paper_benchmarks.py",
            "./src/settings.py",
            DATA_DIR / "rzone_analysis_panel.parquet",
        ],
        "clean": True,
    }
    yield {
        "name": "descriptive",
        "doc": "Raw quantile cells, R-zone validation, timeline, and modern tracker",
        "actions": ["python ./src/descriptive_results.py"],
        "targets": [
            OUTPUT_DIR / "table3_cell_frequencies_raw.csv",
            OUTPUT_DIR / "rzone_validation.csv",
            OUTPUT_DIR / "post_2016_rzone_tracker.csv",
            OUTPUT_DIR / "historical_business_rzone_timeline.png",
            OUTPUT_DIR / "historical_household_rzone_timeline.png",
            OUTPUT_DIR / "figure1_event_history.png",
            OUTPUT_DIR / "figure1_event_history.pdf",
        ],
        "file_dep": [
            "./src/descriptive_results.py",
            "./src/settings.py",
            DATA_DIR / "rzone_analysis_panel.parquet",
        ],
        "clean": True,
    }
    yield {
        "name": "table3",
        "doc": "Publication-style Table 3 cells, inference, and LaTeX",
        "actions": ["python ./src/table3_results.py"],
        "targets": [
            OUTPUT_DIR / "table3_crisis_probabilities.csv",
            OUTPUT_DIR / "table3_replication.tex",
        ],
        "file_dep": [
            "./src/table3_results.py",
            "./src/baseline_regressions.py",
            DATA_DIR / "rzone_analysis_panel.parquet",
        ],
        "clean": True,
    }
    yield {
        "name": "baseline_regressions",
        "doc": "Table 4 country-FE linear probability models with DK errors",
        "actions": ["python ./src/baseline_regressions.py"],
        "targets": [
            OUTPUT_DIR / "table4_baseline_coefficients.csv",
            OUTPUT_DIR / "table4_baseline_models.csv",
        ],
        "file_dep": [
            "./src/baseline_regressions.py",
            "./src/settings.py",
            DATA_DIR / "rzone_analysis_panel.parquet",
        ],
        "clean": True,
    }
    yield {
        "name": "table4",
        "doc": "Publication-style Table 4 regression LaTeX",
        "actions": ["python ./src/table4_results.py"],
        "targets": [OUTPUT_DIR / "table4_replication.tex"],
        "file_dep": [
            "./src/table4_results.py",
            OUTPUT_DIR / "table4_baseline_coefficients.csv",
            OUTPUT_DIR / "table4_baseline_models.csv",
        ],
        "clean": True,
    }
    yield {
        "name": "figure3",
        "doc": "Figure 3 annual fraction of countries in each R-Zone",
        "actions": ["python ./src/figure3_global_rzone.py"],
        "targets": [
            OUTPUT_DIR / "figure3_global_rzone.csv",
            OUTPUT_DIR / "figure3_fraction_countries_rzone.png",
            OUTPUT_DIR / "figure3_fraction_countries_rzone.pdf",
            OUTPUT_DIR / "figure3_fraction_countries_rzone.jpg",
        ],
        "file_dep": [
            "./src/figure3_global_rzone.py",
            "./src/settings.py",
            DATA_DIR / "rzone_analysis_panel.parquet",
        ],
        "clean": True,
    }
    yield {
        "name": "validation",
        "doc": "Tolerance-based validation of Tables 1/3/4 and Figures 1/3",
        "actions": ["python ./src/replication_validation.py"],
        "targets": [OUTPUT_DIR / "replication_validation.csv"],
        "file_dep": [
            "./src/replication_validation.py",
            "./src/exhibit_benchmarks.py",
            "./src/paper_benchmarks.py",
            OUTPUT_DIR / "table1_stats.csv",
            OUTPUT_DIR / "table1_cutoffs.csv",
            OUTPUT_DIR / "table3_crisis_probabilities.csv",
            OUTPUT_DIR / "table4_baseline_coefficients.csv",
            OUTPUT_DIR / "table4_baseline_models.csv",
            OUTPUT_DIR / "figure3_global_rzone.csv",
            DATA_DIR / "rzone_analysis_panel.parquet",
        ],
        "clean": True,
    }
    yield {
        "name": "extensions",
        "doc": "Fragility, missed-crisis, dynamic, and 2023 extension outputs",
        "actions": ["python ./src/extension_results.py"],
        "targets": [
            OUTPUT_DIR / "fragility_regression_coefficients.csv",
            OUTPUT_DIR / "fragility_regression_models.csv",
            OUTPUT_DIR / "dynamic_regression_coefficients.csv",
            OUTPUT_DIR / "dynamic_regression_models.csv",
            OUTPUT_DIR / "missed_crisis_fragility.csv",
            OUTPUT_DIR / "usa_2023_case_study.csv",
        ],
        "file_dep": [
            "./src/extension_results.py",
            "./src/baseline_regressions.py",
            "./src/settings.py",
            DATA_DIR / "rzone_analysis_panel.parquet",
        ],
        "clean": True,
    }
    yield {
        "name": "updated",
        "doc": "Updated Tables 1/3/4 and Figures 1/3 through latest valid year",
        "actions": ["python ./src/updated_results.py"],
        "targets": [
            OUTPUT_DIR / "updated_data_coverage.csv",
            OUTPUT_DIR / "table1_updated_stats.csv",
            OUTPUT_DIR / "table1_updated_quantiles.csv",
            OUTPUT_DIR / "table1_updated.tex",
            OUTPUT_DIR / "table3_updated_crisis_probabilities.csv",
            OUTPUT_DIR / "table3_updated.tex",
            OUTPUT_DIR / "table4_updated_coefficients.csv",
            OUTPUT_DIR / "table4_updated_models.csv",
            OUTPUT_DIR / "table4_updated.tex",
            OUTPUT_DIR / "updated_business_rzone_timeline.png",
            OUTPUT_DIR / "updated_household_rzone_timeline.png",
            OUTPUT_DIR / "figure1_event_history_updated.png",
            OUTPUT_DIR / "figure1_event_history_updated.pdf",
            OUTPUT_DIR / "figure3_global_rzone_updated.csv",
            OUTPUT_DIR / "figure3_fraction_countries_rzone_updated.png",
            OUTPUT_DIR / "figure3_fraction_countries_rzone_updated.pdf",
            OUTPUT_DIR / "figure3_fraction_countries_rzone_updated.jpg",
        ],
        "file_dep": [
            "./src/updated_results.py",
            "./src/updated_crisis_chronology.py",
            "./src/descriptive_results.py",
            "./src/figure3_global_rzone.py",
            "./src/table3_results.py",
            "./src/table4_results.py",
            "./src/baseline_regressions.py",
            DATA_DIR / "rzone_analysis_panel.parquet",
        ],
        "clean": True,
    }
    yield {
        "name": "data_overview",
        "doc": "Original summary table and diagnostic chart for underlying data",
        "actions": ["python ./src/data_overview.py"],
        "targets": [
            OUTPUT_DIR / "data_overview_summary.csv",
            OUTPUT_DIR / "data_overview_summary.tex",
            OUTPUT_DIR / "data_overview_figure.png",
            OUTPUT_DIR / "data_overview_figure.pdf",
            OUTPUT_DIR / "data_overview_figure.jpg",
        ],
        "file_dep": [
            "./src/data_overview.py",
            "./src/settings.py",
            DATA_DIR / "rzone_analysis_panel.parquet",
        ],
        "clean": True,
    }
    yield {
        "name": "final_report",
        "doc": "Build summary fragments for the single final LaTeX report",
        "actions": ["python ./src/final_report_results.py"],
        "targets": [
            OUTPUT_DIR / "report_validation_summary.tex",
            OUTPUT_DIR / "report_fragility_summary.tex",
            OUTPUT_DIR / "report_dynamic_summary.tex",
            OUTPUT_DIR / "report_missed_crises.tex",
            OUTPUT_DIR / "report_tracker_summary.tex",
            OUTPUT_DIR / "report_usa_case_study.tex",
        ],
        "file_dep": [
            "./src/final_report_results.py",
            OUTPUT_DIR / "replication_validation.csv",
            OUTPUT_DIR / "fragility_regression_coefficients.csv",
            OUTPUT_DIR / "fragility_regression_models.csv",
            OUTPUT_DIR / "dynamic_regression_coefficients.csv",
            OUTPUT_DIR / "dynamic_regression_models.csv",
            OUTPUT_DIR / "missed_crisis_fragility.csv",
            OUTPUT_DIR / "post_2016_rzone_tracker.csv",
            OUTPUT_DIR / "usa_2023_case_study.csv",
        ],
        "clean": True,
    }


def task_compile_latex():
    """Compile the LaTeX preview documents to PDFs"""
    yield {
        "name": "table1_preview",
        "doc": "Viewable PDF of the Table 1 replication + divergence report",
        "actions": [
            "latexmk -pdf -halt-on-error -cd ./reports/table1_preview.tex",
            "latexmk -pdf -halt-on-error -c -cd ./reports/table1_preview.tex",
        ],
        "targets": ["./reports/table1_preview.pdf"],
        "file_dep": [
            "./reports/table1_preview.tex",
            OUTPUT_DIR / "table1_summary_stats.tex",
            OUTPUT_DIR / "table1_comparison.tex",
        ],
        "clean": True,
    }
    yield {
        "name": "table3_preview",
        "doc": "Viewable PDF of the Table 3 replication",
        "actions": [
            "latexmk -pdf -halt-on-error -cd ./reports/table3_preview.tex",
            "latexmk -pdf -halt-on-error -c -cd ./reports/table3_preview.tex",
        ],
        "targets": ["./reports/table3_preview.pdf"],
        "file_dep": [
            "./reports/table3_preview.tex",
            OUTPUT_DIR / "table3_replication.tex",
        ],
        "clean": True,
    }
    yield {
        "name": "table4_preview",
        "doc": "Viewable PDF of the Table 4 replication",
        "actions": [
            "latexmk -pdf -halt-on-error -cd ./reports/table4_preview.tex",
            "latexmk -pdf -halt-on-error -c -cd ./reports/table4_preview.tex",
        ],
        "targets": ["./reports/table4_preview.pdf"],
        "file_dep": [
            "./reports/table4_preview.tex",
            OUTPUT_DIR / "table4_replication.tex",
        ],
        "clean": True,
    }
    yield {
        "name": "figure3_preview",
        "doc": "Viewable PDF of Figure 3 with definition and notes",
        "actions": [
            "latexmk -pdf -halt-on-error -cd ./reports/figure3_preview.tex",
            "latexmk -pdf -halt-on-error -c -cd ./reports/figure3_preview.tex",
        ],
        "targets": ["./reports/figure3_preview.pdf"],
        "file_dep": [
            "./reports/figure3_preview.tex",
            OUTPUT_DIR / "figure3_fraction_countries_rzone.jpg",
        ],
        "clean": True,
    }
    yield {
        "name": "table1_updated_preview",
        "doc": "Viewable PDF of updated Table 1",
        "actions": [
            "latexmk -pdf -halt-on-error -cd ./reports/table1_updated_preview.tex",
            "latexmk -pdf -halt-on-error -c -cd ./reports/table1_updated_preview.tex",
        ],
        "targets": ["./reports/table1_updated_preview.pdf"],
        "file_dep": [
            "./reports/table1_updated_preview.tex",
            OUTPUT_DIR / "table1_updated.tex",
        ],
        "clean": True,
    }
    yield {
        "name": "table3_updated_preview",
        "doc": "Viewable PDF of updated Table 3",
        "actions": [
            "latexmk -pdf -halt-on-error -cd ./reports/table3_updated_preview.tex",
            "latexmk -pdf -halt-on-error -c -cd ./reports/table3_updated_preview.tex",
        ],
        "targets": ["./reports/table3_updated_preview.pdf"],
        "file_dep": [
            "./reports/table3_updated_preview.tex",
            OUTPUT_DIR / "table3_updated.tex",
        ],
        "clean": True,
    }
    yield {
        "name": "table4_updated_preview",
        "doc": "Viewable PDF of updated Table 4",
        "actions": [
            "latexmk -pdf -halt-on-error -cd ./reports/table4_updated_preview.tex",
            "latexmk -pdf -halt-on-error -c -cd ./reports/table4_updated_preview.tex",
        ],
        "targets": ["./reports/table4_updated_preview.pdf"],
        "file_dep": [
            "./reports/table4_updated_preview.tex",
            OUTPUT_DIR / "table4_updated.tex",
        ],
        "clean": True,
    }
    yield {
        "name": "figure3_updated_preview",
        "doc": "Viewable PDF of updated Figure 3",
        "actions": [
            "latexmk -pdf -halt-on-error -cd ./reports/figure3_updated_preview.tex",
            "latexmk -pdf -halt-on-error -c -cd ./reports/figure3_updated_preview.tex",
        ],
        "targets": ["./reports/figure3_updated_preview.pdf"],
        "file_dep": [
            "./reports/figure3_updated_preview.tex",
            OUTPUT_DIR / "figure3_fraction_countries_rzone_updated.jpg",
        ],
        "clean": True,
    }
    yield {
        "name": "data_overview",
        "doc": "Typeset original data-overview table and figure with captions",
        "actions": [
            "latexmk -pdf -halt-on-error -cd ./reports/data_overview.tex",
            "latexmk -pdf -halt-on-error -c -cd ./reports/data_overview.tex",
        ],
        "targets": ["./reports/data_overview.pdf"],
        "file_dep": [
            "./reports/data_overview.tex",
            OUTPUT_DIR / "data_overview_summary.tex",
            OUTPUT_DIR / "data_overview_figure.jpg",
        ],
        "clean": True,
    }
    yield {
        "name": "final_report",
        "doc": "Single narrative PDF with every final historical and updated exhibit",
        "actions": [
            "latexmk -pdf -halt-on-error -cd ./reports/final_report.tex",
            "latexmk -pdf -halt-on-error -c -cd ./reports/final_report.tex",
        ],
        "targets": ["./reports/final_report.pdf"],
        "file_dep": [
            "./reports/final_report.tex",
            OUTPUT_DIR / "data_overview_summary.tex",
            OUTPUT_DIR / "data_overview_figure.jpg",
            OUTPUT_DIR / "table1_summary_stats.tex",
            OUTPUT_DIR / "table1_comparison.tex",
            OUTPUT_DIR / "table3_replication.tex",
            OUTPUT_DIR / "table4_replication.tex",
            OUTPUT_DIR / "figure1_event_history.png",
            OUTPUT_DIR / "historical_business_rzone_timeline.png",
            OUTPUT_DIR / "historical_household_rzone_timeline.png",
            OUTPUT_DIR / "figure3_fraction_countries_rzone.jpg",
            OUTPUT_DIR / "table1_updated.tex",
            OUTPUT_DIR / "table3_updated.tex",
            OUTPUT_DIR / "table4_updated.tex",
            OUTPUT_DIR / "figure1_event_history_updated.png",
            OUTPUT_DIR / "updated_business_rzone_timeline.png",
            OUTPUT_DIR / "updated_household_rzone_timeline.png",
            OUTPUT_DIR / "figure3_fraction_countries_rzone_updated.jpg",
            OUTPUT_DIR / "report_validation_summary.tex",
            OUTPUT_DIR / "report_fragility_summary.tex",
            OUTPUT_DIR / "report_dynamic_summary.tex",
            OUTPUT_DIR / "report_missed_crises.tex",
            OUTPUT_DIR / "report_tracker_summary.tex",
            OUTPUT_DIR / "report_usa_case_study.tex",
        ],
        "clean": True,
    }


notebook_tasks = {
    "01_predictable_financial_crises_project_tour.ipynb.py": {
        "path": "./src/01_predictable_financial_crises_project_tour.ipynb.py",
        "file_dep": [
            "./src/paper_benchmarks.py",
            "./src/build_analysis_panel.py",
            "./src/table1_summary_stats.py",
            "./src/updated_crisis_chronology.py",
            DATA_DIR / "rzone_analysis_panel.parquet",
            DATA_DIR / "imf_gdd.parquet",
            OUTPUT_DIR / "table1_stats.csv",
            OUTPUT_DIR / "rzone_validation.csv",
            OUTPUT_DIR / "post_2016_rzone_tracker.csv",
            OUTPUT_DIR / "data_overview_summary.csv",
            OUTPUT_DIR / "data_overview_figure.png",
            OUTPUT_DIR / "table3_crisis_probabilities.csv",
            OUTPUT_DIR / "table4_baseline_coefficients.csv",
            OUTPUT_DIR / "table4_baseline_models.csv",
            OUTPUT_DIR / "figure1_event_history.png",
            OUTPUT_DIR / "updated_data_coverage.csv",
            OUTPUT_DIR / "table4_updated_models.csv",
            OUTPUT_DIR / "figure3_fraction_countries_rzone_updated.png",
            OUTPUT_DIR / "dynamic_regression_models.csv",
            OUTPUT_DIR / "missed_crisis_fragility.csv",
            OUTPUT_DIR / "replication_validation.csv",
        ],
        "targets": [],
    },
}


# fmt: off
def task_run_notebooks():
    """Preps the notebooks for presentation format.
    Execute notebooks if the script version of it has been changed.
    """
    for notebook in notebook_tasks:
        pyfile_path = Path(notebook_tasks[notebook]["path"])
        notebook_path = pyfile_path.with_suffix("")  # strips .py, leaves .ipynb
        notebook_name = notebook_path.stem  # e.g. "01_data_tour"
        yield {
            "name": notebook,
            "actions": [
                """python -c "import sys; from datetime import datetime; print(f'Start """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
                f"jupytext --to notebook --output {notebook_path} {pyfile_path}",
                jupyter_execute_notebook(notebook_path),
                jupyter_to_html(notebook_path),
                mv(notebook_path, OUTPUT_DIR),
                """python -c "import sys; from datetime import datetime; print(f'End """ + notebook + """: {datetime.now()}', file=sys.stderr)" """,
            ],
            "file_dep": [
                pyfile_path,
                *notebook_tasks[notebook]["file_dep"],
            ],
            "targets": [
                OUTPUT_DIR / f"{notebook_name}.html",
                *notebook_tasks[notebook]["targets"],
            ],
            "clean": True,
        }
# fmt: on


def task_run_pytest():
    """Run pytest and save results to OUTPUT_DIR"""
    src_py_files = list(Path("./src").glob("*.py"))
    test_output = OUTPUT_DIR / "pytest_results.xml"

    def run_pytest():
        result = subprocess.run(
            ["pytest", f"--junitxml={test_output}"],
            check=False,
        )
        if result.returncode != 0:
            # Remove the XML so doit won't consider the target up-to-date
            Path(test_output).unlink(missing_ok=True)
            raise RuntimeError(f"pytest failed with exit code {result.returncode}")

    return {
        "actions": [run_pytest],
        "targets": [test_output],
        "file_dep": src_py_files,
        "clean": True,
        "verbosity": 2,
    }
