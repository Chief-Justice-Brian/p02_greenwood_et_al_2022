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


notebook_tasks = {
    "01_table1_validation_tour.ipynb.py": {
        "path": "./src/01_table1_validation_tour.ipynb.py",
        "file_dep": [
            "./src/paper_benchmarks.py",
            "./src/build_analysis_panel.py",
            DATA_DIR / "rzone_analysis_panel.parquet",
            OUTPUT_DIR / "table1_stats.csv",
        ],
        "targets": [],
    },
}


# fmt: off
def task_run_notebooks():
    """Preps the notebooks for presentation format.
    Execute notebooks if the script version of it has been changed.
    """
    for notebook in notebook_tasks.keys():
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
