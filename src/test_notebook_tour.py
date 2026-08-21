"""Structural and execution checks for the reader-facing Jupyter data tour."""

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATH = ROOT / "src" / "01_predictable_financial_crises_project_tour.ipynb.py"
NOTEBOOK_PATH = ROOT / "_output" / "01_predictable_financial_crises_project_tour.ipynb"
HTML_PATH = ROOT / "_output" / "01_predictable_financial_crises_project_tour.html"


def test_notebook_source_has_hw_guide_structure_and_analysis_tour():
    # The tour source must keep every HW-guide section heading, since the
    # rubric grades the notebook as a structured walk through the analysis.
    source = SOURCE_PATH.read_text(encoding="utf-8")
    required_sections = [
        "## Summary",
        "## Learning Outcomes",
        "## Analysis Roadmap",
        "## Step 1. Load the Cleaned Analysis Panel",
        "## Step 2. Inspect Coverage and Source Splicing",
        "## Step 3. Understand the Historical Forecasting Sample",
        "## Step 4. Reconstruct the Core Variables",
        "## Step 5. Reproduce the Historical Descriptive Analysis",
        "## Step 6. Read the Fixed-Effects Regression Output",
        "## Step 7. Tour the Post-Publication Update",
        "## Step 8. See What the Extensions Add",
        "## Step 9. Check Replication Tolerances",
        "## Where to Go Next",
    ]
    for section in required_sections:
        assert section in source


@pytest.mark.skipif(
    not NOTEBOOK_PATH.exists() or not HTML_PATH.exists(),
    reason="notebook not rendered -- run its doit task",
)
def test_rendered_notebook_is_executed_without_cell_errors():
    # The shipped notebook must be fully executed with no error outputs and
    # report the stricter paper-scaled audit result, including known misses.
    notebook = json.loads(NOTEBOOK_PATH.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) >= 20
    assert all(cell.get("execution_count") is not None for cell in code_cells)
    errors = [
        output
        for cell in code_cells
        for output in cell.get("outputs", [])
        if output.get("output_type") == "error"
    ]
    assert errors == []
    html = HTML_PATH.read_text(encoding="utf-8")
    assert "Learning Outcomes" in html
    assert "Benchmarks outside tolerance: 41" in html
