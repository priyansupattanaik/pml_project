from __future__ import annotations

import contextlib
import io
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nbformat
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = BASE_DIR / "finalcodemlproject.ipynb"
DATA_PATH = BASE_DIR / "Imperfect_HR_Attrition (1) (1).csv"


def _has_xgboost() -> bool:
    try:
        import xgboost  # noqa: F401

        return True
    except Exception:
        return False


def _display_stub(*args: Any, **kwargs: Any) -> None:
    return None


def _prepare_source(source: str, xgboost_available: bool) -> str | None:
    source = source.replace(
        'pd.read_csv("D:/ML Project/archive/Imperfect_HR_Attrition (1).csv")',
        f'pd.read_csv(r"{DATA_PATH}")',
    )

    if "os.walk('/kaggle/input')" in source:
        return None

    if not xgboost_available:
        source = re.sub(r"^\s*import\s+xgboost\s+as\s+xgb\s*$", "", source, flags=re.MULTILINE)
        if source.lstrip().startswith("#XGBOOST"):
            return None
        if "# XGBoost" in source:
            source = source.split("# XGBoost", 1)[0]

    if source.lstrip().startswith("#HEATMAP"):
        return None
    if source.lstrip().startswith("# DECISION TREE PLOT"):
        return None
    if source.lstrip().startswith("#RANDOM FOREST FEATURE IMPORTANCE"):
        return None

    return source


def load_notebook_namespace() -> dict[str, Any]:
    """Execute the project notebook and return its trained variables.

    The Streamlit app deliberately reuses the existing notebook code. This bridge
    only fixes the local CSV path, suppresses notebook display output, and skips
    optional XGBoost cells when the package is not installed.
    """

    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)
    xgboost_available = _has_xgboost()
    namespace: dict[str, Any] = {
        "__name__": "__notebook_bridge__",
        "display": _display_stub,
    }

    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        for cell in notebook.cells:
            if cell.get("cell_type") != "code":
                continue

            source = _prepare_source(cell.get("source", ""), xgboost_available)
            if not source or not source.strip():
                continue

            exec(compile(source, str(NOTEBOOK_PATH), "exec"), namespace)
            plt.close("all")

    namespace["xgboost_available"] = xgboost_available
    namespace["raw_df"] = pd.read_csv(DATA_PATH)
    return namespace
