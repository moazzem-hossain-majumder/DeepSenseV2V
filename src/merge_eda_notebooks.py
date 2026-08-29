"""Merge duplicate EDA notebooks into a single project notebook."""
import json
from pathlib import Path

PROJECT = Path(r"D:\424_project-main")
DEEPSENSE = Path(r"D:\DeepSense_V2V")

dataset_nb = json.loads((DEEPSENSE / "eda_annotated.ipynb").read_text(encoding="utf-8"))
pred_nb = json.loads((PROJECT / "eda.ipynb").read_text(encoding="utf-8"))

intro = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "# DeepSense 6G V2V Beam Tracking — Merged EDA\n",
        "\n",
        "This notebook is the union of:\n",
        "1. **Dataset / feasibility EDA** (Scenario 36 CSV, GPS, RGB, 256-beam reconstruction, 1 dB / 3 dB near-optimal sets, weak-link analysis).\n",
        "2. **Model-annotated EDA** (Top-k, APL, CRC candidate sets after GPU training).\n",
        "\n",
        "Raw files are auto-detected at `D:\\\\DeepSense_V2V\\\\data` (or `DEEPSENSE_DATA_ROOT`).\n",
        "\n",
        "**GPU run order**\n",
        "1. `python train.py --config config.yaml --device cuda --model P3`\n",
        "2. `python -m src.generate_annotated_eda --output_dir results`\n",
        "3. Re-run the model-annotated section at the bottom of this notebook.\n",
    ],
}

path_cell = {
    "cell_type": "code",
    "execution_count": None,
    "metadata": {},
    "outputs": [],
    "source": [
        "import os\n",
        "from pathlib import Path\n",
        "\n",
        "CANDIDATES = [\n",
        "    os.environ.get('DEEPSENSE_DATA_ROOT', ''),\n",
        "    r'D:\\DeepSense_V2V\\data',\n",
        "    str(Path.cwd()),\n",
        "]\n",
        "DATA_ROOT = None\n",
        "for c in CANDIDATES:\n",
        "    if not c:\n",
        "        continue\n",
        "    p = Path(c)\n",
        "    if (p / 'scenario36.csv').exists() or (p / 'scenario36' / 'scenario36.csv').exists():\n",
        "        DATA_ROOT = p\n",
        "        break\n",
        "print('DATA_ROOT =', DATA_ROOT)\n",
        "assert DATA_ROOT is not None, 'Set DEEPSENSE_DATA_ROOT or place scenario36.csv on disk'\n",
        "CSV_PATH = DATA_ROOT / 'scenario36.csv' if (DATA_ROOT / 'scenario36.csv').exists() else DATA_ROOT / 'scenario36' / 'scenario36.csv'\n",
        "print('CSV_PATH =', CSV_PATH)\n",
    ],
}

# Drop empty markdown stubs at the start of the dataset notebook
dataset_cells = [c for c in dataset_nb["cells"] if not (
    c.get("cell_type") == "markdown" and (not "".join(c.get("source", [])).strip())
)]

# Rewrite first load cell if it uses a hardcoded CSV path
for cell in dataset_cells:
    if cell.get("cell_type") != "code":
        continue
    src = "".join(cell.get("source", []))
    if "pd.read_csv" in src and "scenario36.csv" in src and "DATA_ROOT" not in src:
        cell["source"] = [
            "import pandas as pd\n",
            "df = pd.read_csv(CSV_PATH)\n",
            "print('Shape (rows, columns):', df.shape)\n",
            "print('\\nColumn names:')\n",
            "print(list(df.columns))\n",
            "print('\\nData types:')\n",
            "print(df.dtypes.to_string())\n",
            "print('\\nFirst row, fully expanded:')\n",
            "pd.set_option('display.max_colwidth', None)\n",
            "pd.set_option('display.max_columns', None)\n",
            "print(df.iloc[0].to_string())\n",
        ]
        cell["outputs"] = []
        cell["execution_count"] = None
        break

pred_cells = pred_nb["cells"]
# Skip the old title cell; keep dictionary + analysis
if pred_cells and pred_cells[0].get("cell_type") == "markdown":
    first = "".join(pred_cells[0].get("source", []))
    if "Merged EDA" in first:
        pred_cells = pred_cells[1:]

divider = {
    "cell_type": "markdown",
    "metadata": {},
    "source": [
        "---\n",
        "\n",
        "# Part 2 — Model-annotated EDA (after GPU training)\n",
        "\n",
        "Requires `data/processed/annotated_scenario36_with_predictions.csv` from `python -m src.generate_annotated_eda`.\n",
    ],
}

merged = {
    "nbformat": 4,
    "nbformat_minor": 5,
    "metadata": dataset_nb.get("metadata") or pred_nb.get("metadata"),
    "cells": [intro, path_cell] + dataset_cells + [divider] + pred_cells,
}

out = PROJECT / "eda.ipynb"
out.write_text(json.dumps(merged, indent=1), encoding="utf-8")
print(f"Wrote {out} with {len(merged['cells'])} cells")
