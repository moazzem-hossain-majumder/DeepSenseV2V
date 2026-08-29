import json
from pathlib import Path

ROOT = Path(r"d:\424_project-main\notebooks")
ROOT.mkdir(parents=True, exist_ok=True)

cells_by_file = {
    "01_eda_beam_reconstruction.ipynb": [
        ("markdown", "# Phase 1 — Beam reconstruction and feasibility\n\nImports `src.beam_reconstruction`. Also see `eda_annotated.ipynb` for the full annotated dataset EDA."),
        ("code", "from src.beam_reconstruction import verify_reconstruction_and_feasibility\nverify_reconstruction_and_feasibility(output_dir='results/eda')"),
    ],
    "02_gps_features.ipynb": [
        ("markdown", "# Phase 3 — GPS branch\n\nLocal ENU, distance/bearing, velocity proxies, HDOP flag. Scaler fit on train only."),
        ("code", "from src.gps_features import GPS_FEATURE_NAMES, extract_sequence_gps_features\nprint(GPS_FEATURE_NAMES)"),
    ],
    "03_rgb_features.ipynb": [
        ("markdown", "# Phase 3 — RGB branch\n\nResize, no horizontal flip (geometry), RAM cache in `src.dataset`."),
        ("code", "from src.rgb_features import get_rgb_transforms\nprint(get_rgb_transforms((96, 96), is_training=True))"),
    ],
    "04_baseline_models.ipynb": [
        ("markdown", "# Phase 4 — B0–B4\n\nB1/B3 Core; B0 geometric, B2 RGB-only, B4 deeper Transformer Stretch."),
        ("code", "from src.baseline_models import B0_Geometric, B1_GPSOnly, B2_RGBOnly\nfrom src.fusion_models import B3_Fusion, B4_MultimodalTransformer\nprint('B0-B4 imported')"),
    ],
    "05_profile_models.ipynb": [
        ("markdown", "# Phase 5 — P1/P2/P3\n\nP3 multi-task + smoothness penalty (Phase 5 item 4a)."),
        ("code", "from src.profile_models import P1_ClassificationOnly, P2_ProfileOnly, P3_MultiTaskProfile, MultiTaskLoss\nprint('P1-P3 imported')"),
    ],
    "06_candidate_sets.ipynb": [
        ("markdown", "# Phase 6 — Top-k and static CRC"),
        ("code", "from src.candidate_sets import build_topk_candidate_set, StaticConformalRiskControl, ExactLabelCRC\nprint('candidate sets OK')"),
    ],
    "07_online_control.ipynb": [
        ("markdown", "# Phase 7 — ACI (integral) and Conformal PID"),
        ("code", "from src.online_controller import run_online_aci_controller, run_online_pid_controller, select_best_eta_on_val\nprint('ACI + PID OK')"),
    ],
    "08_evaluation.ipynb": [
        ("markdown", "# Phase 8 — Metrics, block bootstrap, telecom tables\n\nSee `results/ablation_table.csv`, `telecom_utility_table.csv`, `significance_tests.csv`."),
        ("code", "import pandas as pd\nfrom pathlib import Path\np = Path('results/ablation_table.csv')\nprint(pd.read_csv(p) if p.exists() else 'run python -m src.complete_blueprint_steps')"),
    ],
}

def nb(cells):
    out = []
    for i, (kind, src) in enumerate(cells):
        if kind == "markdown":
            out.append({"cell_type": "markdown", "metadata": {}, "source": [src]})
        else:
            out.append({"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": [src]})
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}},
        "cells": out,
    }

for name, cells in cells_by_file.items():
    (ROOT / name).write_text(json.dumps(nb(cells), indent=1), encoding="utf-8")
    print("wrote", name)
