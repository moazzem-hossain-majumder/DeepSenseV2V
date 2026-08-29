"""Resolve DeepSense Scenario 36 raw files without copying the RGB archive."""

import os

_DEFAULT_HINTS = [
    os.environ.get("DEEPSENSE_DATA_ROOT", ""),
    r"D:\DeepSense_V2V\data",
    r"D:\DeepSense_V2V",
]


def _has_scenario36(root):
    if not root or not os.path.isdir(root):
        return False
    csv_a = os.path.join(root, "scenario36.csv")
    csv_b = os.path.join(root, "scenario36", "scenario36.csv")
    pkl_a = os.path.join(root, "scenario36.p")
    pkl_b = os.path.join(root, "scenario36", "scenario36.p")
    return (os.path.isfile(csv_a) or os.path.isfile(csv_b)) and (
        os.path.isfile(pkl_a) or os.path.isfile(pkl_b)
    )


def resolve_raw_data_root(data_root="."):
    """
    Return a directory that contains scenario36.csv / scenario36.p
    (or those files inside a scenario36/ subfolder).
    """
    candidates = [os.path.abspath(data_root), os.path.abspath(os.path.join(data_root, "data"))]
    candidates.extend([os.path.abspath(h) for h in _DEFAULT_HINTS if h])
    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if _has_scenario36(c):
            return c
        nested = os.path.join(c, "data")
        if nested not in seen and _has_scenario36(nested):
            return nested
    raise FileNotFoundError(
        "Could not find Scenario 36 (scenario36.csv + scenario36.p). "
        "Pass --data_root pointing at the folder that contains those files, "
        "or set DEEPSENSE_DATA_ROOT. Looked in: " + ", ".join(candidates)
    )


def scenario36_csv_path(raw_root):
    p = os.path.join(raw_root, "scenario36.csv")
    if os.path.isfile(p):
        return p
    return os.path.join(raw_root, "scenario36", "scenario36.csv")


def scenario36_pkl_path(raw_root):
    p = os.path.join(raw_root, "scenario36.p")
    if os.path.isfile(p):
        return p
    return os.path.join(raw_root, "scenario36", "scenario36.p")
