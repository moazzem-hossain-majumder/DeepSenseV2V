"""Resolve DeepSense raw files across Scenarios 36, 37, 38, and 39."""

import os

_DEFAULT_HINTS = [
    os.environ.get("DEEPSENSE_DATA_ROOT", ""),
    r"D:\DeepSense_V2V\data",
    r"D:\DeepSense_V2V",
]

SCENARIOS = (36, 37, 38, 39)


def _has_scenarios(root):
    if not root or not os.path.isdir(root):
        return False
    # Check if any scenario csv exists directly or in scenario folder
    for s in SCENARIOS:
        csv_a = os.path.join(root, f"scenario{s}.csv")
        csv_b = os.path.join(root, f"scenario{s}", f"scenario{s}.csv")
        if os.path.isfile(csv_a) or os.path.isfile(csv_b):
            return True
    return False


def resolve_raw_data_root(data_root="."):
    """
    Return a directory that contains scenario CSVs across Scenarios 36-39.
    """
    candidates = [os.path.abspath(data_root), os.path.abspath(os.path.join(data_root, "data"))]
    candidates.extend([os.path.abspath(h) for h in _DEFAULT_HINTS if h])
    seen = set()
    for c in candidates:
        if c in seen:
            continue
        seen.add(c)
        if _has_scenarios(c):
            return c
        nested = os.path.join(c, "data")
        if nested not in seen and _has_scenarios(nested):
            return nested
    raise FileNotFoundError(
        "Could not find DeepSense multi-scenario raw data (Scenarios 36-39). "
        "Pass --data_root pointing at the folder that contains those files, "
        "or set DEEPSENSE_DATA_ROOT. Looked in: " + ", ".join(candidates)
    )


def scenario_csv_path(scenario, raw_root):
    s = str(scenario).replace("scenario", "")
    p = os.path.join(raw_root, f"scenario{s}.csv")
    if os.path.isfile(p):
        return p
    return os.path.join(raw_root, f"scenario{s}", f"scenario{s}.csv")


def scenario_pkl_path(scenario, raw_root):
    s = str(scenario).replace("scenario", "")
    p = os.path.join(raw_root, f"scenario{s}.p")
    if os.path.isfile(p):
        return p
    return os.path.join(raw_root, f"scenario{s}", f"scenario{s}.p")


# Aliases for scenario 36 (one of the four scenarios)
def scenario36_csv_path(raw_root):
    return scenario_csv_path(36, raw_root)


def scenario36_pkl_path(raw_root):
    return scenario_pkl_path(36, raw_root)
