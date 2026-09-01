"""End-to-end GPU pipeline: feasibility check, models B1/B3/P1/P3, annotated EDA."""
import os
import sys
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(ROOT)

PYTHON = sys.executable


def run(cmd, check=True):
    print("\n>>>", " ".join(cmd), flush=True)
    env = os.environ.copy()
    env.setdefault("DEEPSENSE_DATA_ROOT", r"D:\DeepSense_V2V\data")
    env["PYTHONUNBUFFERED"] = "1"
    return subprocess.run(cmd, check=check, env=env)


def main():
    env = os.environ.copy()
    env.setdefault("DEEPSENSE_DATA_ROOT", r"D:\DeepSense_V2V\data")
    env["PYTHONUNBUFFERED"] = "1"

    print("=== Beam reconstruction and feasibility check ===", flush=True)
    run([PYTHON, "-m", "src.beam_reconstruction"])

    print("=== GPU dry-run (P3) ===", flush=True)
    r = subprocess.run(
        [PYTHON, "train.py", "--config", "config.yaml", "--device", "cuda", "--model", "P3", "--dry_run"],
        cwd=ROOT,
    )
    if r.returncode != 0:
        print("Dry-run failed; retrying at batch_size=32", flush=True)
        run([PYTHON, "train.py", "--config", "config.yaml", "--device", "cuda", "--model", "P3", "--batch_size", "32", "--dry_run"])

    core_models = ["P3", "B1", "B3", "P1"]
    for name in core_models:
        print(f"=== Train {name} on CUDA ===", flush=True)
        run([PYTHON, "train.py", "--config", "config.yaml", "--device", "cuda", "--model", name, "--output_dir", "results"])

    print("=== Annotated EDA inference ===", flush=True)
    run([PYTHON, "-m", "src.generate_annotated_eda", "--output_dir", "results"])
    print("Pipeline complete.", flush=True)


if __name__ == "__main__":
    main()
