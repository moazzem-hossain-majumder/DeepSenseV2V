import argparse
import json
import shutil
import time
from datetime import datetime, timezone
from src.acquire_scenarios import save_json
from src.publication_data import digest
from src.swin_comparison_plan import OUT, PLAN, ROOT


def run(sid):
    plan = next(p for p in PLAN if p["id"] == sid)
    folder = ROOT / plan["directory"]
    folder.mkdir(parents=True, exist_ok=True)
    def progress(**fields):
        save_json(folder / "status.json", dict(updated_utc=datetime.now(timezone.utc).isoformat(), **fields))
    began = time.monotonic()
    try:
        progress(status="preparing", model=plan["models"][0])
        freeze = OUT / "stage268/source_manifest.json"
        if not freeze.exists():
            if sid != 268:
                raise ValueError("Missing frozen protocol")
            paths = list(ROOT.glob("src/swin_comparison_*.py")) + [ROOT / "paper/SWIN_COMPARISON_PROTOCOL.md"]
            target = folder / "source"
            target.mkdir(exist_ok=True)
            for path in paths:
                shutil.copy2(path, target / path.name)
            save_json(freeze, {p.relative_to(ROOT).as_posix(): digest(p) for p in paths})
            save_json(folder / "plan.json", PLAN)
        for path, checksum in json.loads(freeze.read_text()).items():
            if digest(ROOT / path) != checksum:
                raise ValueError(f"Frozen file changed: {path}")
        if sid == 268:
            from src.swin_comparison_data import prepare
            prepare(progress)
        elif plan["kind"] == "train":
            from src.swin_comparison_training import train
            train(plan, progress)
            return
        else:
            from src.swin_comparison_evidence import report
            report(folder)
        save_json(folder / "analysis_history.json", [dict(epoch=1, seconds=time.monotonic()-began)])
        save_json(folder / "analysis_finished.json", dict(reason="complete"))
        progress(status="complete", model="analysis")
    except Exception as exc:
        progress(status="failed", error=str(exc))
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", type=int, required=True)
    run(parser.parse_args().stage)
