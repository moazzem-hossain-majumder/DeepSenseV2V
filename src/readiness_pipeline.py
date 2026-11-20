"""Resume integrity/EDA, then run the gated GPU smoke matrix independently of chat."""
import json
from pathlib import Path
import subprocess
import sys

import psutil

from src.acquire_scenarios import ROOT, save_json

OUT=ROOT/'results'/'readiness'


def wait_existing_audit():
    for process in psutil.process_iter(['pid','cmdline']):
        args=process.info['cmdline'] or []
        if 'src.readiness_audit' not in args: continue
        try:
            if Path(process.cwd()).resolve()!=ROOT.resolve(): continue
            print(f'Waiting for existing audit PID {process.pid}; no duplicate scan',flush=True)
            while True:
                try:
                    process.wait(timeout=15)
                    break
                except psutil.TimeoutExpired: pass
        except (psutil.NoSuchProcess,psutil.AccessDenied): pass


def main():
    OUT.mkdir(parents=True,exist_ok=True)
    save_json(OUT/'pipeline_status.json',{'stage':'integrity_audit','status':'running'})
    wait_existing_audit()
    summary=OUT/'integrity_summary.json'
    if not summary.exists() or not json.loads(summary.read_text())['passed']:
        code=subprocess.call([sys.executable,'-u','-m','src.readiness_audit','--workers','8','--resume'],cwd=ROOT)
        if code:
            save_json(OUT/'pipeline_status.json',{'stage':'integrity_audit','status':'needs_attention','exit_code':code})
            return 2
    from src.readiness_report import main as report
    report()
    save_json(OUT/'pipeline_status.json',{'stage':'training_smoke','status':'running'})
    code=subprocess.call([sys.executable,'-u','-m','src.readiness_smoke'],cwd=ROOT)
    report()
    save_json(OUT/'pipeline_status.json',{'stage':'training_smoke','status':'complete' if code==0 else 'needs_attention','exit_code':code})
    return 0 if code==0 else 2


if __name__=='__main__': raise SystemExit(main())
