"""Execute one predeclared publication-development stage, with truthful terminal status."""
import argparse
from datetime import datetime,timezone
import json
import platform
import shutil
import subprocess
import sys
import time
import torch
from src.publication_plan import ROOT,OUT,PLAN,COMMON
from src.publication_data import digest
from src.acquire_scenarios import save_json


def source_files():
    return sorted({*ROOT.glob('src/publication_*.py'),ROOT/'src/models.py',ROOT/'src/gps_features.py',
        ROOT/'src/multiscenario_windows.py',ROOT/'src/unattended_queue.py',ROOT/'src/supervise_work.py'})


def run(stage):
    plan=next(p for p in PLAN if p['id']==stage);folder=ROOT/plan['directory'];folder.mkdir(parents=True,exist_ok=True)
    def progress(**values): save_json(folder/'status.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),**values))
    progress(status='preparing',model=plan['models'][0])
    start=time.monotonic()
    try:
        if plan['kind']=='prepare':
            from src.publication_data import prepare
            snapshot=folder/'source';snapshot.mkdir(exist_ok=True)
            hashes={}
            for path in source_files():
                rel=path.relative_to(ROOT).as_posix();hashes[rel]=digest(path)
                shutil.copy2(path,snapshot/path.name)
            save_json(folder/'code_manifest.json',hashes)
            shutil.copy2(ROOT/'paper/BENCHMARK_PROTOCOL_LEDGER.md',folder/'PROTOCOL.md')
            save_json(folder/'plan.json',dict(stages=PLAN,common=COMMON))
            save_json(folder/'environment.json',dict(python=sys.version,platform=platform.platform(),torch=torch.__version__,
                cuda=torch.version.cuda,gpu=torch.cuda.get_device_name(0),
                pretrained_resnet18_sha256=digest(__import__('pathlib').Path(torch.hub.get_dir())/'checkpoints/resnet18-f37072fd.pth')))
            (folder/'pip_freeze.txt').write_text(subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True),encoding='utf-8')
            prepare(progress)
        else:
            recorded=json.loads((OUT/'stage8/code_manifest.json').read_text())
            for rel,sha in recorded.items():
                if digest(ROOT/rel)!=sha: raise ValueError('Declared experiment code changed; new plan version required')
            if plan['kind']=='train':
                from src.publication_training import train
                train(plan,progress)
                if json.loads((folder/'status.json').read_text()).get('status')=='paused_by_user': return
            else:
                from src.publication_evidence import audit,gate,heldout,report
                if plan['kind']=='audit': audit(folder,progress)
                elif plan['kind']=='gate': gate(folder,progress)
                elif plan['kind']=='calibrate': heldout(folder,progress,'calib')
                elif plan['kind']=='test': heldout(folder,progress,'test')
                elif plan['kind']=='report': report(folder,progress)
        if plan['kind']!='train':
            save_json(folder/'analysis_history.json',[dict(epoch=1,seconds=time.monotonic()-start)])
            save_json(folder/'analysis_finished.json',dict(reason='skipped_by_quality_gate' if (folder/'skipped.json').exists() else 'analysis_complete'))
        progress(status='complete',scope='Publication-development pipeline; scientific readiness is a separate conclusion.')
    except Exception as exc:
        progress(status='failed',error=f'{type(exc).__name__}: {exc}')
        raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=int,required=True)
    run(parser.parse_args().stage)
