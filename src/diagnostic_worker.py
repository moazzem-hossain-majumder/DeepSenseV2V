import argparse
import json
import shutil
import time
from datetime import datetime,timezone
from src.diagnostic_plan import ROOT,OUT,PREVIOUS,PLAN
from src.publication_data import digest
from src.acquire_scenarios import save_json

def run(sid):
    plan=next(p for p in PLAN if p['id']==sid);folder=ROOT/plan['directory'];folder.mkdir(parents=True,exist_ok=True)
    def progress(**v):save_json(folder/'status.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),**v))
    progress(status='preparing',model='analysis');started=time.monotonic()
    try:
        if (ROOT/'results/publication_v1/stage35/test_lock.json').exists():raise ValueError('Test exposed; diagnostic cycle refused')
        for manifest in (ROOT/'results/publication_v1/stage8/code_manifest.json',PREVIOUS/'stage44/source_manifest.json'):
            if any(digest(ROOT/p)!=h for p,h in json.loads(manifest.read_text()).items()):raise ValueError('Frozen source changed')
        if sid==62:
            target=folder/'source';target.mkdir(exist_ok=True);hashes={}
            for p in ROOT.glob('src/diagnostic_*.py'):
                hashes[p.relative_to(ROOT).as_posix()]=digest(p);shutil.copy2(p,target/p.name)
            save_json(folder/'source_manifest.json',hashes);save_json(folder/'plan.json',PLAN)
        else:
            if any(digest(ROOT/p)!=h for p,h in json.loads((OUT/'stage62/source_manifest.json').read_text()).items()):raise ValueError('Diagnostic source changed')
        if plan['kind']=='intervention':
            from src.diagnostic_intervention import run as evaluate
            evaluate(plan,folder,progress)
        else:
            from src.diagnostic_analysis import alignment,failures,report
            {'alignment':alignment,'failures':failures,'report':report}[plan['kind']](folder,progress)
        save_json(folder/'analysis_history.json',[dict(epoch=1,seconds=time.monotonic()-started)])
        save_json(folder/'analysis_finished.json',dict(reason='diagnostic_complete'))
        progress(status='complete',model='analysis')
    except Exception as exc:progress(status='failed',error=f'{type(exc).__name__}: {exc}');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',type=int,required=True);run(p.parse_args().stage)
