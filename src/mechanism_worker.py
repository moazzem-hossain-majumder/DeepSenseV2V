import argparse
import json
import shutil
import time
from datetime import datetime,timezone
from src.mechanism_plan import ROOT,OUT,PLAN
from src.publication_data import digest
from src.acquire_scenarios import save_json

def run(sid):
    plan=next(p for p in PLAN if p['id']==sid);folder=ROOT/plan['directory'];folder.mkdir(parents=True,exist_ok=True)
    def progress(**v):save_json(folder/'status.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),**v))
    progress(status='preparing',model='analysis');started=time.monotonic()
    try:
        if (ROOT/'results/publication_v1/stage35/test_lock.json').exists():raise ValueError('Test exposed')
        for path in ('results/publication_v1/stage8/code_manifest.json','results/conservative_v1/stage44/source_manifest.json','results/camera_v1/stage81/source_manifest.json','results/rerank_v1/stage96/source_manifest.json'):
            if any(digest(ROOT/p)!=h for p,h in json.loads((ROOT/path).read_text()).items()):raise ValueError('Frozen source changed')
        if sid==117:
            target=folder/'source';target.mkdir(exist_ok=True);hashes={}
            for p in ROOT.glob('src/mechanism_*.py'):
                hashes[p.relative_to(ROOT).as_posix()]=digest(p);shutil.copy2(p,target/p.name)
            save_json(folder/'source_manifest.json',hashes);save_json(folder/'plan.json',PLAN)
        elif any(digest(ROOT/p)!=h for p,h in json.loads((OUT/'stage117/source_manifest.json').read_text()).items()):raise ValueError('Mechanism source changed')
        if plan['kind']=='sample':
            from src.mechanism_sample import sample
            sample(folder,progress)
        elif plan['kind']=='memorize':
            from src.mechanism_memorize import run as memorize
            memorize(plan,folder,progress)
            if json.loads((folder/'status.json').read_text())['status']=='paused_by_user':return
        else:
            from src.mechanism_review import benchmark,report
            {'benchmark':benchmark,'report':report}[plan['kind']](folder,progress)
        save_json(folder/'analysis_history.json',[dict(epoch=1,seconds=time.monotonic()-started)])
        save_json(folder/'analysis_finished.json',dict(reason='audit_complete'))
        progress(status='complete',model='analysis')
    except Exception as exc:progress(status='failed',error=f'{type(exc).__name__}: {exc}');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',type=int,required=True);run(p.parse_args().stage)
