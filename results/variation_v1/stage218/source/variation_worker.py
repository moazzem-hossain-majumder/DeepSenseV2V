import argparse
import json
import shutil
import time
from datetime import datetime,timezone
from src.variation_plan import ROOT,OUT,PLAN
from src.publication_data import digest
from src.acquire_scenarios import save_json

def run(sid):
    plan=next(p for p in PLAN if p['id']==sid);folder=ROOT/plan['directory'];folder.mkdir(parents=True,exist_ok=True)
    def progress(**v):save_json(folder/'status.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),**v))
    def training_progress(**v):
        if v.get('status')=='complete':v['status']='finalizing'
        progress(**v)
    progress(status='preparing',model=plan['models'][0]);started=time.monotonic()
    try:
        if (ROOT/'results/publication_v1/stage35/test_lock.json').exists():raise ValueError('Test exposed')
        for path in ('results/publication_v1/stage8/code_manifest.json','results/conservative_v1/stage44/source_manifest.json','results/camera_v1/stage81/source_manifest.json','results/rerank_v1/stage96/source_manifest.json','results/factorial_v1/stage124/source_manifest.json','results/penalty_v1/stage134/source_manifest.json','results/replicate_v1/stage140/source_manifest.json','results/resolution_v1/stage151/source_manifest.json','results/rescheck_v1/stage155/source_manifest.json','results/transfer_v1/stage164/source_manifest.json','results/steady_v1/stage190/source_manifest.json','results/anchor_v1/stage211/source_manifest.json'):
            if any(digest(ROOT/p)!=h for p,h in json.loads((ROOT/path).read_text()).items()):raise ValueError('Frozen source changed')
        deadline=datetime.fromisoformat(json.loads((ROOT/'results/autonomous_17h_window.json').read_text())['ends_utc'])
        if datetime.now(timezone.utc)>=deadline and plan['kind']!='report':
            save_json(folder/'skipped.json',dict(reason='Window ended'));progress(status='complete',model='analysis');return
        if sid==218:
            target=folder/'source';target.mkdir(exist_ok=True);hashes={}
            for p in ROOT.glob('src/variation_*.py'):
                hashes[p.relative_to(ROOT).as_posix()]=digest(p);shutil.copy2(p,target/p.name)
            save_json(folder/'source_manifest.json',hashes);save_json(folder/'plan.json',PLAN)
        else:
            if any(digest(ROOT/p)!=h for p,h in json.loads((OUT/'stage218/source_manifest.json').read_text()).items()):raise ValueError('Resolution replication source changed')
            from src.variation_analysis import audit,report
            if plan['kind']=='audit':audit(plan,folder,progress)
            else:report(folder,progress)
        if plan['kind']!='train':
            save_json(folder/'analysis_history.json',[dict(epoch=1,seconds=time.monotonic()-started)])
            save_json(folder/'analysis_finished.json',dict(reason='complete'))
        progress(status='complete',model=plan['models'][0])
    except Exception as exc:progress(status='failed',error=f'{type(exc).__name__}: {exc}');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',type=int,required=True);run(p.parse_args().stage)
