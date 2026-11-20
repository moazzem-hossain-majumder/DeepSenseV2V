import argparse
import json
import shutil
import time
from datetime import datetime,timezone
from src.temporal_plan import ROOT,OUT,PLAN,WINDOW
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
        for path in ('results/publication_v1/stage8/code_manifest.json','results/conservative_v1/stage44/source_manifest.json','results/camera_v1/stage81/source_manifest.json','results/rerank_v1/stage96/source_manifest.json','results/factorial_v1/stage124/source_manifest.json','results/penalty_v1/stage134/source_manifest.json','results/replicate_v1/stage140/source_manifest.json','results/resolution_v1/stage151/source_manifest.json','results/rescheck_v1/stage155/source_manifest.json','results/transfer_v1/stage164/source_manifest.json','results/robust_v1/stage180/source_manifest.json','results/steady_v1/stage190/source_manifest.json','results/motion_v1/stage221/source_manifest.json'):
            if any(digest(ROOT/p)!=h for p,h in json.loads((ROOT/path).read_text()).items()):raise ValueError('Frozen source changed')
        if sid==228:
            target=folder/'source';target.mkdir(exist_ok=True);hashes={}
            for p in ROOT.glob('src/temporal_*.py'):
                hashes[p.relative_to(ROOT).as_posix()]=digest(p);shutil.copy2(p,target/p.name)
            save_json(folder/'source_manifest.json',hashes);save_json(folder/'plan.json',PLAN)
        else:
            if any(digest(ROOT/p)!=h for p,h in json.loads((OUT/'stage228/source_manifest.json').read_text()).items()):raise ValueError('Resolution replication source changed')
            if plan['kind']=='train':
                if datetime.now(timezone.utc)>=datetime.fromisoformat(json.loads(WINDOW.read_text())['ends_utc']) and not (folder/'started_training.json').exists():
                    save_json(folder/'skipped.json',dict(reason='17-hour authorization window expired'))
                    progress(status='complete',model='fusion_ce',scope='Skipped after deadline');return
                if not (folder/'started_training.json').exists():save_json(folder/'started_training.json',dict(utc=datetime.now(timezone.utc).isoformat()))
                from src.motion_training import run as train
                train(plan,training_progress)
                if json.loads((folder/'status.json').read_text())['status']=='paused_by_user':return
            else:
                from src.temporal_evidence import report
                report(folder,progress)
        if plan['kind']!='train':
            save_json(folder/'analysis_history.json',[dict(epoch=1,seconds=time.monotonic()-started)])
            save_json(folder/'analysis_finished.json',dict(reason='complete'))
        progress(status='complete',model=plan['models'][0])
    except Exception as exc:progress(status='failed',error=f'{type(exc).__name__}: {exc}');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',type=int,required=True);run(p.parse_args().stage)
