import argparse
import json
import shutil
import time
from datetime import datetime,timezone
from src.rerank_plan import ROOT,OUT,PLAN
from src.publication_data import digest
from src.acquire_scenarios import save_json

def run(sid):
    plan=next(p.copy() for p in PLAN if p['id']==sid);folder=ROOT/plan['directory'];folder.mkdir(parents=True,exist_ok=True)
    def progress(**v):save_json(folder/'status.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),**v))
    def training_progress(**v):
        if v.get('status')=='complete':v['status']='finalizing'
        progress(**v)
    def skip(reason):
        save_json(folder/'skipped.json',dict(reason=reason));progress(status='complete',model=plan['models'][0],scope='Skipped: '+reason)
    progress(status='preparing',model=plan['models'][0]);started=time.monotonic()
    try:
        if (ROOT/'results/publication_v1/stage35/test_lock.json').exists():raise ValueError('Test already exposed')
        for path in ('results/publication_v1/stage8/code_manifest.json','results/conservative_v1/stage44/source_manifest.json','results/camera_v1/stage81/source_manifest.json'):
            if any(digest(ROOT/p)!=h for p,h in json.loads((ROOT/path).read_text()).items()):raise ValueError('Frozen source changed')
        if sid==96:
            target=folder/'source';target.mkdir(exist_ok=True);hashes={}
            for p in ROOT.glob('src/rerank_*.py'):
                hashes[p.relative_to(ROOT).as_posix()]=digest(p);shutil.copy2(p,target/p.name)
            save_json(folder/'source_manifest.json',hashes);save_json(folder/'plan.json',PLAN)
        elif any(digest(ROOT/p)!=h for p,h in json.loads((OUT/'stage96/source_manifest.json').read_text()).items()):raise ValueError('Reranking source changed')
        if plan['kind'] in ('pilot','expand'):
            if plan['kind']=='pilot':
                if plan['k'] not in json.loads((OUT/'stage96/coverage.json').read_text())['eligible_k']:
                    skip('Insufficient training shortlist headroom');return
            else:
                chosen=json.loads((OUT/'stage101/selection.json').read_text())['chosen']
                if not chosen:skip('No qualifying visual pilot');return
                plan['k']=chosen['k']
            from src.rerank_training import run as train
            train(plan,training_progress)
            if json.loads((folder/'status.json').read_text())['status']=='paused_by_user':return
        else:
            from src.rerank_evidence import coverage,gate,report
            {'coverage':coverage,'gate':gate,'report':report}[plan['kind']](folder,progress)
            save_json(folder/'analysis_history.json',[dict(epoch=1,seconds=time.monotonic()-started)])
            save_json(folder/'analysis_finished.json',dict(reason='complete'))
        progress(status='complete',model=plan['models'][0])
    except Exception as exc:progress(status='failed',error=f'{type(exc).__name__}: {exc}');raise

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',type=int,required=True);run(p.parse_args().stage)
