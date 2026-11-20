import argparse
from datetime import datetime,timezone
import json
import shutil
import time
from src.conservative_plan import ROOT,OUT,PARENT,PLAN
from src.publication_data import digest
from src.acquire_scenarios import save_json


def run(sid):
    plan=next(p for p in PLAN if p['id']==sid);folder=ROOT/plan['directory'];folder.mkdir(parents=True,exist_ok=True)
    def progress(**values):save_json(folder/'status.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),**values))
    def training_progress(**values):
        if values.get('status')=='complete':values['status']='finalizing_selection'
        progress(**values)
    progress(status='preparing',model=plan['models'][0]);began=time.monotonic()
    try:
        if (PARENT/'stage35/test_lock.json').exists() or (PARENT/'stage35/test_report.json').exists():raise ValueError('Test already exposed; development queue refused')
        original=json.loads((PARENT/'stage8/code_manifest.json').read_text())
        if any(digest(ROOT/p)!=sha for p,sha in original.items()):raise ValueError('Frozen parent code changed')
        if sid==44:
            snapshot=folder/'source';snapshot.mkdir(exist_ok=True);hashes={}
            for path in sorted(ROOT.glob('src/conservative_*.py')):
                hashes[path.relative_to(ROOT).as_posix()]=digest(path);shutil.copy2(path,snapshot/path.name)
            save_json(folder/'source_manifest.json',hashes);save_json(folder/'plan.json',PLAN)
            from src.conservative_geometry import prepare
            prepare(folder,progress)
        else:
            hashes=json.loads((OUT/'stage44/source_manifest.json').read_text())
            if any(digest(ROOT/p)!=sha for p,sha in hashes.items()):raise ValueError('Conservative source changed')
            if plan['kind']=='train':
                from src.conservative_training import run as train
                train(plan,training_progress)
                if json.loads((folder/'status.json').read_text())['status']=='paused_by_user':return
            else:
                from src.conservative_analysis import diagnostics,compare,report
                {'diagnostics':diagnostics,'compare':compare,'report':report}[plan['kind']](folder,progress)
        if plan['kind']!='train':
            save_json(folder/'analysis_history.json',[dict(epoch=1,seconds=time.monotonic()-began)])
            save_json(folder/'analysis_finished.json',dict(reason='analysis_complete'))
        progress(status='complete',scope='Development cycle only; calibration/test reserved.')
    except Exception as exc:
        progress(status='failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=int,required=True)
    run(parser.parse_args().stage)
