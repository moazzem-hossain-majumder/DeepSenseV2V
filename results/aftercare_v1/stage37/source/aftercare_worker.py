import argparse
from datetime import datetime,timezone
import json
import shutil
import time
from src.aftercare_plan import ROOT,OUT,PARENT,plan_for
from src.publication_data import digest
from src.acquire_scenarios import save_json


def run(stage):
    route=json.loads((OUT/'route.json').read_text())
    plan=next(p for p in plan_for(route['branch']) if p['id']==stage)
    folder=ROOT/plan['directory'];folder.mkdir(parents=True,exist_ok=True)
    def progress(**values): save_json(folder/'status.json',dict(updated_utc=datetime.now(timezone.utc).isoformat(),**values))
    progress(status='preparing',model=plan['models'][0]);start=time.monotonic()
    try:
        parent_hashes=json.loads((PARENT/'stage8/code_manifest.json').read_text())
        if any(digest(ROOT/rel)!=sha for rel,sha in parent_hashes.items()): raise ValueError('Frozen parent code changed')
        if plan['kind']=='route':
            snapshot=folder/'source';snapshot.mkdir(exist_ok=True)
            hashes={}
            for path in sorted(ROOT.glob('src/aftercare_*.py')):
                hashes[path.relative_to(ROOT).as_posix()]=digest(path);shutil.copy2(path,snapshot/path.name)
            save_json(folder/'source_manifest.json',hashes)
            save_json(folder/'branch_plan.json',dict(route=route,stages=plan_for(route['branch']),
                parent_queue_sha256=digest(PARENT/'queue.json'),
                parent_decision_sha256=digest(PARENT/'stage33/decision.json'),
                parent_source=parent_hashes))
        else:
            declared=json.loads((OUT/'stage37/source_manifest.json').read_text())
            if any(digest(ROOT/rel)!=sha for rel,sha in declared.items()): raise ValueError('Follow-up source changed after declaration')
            if plan['kind']=='train':
                from src.aftercare_geometry import run as train
                train(plan,progress)
                if json.loads((folder/'status.json').read_text())['status']=='paused_by_user': return
            else:
                from src.aftercare_analysis import image_eda,diagnostics,reproduce,test_audit,draft,review_report,improvement_report
                functions=dict(images=image_eda,diagnostics=diagnostics,reproduce=reproduce,test_audit=test_audit,
                    draft=draft,review_report=review_report,improvement_report=improvement_report)
                functions[plan['kind']](folder,progress)
        if plan['kind']!='train':
            save_json(folder/'analysis_history.json',[dict(epoch=1,seconds=time.monotonic()-start)])
            save_json(folder/'analysis_finished.json',dict(reason='analysis_complete'))
        progress(status='complete',scope='Conditional follow-up stage, not an automatic publication-readiness claim.')
    except Exception as exc:
        progress(status='failed',error=f'{type(exc).__name__}: {exc}');raise


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',type=int,required=True)
    run(parser.parse_args().stage)
