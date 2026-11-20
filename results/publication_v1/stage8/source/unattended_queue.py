"""Durable finite queue. Only advance when worker exits successfully AND output passes gate."""
import json
import os
import subprocess
import sys
import time
from pathlib import Path
import psutil
from src.acquire_scenarios import ROOT,save_json

QUEUE=ROOT/'results/unattended_queue.json'


def completed(folder):
    path=folder/'status.json'
    return path.exists() and json.loads(path.read_text())['status']=='complete'


def run(plan=None,worker_module='src.unattended_stages',job='queue',queue_file=None):
    if plan is None:
        from src.unattended_stages import PLAN
    else: PLAN=plan
    state_path=queue_file or QUEUE
    if not completed(ROOT/'results/validation_diagnostics_v1'):
        raise ValueError('Stage 3 completion gate failed')
    state=json.loads(state_path.read_text()) if state_path.exists() else dict(status='running',stages={})
    def persist(): save_json(state_path,state)
    for plan in PLAN:
        state['stages'].setdefault(str(plan['id']),dict(title=plan['title'],status='queued',failures=0))
    for plan in PLAN:
        key=str(plan['id']);folder=ROOT/plan['directory'];folder.mkdir(parents=True,exist_ok=True)
        record=state['stages'].setdefault(key,dict(title=plan['title'],status='queued',failures=0))
        if completed(folder): record['status']='complete';persist();continue
        if record['failures']>=3:
            state['status']='needs_attention';persist();return 2
        if (ROOT/'pause.flag').exists():
            state['status']='paused_by_user';persist();return 0
        state.update(status='running',active_stage=plan['id'])
        record['status']='running';persist()
        save_json(ROOT/'results/active_training.json',{**plan,'job':job,'queue_file':str(state_path)})
        # Adopt a surviving worker after a queue-process interruption instead of duplicating it.
        process=None
        if record.get('pid'):
            try:
                candidate=psutil.Process(record['pid'])
                if candidate.create_time()==record.get('created') and worker_module in candidate.cmdline():
                    process=candidate
            except psutil.NoSuchProcess: pass
        with (folder/'worker.log').open('a',encoding='utf-8') as log:
            if process is None:
                worker=subprocess.Popen([sys.executable,'-u','-m',worker_module,'--stage',key],
                    cwd=ROOT,stdout=log,stderr=log,creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                process=psutil.Process(worker.pid)
                record.update(pid=worker.pid,created=process.create_time());persist()
            exit_code=None
            while process.is_running():
                try: exit_code=process.wait(timeout=15)
                except psutil.TimeoutExpired: continue
                break
        if completed(folder) and exit_code in (None,0):
            record['status']='complete';record.pop('pid',None);persist()
        else:
            status_path=folder/'status.json'
            output=json.loads(status_path.read_text()) if status_path.exists() else {}
            if output.get('status')=='paused_by_user':
                state['status']='paused_by_user';persist();return 0
            record.update(status='retry_pending',failures=record['failures']+1,error=output.get('error','Worker ended without complete output'))
            state['status']='retry_pending';persist()
            return 1
    state['status']='complete';persist()
    return 0


if __name__=='__main__': sys.exit(run())
