"""Wait for Stage 36, then supervise the conditional follow-up queue independently."""
import json
import os
import subprocess
import sys
import time
import psutil
from datetime import datetime,timezone
from src.aftercare_plan import ROOT,OUT,PARENT,choose_branch
from src.acquire_scenarios import save_json


def read(path): return json.loads(path.read_text()) if path.exists() else {}


def service():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'service.lock').open('a+b');lock.seek(0)
    if os.name=='nt':
        import msvcrt
        try: msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError: return
    status_path=ROOT/'results/data_acquisition/aftercare_supervisor.json'
    def status(**values): save_json(status_path,dict(pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),**values))
    try:
        while True:
            if (ROOT/'pause.flag').exists(): status(status='paused_by_user');return
            parent=read(PARENT/'queue.json')
            if parent.get('status')=='complete' and read(PARENT/'stage36/status.json').get('status')=='complete': break
            status(status='waiting_for_stage36',parent_status=parent.get('status'),parent_stage=parent.get('active_stage'))
            time.sleep(15)
        route_file=OUT/'route.json'
        if not route_file.exists():
            gate=read(PARENT/'stage33/decision.json')
            if 'passed' not in gate: status(status='needs_attention',error='Missing development decision');return
            exposed=(PARENT/'stage35/test_lock.json').exists() or (PARENT/'stage35/test_report.json').exists() or any((PARENT/'stage35').glob('*_predictions.npz'))
            branch=choose_branch(gate['passed'],exposed)
            save_json(route_file,dict(branch=branch,development_passed=gate['passed'],test_exposed=exposed,
                decided_utc=datetime.now(timezone.utc).isoformat(),
                rule='Test exposure always disables automatic retraining, regardless of test scores.'))
        if read(route_file)['branch']=='inconsistent_holdout_state':
            status(status='needs_attention',error='Gate passed but completed queue has no test record');return
        executable=str(__import__('pathlib').Path(sys.executable).with_name('python.exe'))
        attempt=0
        while True:
            if read(OUT/'queue.json').get('status')=='complete': status(status='finished');return
            if (ROOT/'pause.flag').exists(): status(status='paused_by_user');return
            attempt+=1
            with (OUT/'service.log').open('a',encoding='utf-8') as log:
                process=None
                previous=read(status_path)
                if previous.get('worker_pid') and previous.get('worker_created'):
                    try:
                        candidate=psutil.Process(previous['worker_pid'])
                        if candidate.create_time()==previous['worker_created'] and 'src.aftercare_queue' in candidate.cmdline(): process=candidate
                    except (psutil.NoSuchProcess,psutil.AccessDenied): pass
                if process is None:
                    launched=subprocess.Popen([executable,'-u','-m','src.aftercare_queue'],cwd=ROOT,stdout=log,stderr=log,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                    process=psutil.Process(launched.pid)
                while True:
                    status(status='running',worker_pid=process.pid,worker_created=process.create_time(),attempt=attempt)
                    try: code=process.wait(timeout=15);break
                    except psutil.TimeoutExpired: pass
            if code is None and read(OUT/'queue.json').get('status')=='complete': code=0
            if code==0: status(status='finished' if read(OUT/'queue.json').get('status')=='complete' else 'paused_by_user');return
            if code==2 or attempt>=6: status(status='needs_attention',exit_code=code);return
            status(status='retrying_in_15_seconds',exit_code=code);time.sleep(15)
    finally: lock.close()


if __name__=='__main__': service()
