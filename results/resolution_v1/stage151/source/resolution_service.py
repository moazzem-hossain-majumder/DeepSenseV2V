import json
import os
import subprocess
import sys
import time
from datetime import datetime,timezone
from pathlib import Path
import psutil
from src.resolution_plan import ROOT,OUT
from src.acquire_scenarios import save_json


def read(path):return json.loads(path.read_text()) if path.exists() else {}


def service():
    OUT.mkdir(parents=True,exist_ok=True);lock=(OUT/'service.lock').open('a+b');lock.seek(0)
    if os.name=='nt':
        import msvcrt
        try:msvcrt.locking(lock.fileno(),msvcrt.LK_NBLCK,1)
        except OSError:return
    target=ROOT/'results/data_acquisition/resolution_supervisor.json'
    def status(**values):save_json(target,dict(pid=os.getpid(),updated_utc=datetime.now(timezone.utc).isoformat(),**values))
    try:
        for attempt in range(1,7):
            if read(OUT/'queue.json').get('status')=='complete':status(status='finished');return
            if (ROOT/'pause.flag').exists():status(status='paused_by_user');return
            process=None;prior=read(target)
            if prior.get('worker_pid') and prior.get('worker_created'):
                try:
                    p=psutil.Process(prior['worker_pid'])
                    if p.create_time()==prior['worker_created'] and 'src.resolution_queue' in p.cmdline():process=p
                except (psutil.NoSuchProcess,psutil.AccessDenied):pass
            with (OUT/'service.log').open('a',encoding='utf-8') as log:
                if process is None:
                    child=subprocess.Popen([str(Path(sys.executable).with_name('python.exe')),'-u','-m','src.resolution_queue'],cwd=ROOT,stdout=log,stderr=log,
                        creationflags=subprocess.CREATE_NO_WINDOW if os.name=='nt' else 0)
                    process=psutil.Process(child.pid)
                while True:
                    status(status='running',worker_pid=process.pid,worker_created=process.create_time(),attempt=attempt)
                    try:code=process.wait(timeout=15);break
                    except psutil.TimeoutExpired:pass
            if code==0 or read(OUT/'queue.json').get('status')=='complete':
                status(status='finished' if read(OUT/'queue.json').get('status')=='complete' else 'paused_by_user');return
            if code==2 or attempt==6:status(status='needs_attention',exit_code=code);return
            status(status='retrying_in_15_seconds',exit_code=code);time.sleep(15)
    finally:lock.close()


if __name__=='__main__':service()






