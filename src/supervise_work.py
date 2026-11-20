"""Keep a download or explicitly selected training job alive across process failures.

Download: python -m src.supervise_work
Training (after data/protocol gates): python -m src.supervise_work --job train -- --model P3 --seeds 42,7,13,21,101
Training is resumed from the existing checkpoint and does not require connectivity.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / 'results' / 'data_acquisition'


def write_status(path, **values):
    values['updated_utc'] = datetime.now(timezone.utc).isoformat()
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(values, indent=2), encoding='utf-8')
    temp.replace(path)


def command_for(job, extra):
    executable = str(Path(sys.executable).with_name('python.exe')) if os.name == 'nt' else sys.executable
    if job == 'download':
        if extra:
            raise ValueError('Download supervisor uses the complete 36–39 queue')
        return [executable, '-u', str(ROOT / 'src' / 'acquire_scenarios.py'), '--workers', '4']
    if job == 'readiness':
        if extra: raise ValueError('Readiness uses the fixed audited smoke matrix')
        return [executable, '-u', '-m', 'src.readiness_pipeline']
    if job == 'pilot':
        if extra: raise ValueError('Pilot uses its versioned configuration')
        return [executable, '-u', '-m', 'src.multiscenario_pilot']
    if job == 'extended':
        if extra: raise ValueError('Extended training uses its versioned configuration')
        return [executable, '-u', '-m', 'src.multiscenario_extended']
    if job == 'diagnostics':
        if extra: raise ValueError('Diagnostics uses a fixed validation-only protocol')
        return [executable, '-u', '-m', 'src.validation_diagnostics']
    if job == 'queue':
        if extra: raise ValueError('Queue uses its reviewed stage plan')
        return [executable, '-u', '-m', 'src.unattended_queue']
    if job == 'publication':
        if extra: raise ValueError('Publication queue uses its predeclared plan')
        return [executable,'-u','-m','src.publication_queue']
    if not extra:
        raise ValueError('Training requires explicit model/experiment arguments')
    return [executable, '-u', str(ROOT / 'train.py'), *extra,
            *([] if '--resume' in extra else ['--resume'])]


def supervise(job, extra):
    REPORTS.mkdir(parents=True, exist_ok=True)
    command = command_for(job, extra)
    # Windows releases this lock automatically if the supervisor crashes.
    lock = (REPORTS / f'{job}_supervisor.lock').open('a+b')
    lock.seek(0)
    if os.name == 'nt':
        import msvcrt
        try:
            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        except OSError:
            lock.close()
            return  # Another supervisor owns this queue.
    status_path = REPORTS / f'{job}_supervisor.json'
    attempt = 0
    try:
        while True:
            diagnostics_status = ROOT / 'results' / 'validation_diagnostics_v1' / 'status.json'
            if job == 'diagnostics' and diagnostics_status.exists() and json.loads(diagnostics_status.read_text()).get('status') == 'complete':
                write_status(status_path, status='finished', pid=os.getpid(), job=job)
                return
            extended_status = ROOT / 'results' / 'multiscenario_extended_v1' / 'status.json'
            if job == 'extended' and extended_status.exists() and json.loads(extended_status.read_text()).get('status') == 'complete':
                write_status(status_path, status='finished', pid=os.getpid(), job=job)
                return
            pilot_status = ROOT / 'results' / 'multiscenario_pilot_v1' / 'status.json'
            if job == 'pilot' and pilot_status.exists() and json.loads(pilot_status.read_text()).get('status') == 'complete':
                write_status(status_path, status='finished', pid=os.getpid(), job=job)
                return
            readiness_status = ROOT / 'results' / 'readiness' / 'pipeline_status.json'
            if job == 'readiness' and readiness_status.exists() and json.loads(readiness_status.read_text()).get('status') == 'complete':
                write_status(status_path, status='finished', pid=os.getpid(), job=job)
                return
            completed = REPORTS / 'run_status.json'
            if job == 'download' and completed.exists():
                previous = json.loads(completed.read_text(encoding='utf-8'))
                if previous.get('status') == 'complete' and set(previous.get('scenarios', [])) == {36, 37, 38, 39}:
                    write_status(status_path, status='finished', pid=os.getpid(), job=job)
                    return
            if job in ('train', 'pilot', 'extended') and (ROOT / 'pause.flag').exists():
                write_status(status_path, status='paused_by_user', pid=os.getpid())
                return
            attempt += 1
            with (REPORTS / f'{job}_worker.log').open('a', encoding='utf-8') as log:
                log.write(f'\n--- Attempt {attempt}: {datetime.now(timezone.utc).isoformat()} ---\n')
                log.flush()
                process = subprocess.Popen(command, cwd=ROOT, stdout=log, stderr=log,
                    creationflags=subprocess.CREATE_NO_WINDOW if os.name == 'nt' else 0)
                while True:
                    write_status(status_path, status='running', pid=os.getpid(),
                                 worker_pid=process.pid, attempt=attempt, job=job)
                    try:
                        code = process.wait(timeout=15)
                        break
                    except subprocess.TimeoutExpired:
                        pass
            if code == 0:
                write_status(status_path, status='finished', pid=os.getpid(), attempt=attempt, job=job)
                return
            if job in ('readiness', 'queue', 'publication') and code == 2:
                write_status(status_path, status='needs_attention', pid=os.getpid(), job=job)
                return
            # Preserve failures requiring a data/code/storage fix instead of repeatedly
            # extracting corrupt data or overwriting outputs on a full disk.
            run_status = REPORTS / 'run_status.json'
            if job == 'download' and run_status.exists():
                record = json.loads(run_status.read_text(encoding='utf-8'))
                error = record.get('error', '')
                if record.get('pid') == process.pid and record.get('status') == 'failed' and error.startswith(
                    ('ValueError:', 'BadZipFile:', 'OSError:', 'PermissionError:', 'FileNotFoundError:')):
                    write_status(status_path, status='needs_attention', pid=os.getpid(), error=error, job=job)
                    return
            write_status(status_path, status='retrying_in_15_seconds', pid=os.getpid(),
                         exit_code=code, attempt=attempt, job=job)
            time.sleep(15)
    finally:
        lock.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', choices=('download', 'train', 'readiness', 'pilot', 'extended', 'diagnostics', 'queue', 'publication'), default='download')
    parser.add_argument('extra', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    extra = args.extra[1:] if args.extra[:1] == ['--'] else args.extra
    supervise(args.job, extra)
