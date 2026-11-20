"""Resume Part 1 transfers on D:, CRC-check extraction, then remove archives.

Uses the public, logged-out Dropbox Transfer endpoints used by its download page.
No browser cookies or account credentials are needed. URLs can expire: update
TRANSFER_URLS from the official DeepSense page if that happens.
"""
import argparse
import bisect
import csv
import hashlib
import io
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import sys
import time
import tempfile
from datetime import datetime, timezone
import zipfile
from concurrent.futures import ThreadPoolExecutor

import requests

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / 'data' / 'raw'
REPORTS = ROOT / 'results' / 'data_acquisition'
TRANSFER_URLS = {
    36: 'https://www.dropbox.com/transfer/AAAAACewLi5uThal3r-JVWxFsAXpWl2ROAQLOdwrsRr1KCr6dtFe6Ps',
    37: 'https://www.dropbox.com/transfer/AAAAAKPth2dQOmBxWa8Qo5fOmmcc1FQqyNScK2-d-rC_C-nqG-8PAeY',
    38: 'https://www.dropbox.com/transfer/AAAAAOXEUm36HgN8dt-Wqy9iTkIiEp5I90LuOsuu8uif30meVecW-sc',
    39: 'https://www.dropbox.com/transfer/AAAAACz9W22DotzhQlfZZGG3RTHmhTIqdaOLqMC6HrM1i13hCA02tCI',
}
MODALITIES = {'gps': ['unit1_gps1', 'unit2_gps1'],
              'pwr': [f'unit1_pwr{i}' for i in range(1, 5)],
              'rgb180': ['unit1_rgb5', 'unit1_rgb6']}
RECONNECT_SECONDS = 15


def retryable_network_error(error):
    if isinstance(error, requests.HTTPError):
        return error.response.status_code in (408, 429) or error.response.status_code >= 500
    return isinstance(error, (requests.ConnectionError, requests.Timeout,
                              requests.exceptions.ChunkedEncodingError,
                              requests.exceptions.ContentDecodingError))


def wait_for_connection(label):
    """Check Dropbox reachability every 15 seconds until the outage ends."""
    print(f'{label}: connection interrupted; checking every {RECONNECT_SECONDS}s; progress retained', flush=True)
    next_check = time.monotonic() + RECONNECT_SECONDS
    while True:
        time.sleep(max(0, next_check - time.monotonic()))
        next_check = time.monotonic() + RECONNECT_SECONDS
        try:
            with requests.head('https://www.dropbox.com/', timeout=(5, 5),
                               allow_redirects=True) as response:
                if response.status_code < 500 and response.status_code not in (408, 429):
                    print(f'{label}: connection available; resuming', flush=True)
                    return
        except requests.RequestException:
            pass
        print(f'{label}: still unavailable; next check in {RECONNECT_SECONDS}s', flush=True)


def network_request(operation, label):
    """Retry network/service outages indefinitely, including metadata and URL refresh."""
    while True:
        try:
            response = operation()
            try:
                response.raise_for_status()
            except requests.RequestException:
                response.close()
                raise
            return response
        except requests.RequestException as error:
            if not retryable_network_error(error):
                raise
            wait_for_connection(label)


def save_json(path, data):
    """Atomic JSON publication, tolerating brief Windows reader/AV locks.

    Never truncate the live file. Unique temporary files also isolate writers.
    Persistent permission failures still surface after a bounded retry period.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(mode='w',encoding='utf-8',dir=path.parent,
            prefix=path.name+'.',suffix='.tmp',delete=False) as stream:
        temp=Path(stream.name)
        json.dump(data,stream,indent=2)
    try:
        for attempt in range(101):
            try:
                temp.replace(path)
                return
            except PermissionError:
                if attempt==100: raise
                time.sleep(.1)
    finally:
        temp.unlink(missing_ok=True)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f, 'sha256').hexdigest()


class Transfer:
    def __init__(self, url):
        self.session = requests.Session()
        network_request(lambda: self.session.get(url, timeout=(10, 30)), 'Transfer access').close()
        self.transfer_id = 'cap_pid_ft:' + url.rsplit('/', 1)[-1]
        self.headers = {'X-CSRF-Token': self.session.cookies.get('t', ''),
                        'X-Dropbox-Uid': '-1'}

    def rpc(self, operation, **params):
        response = network_request(lambda: self.session.post(
            'https://www.dropbox.com/2/file_transfers/files/' + operation + '_logged_out',
            json={'transfer_id': self.transfer_id, **params},
            headers=self.headers, timeout=(10, 30)), f'Transfer {operation}')
        with response:
            return response.json()

    def files(self):
        return self.rpc('list')['files']

    def download_url(self, file):
        info = self.rpc('download', requested_files_by_id=[file['transfer_file_id']],
                        is_preview_download=False)
        if json.loads(info['post_params']):
            raise RuntimeError('Unexpected download POST parameters; inspect before continuing')
        return info['url']


def download(url, path, expected, refresh_url=None):
    """Only promote a partial file after HTTP and byte-count checks succeed."""
    if path.exists():
        if path.stat().st_size != expected:
            raise ValueError(f'Existing size mismatch: {path}')
        return
    partial = path.with_name(path.name + '.partial')
    path.parent.mkdir(parents=True, exist_ok=True)
    failures = 0
    last_report = -1
    while True:
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > expected:
            raise ValueError(f'Oversized partial: {partial}')
        if offset == expected:
            partial.replace(path)
            return
        if shutil.disk_usage(path.parent).free < expected - offset + 5 * 1024**3:
            raise OSError('Insufficient disk space; keeping partial for resume')
        end = min(offset + 32 * 1024**2, expected) - 1
        try:
            with requests.get(url, headers={'Range': f'bytes={offset}-{end}'},
                              stream=True, timeout=(30, 90)) as response:
                response.raise_for_status()
                cr = response.headers.get('Content-Range', '')
                if response.status_code != 206 or cr != f'bytes {offset}-{end}/{expected}':
                    raise ValueError('Server did not honor the exact requested byte range')
                with partial.open('ab' if offset else 'wb') as output:
                    for chunk in response.iter_content(256 * 1024):
                        output.write(chunk)
            if partial.stat().st_size != end + 1:
                raise requests.ConnectionError(f'Incomplete download: {path.name}')
            failures = 0
            progress = (end + 1) // (256 * 1024**2)
            if progress != last_report:
                print(f'{path.name}: {end+1:,}/{expected:,} bytes', flush=True)
                last_report = progress
            if end + 1 == expected:
                partial.replace(path)
                print(f'Downloaded {path.name}: {expected:,} bytes', flush=True)
                return
            continue
        except requests.RequestException as error:
            if isinstance(error, requests.HTTPError) and error.response.status_code in (401, 403):
                failures += 1
                if refresh_url is None or failures >= 5:
                    raise
                url = refresh_url()
            elif retryable_network_error(error):
                wait_for_connection(path.name)
            else:
                raise


class SplitReader(io.RawIOBase):
    """Seek across .zip.001 volumes without making a second joined archive."""
    def __init__(self, paths):
        super().__init__()
        self.handles = [p.open('rb') for p in paths]
        self.ends = []
        size = 0
        for p in paths:
            size += p.stat().st_size
            self.ends.append(size)
        self.size, self.pos = size, 0

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, offset, whence=0):
        pos = offset if whence == 0 else self.pos + offset if whence == 1 else self.size + offset
        if pos < 0: raise ValueError('Negative seek')
        self.pos = pos
        return pos

    def read(self, size=-1):
        remaining = self.size - self.pos if size < 0 else min(size, self.size - self.pos)
        chunks = []
        while remaining > 0:
            index = bisect.bisect_right(self.ends, self.pos)
            start = self.ends[index - 1] if index else 0
            handle = self.handles[index]
            handle.seek(self.pos - start)
            chunk = handle.read(min(remaining, self.ends[index] - self.pos))
            if not chunk: raise EOFError('Truncated archive volume')
            chunks.append(chunk)
            self.pos += len(chunk)
            remaining -= len(chunk)
        return b''.join(chunks)

    def close(self):
        for handle in self.handles: handle.close()
        super().close()


def contained(base, name):
    path = (base / name).resolve()
    if not path.is_relative_to(base.resolve()):
        raise ValueError(f'Path escapes destination: {name}')
    return path


def verify_references(scenario, modality):
    folder = RAW / f'scenario{scenario}'
    with (folder / f'scenario{scenario}.csv').open(newline='', encoding='utf-8-sig') as f:
        rows = list(csv.DictReader(f))
    references = {col.replace('_', '/'): set() for col in MODALITIES[modality]}
    for row in rows:
        for col in MODALITIES[modality]:
            name = row[col].replace('\\', '/').removeprefix('./')
            relative = PurePosixPath(name)
            parent = col.replace('_', '/')
            if len(relative.parts) != 3 or str(relative.parent) != parent or relative.name in ('.', '..'):
                raise ValueError(f'Unexpected CSV modality path: {name}')
            references[parent].add(relative.name)
    missing = []
    for parent, names in references.items():
        directory = contained(folder, parent)
        found = set()
        if directory.is_dir():
            with os.scandir(directory) as entries:
                found = {e.name for e in entries if e.is_file(follow_symlinks=False)
                         and e.stat(follow_symlinks=False).st_size > 0}
        missing.extend(f'{parent}/{name}' for name in names - found)
    if missing:
        raise ValueError(f'{len(missing)} missing/empty {modality} files; first: {missing[:3]}')
    return {'csv_rows': len(rows), 'unique_referenced_files': sum(map(len, references.values()))}


def extract_verify(scenario, modality, paths):
    scenario_root = RAW / f'scenario{scenario}'
    with SplitReader(paths) as stream, zipfile.ZipFile(stream) as archive:
        entries = archive.infolist()
        total = sum(i.file_size for i in entries)
        if shutil.disk_usage(RAW).free < total + 5 * 1024**3:
            raise OSError('Insufficient extraction space; archives retained')
        count = 0
        for info in entries:
            dest = contained(RAW, info.filename)
            if not dest.is_relative_to(scenario_root.resolve()):
                raise ValueError('Unexpected scenario in archive')
            if (info.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Archive symlink refused')
            if info.is_dir():
                dest.mkdir(parents=True, exist_ok=True)
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            temporary = dest.with_name(dest.name + '.extracting')
            # Reading to EOF checks the ZIP member CRC before promotion.
            with archive.open(info) as source, temporary.open('wb') as output:
                shutil.copyfileobj(source, output, 1024**2)
            if temporary.stat().st_size != info.file_size:
                raise ValueError('Extraction size mismatch')
            temporary.replace(dest)
            count += 1
    result = verify_references(scenario, modality)
    result.update(extracted_files=count, extracted_bytes=total, crc_verified=True)
    return result


def acquire(scenarios, workers, metadata_only=False):
    RAW.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)
    # Obtain all manifests first, so later scenarios are documented before large transfers.
    manifests = {}
    for scenario in scenarios:
        files = Transfer(TRANSFER_URLS[scenario]).files()
        prefix = f'scenario{scenario}'
        selected = [f for f in files if f['relative_path'] in (prefix+'.csv', prefix+'.p')
                    or any(f['relative_path'].startswith(prefix+'_'+m+'.zip.') for m in MODALITIES)]
        if not selected or any(Path(f['relative_path']).name != f['relative_path'] for f in selected):
            raise ValueError('Unexpected transfer contents')
        manifests[scenario] = selected
        save_json(REPORTS / f'scenario{scenario}_manifest.json', {
            'source': TRANSFER_URLS[scenario], 'files': [
                {k: f[k] for k in ('relative_path', 'file_size', 'transfer_file_id')} for f in selected]})
        print(f'Scenario {scenario}: {sum(f["file_size"] for f in selected)/1024**3:.2f} GiB selected', flush=True)
    for scenario, files in manifests.items():
        for file in files:
            if not file['relative_path'].endswith(('.csv', '.p')): continue
            dest = RAW / f'scenario{scenario}' / file['relative_path']
            if not dest.exists():
                download(Transfer(TRANSFER_URLS[scenario]).download_url(file), dest, file['file_size'])
            if dest.stat().st_size != file['file_size']: raise ValueError('Metadata size mismatch')
    if metadata_only: return
    for scenario, files in manifests.items():
        for modality in MODALITIES:
            receipt = REPORTS / f'scenario{scenario}_{modality}.json'
            if receipt.exists() and json.loads(receipt.read_text())['status'] == 'verified_archives_deleted':
                verify_references(scenario, modality)
                print(f'Skipping verified Scenario {scenario} {modality}', flush=True)
                continue
            group = sorted([f for f in files if f'_{modality}.zip.' in f['relative_path']], key=lambda f:f['relative_path'])
            if not group: raise ValueError(f'Missing {scenario} {modality} archive')
            if [int(f['relative_path'].rsplit('.',1)[1]) for f in group] != list(range(1,len(group)+1)):
                raise ValueError('Missing archive volume')
            paths = [RAW / '_zips' / f'scenario{scenario}' / f['relative_path'] for f in group]
            needed = sum(f['file_size'] for f,p in zip(group,paths) if not p.exists())
            if shutil.disk_usage(RAW).free < needed + 5*1024**3: raise OSError('Not enough space for archive group')
            jobs = []
            with ThreadPoolExecutor(max_workers=workers) as executor:
                transfer = Transfer(TRANSFER_URLS[scenario])
                for file,path in zip(group,paths):
                    if path.exists() and path.stat().st_size == file['file_size']: continue
                    url = transfer.download_url(file)
                    jobs.append(executor.submit(download, url, path, file['file_size'],
                        lambda file=file, scenario=scenario: Transfer(TRANSFER_URLS[scenario]).download_url(file)))
                for job in jobs: job.result()
            hashes = {p.name: digest(p) for p in paths}
            print(f'Extracting Scenario {scenario} {modality}', flush=True)
            result = extract_verify(scenario, modality, paths)
            result.update(status='verified', archive_sha256=hashes, source=TRANSFER_URLS[scenario])
            save_json(receipt, result)
            # Only the exact verified archive group under the workspace may be removed.
            for path in paths:
                checked = contained(RAW / '_zips', str(path.relative_to(RAW / '_zips')))
                checked.unlink()
            result['status'] = 'verified_archives_deleted'
            save_json(receipt, result)
            print(f'Verified Scenario {scenario} {modality}; removed archives', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--scenarios', nargs='+', type=int, choices=TRANSFER_URLS, default=list(TRANSFER_URLS))
    parser.add_argument('--workers', type=int, choices=range(1,5), default=2)
    parser.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args()
    status_path = REPORTS / 'run_status.json'
    status = {'status': 'running', 'pid': os.getpid(), 'scenarios': args.scenarios,
              'started_utc': datetime.now(timezone.utc).isoformat(), 'workers': args.workers}
    save_json(status_path, status)
    try:
        acquire(args.scenarios, args.workers, args.metadata_only)
        if not args.metadata_only and set(args.scenarios) == set(TRANSFER_URLS):
            sys.path.insert(0, str(ROOT))
            from src.audit_scenario_data import audit
            audit()
        status['status'] = 'metadata_complete' if args.metadata_only else 'complete'
    except Exception as error:
        status.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        status['updated_utc'] = datetime.now(timezone.utc).isoformat()
        save_json(status_path, status)
