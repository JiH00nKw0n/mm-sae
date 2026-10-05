"""Execute config-defined Python jobs sequentially, with durable status and restart checks."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import yaml

from mm_sae.io import atomic_json, code_digest, file_lock, sha256


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=Path, required=True)
    args = parser.parse_args()
    path = args.config.resolve()
    options = yaml.safe_load(path.read_text())
    root = (path.parent / options['output']).resolve()
    logs = root / 'logs'
    logs.mkdir(parents=True, exist_ok=True)
    jobs = options['jobs']
    # CLI config files and script bodies are also part of the immutable launch record.
    inputs = {str(path): sha256(path)}
    for job in jobs:
        for item in job['argv']:
            p = Path(item)
            if p.is_file() and p.suffix in {'.py', '.yaml'}:
                inputs[str(p.resolve())] = sha256(p)
    signature = dict(code=code_digest(), inputs=inputs, jobs=jobs)
    receipt = root / 'signature.json'
    with file_lock(root / '.lock', blocking=False):
        if receipt.exists() and json.loads(receipt.read_text()) != signature:
            raise ValueError('Sequence code/config changed; use a new output directory')
        atomic_json(receipt, signature)
        status: dict[str, Any] = dict(state='running', pid=os.getpid(), started_at=time.time(),
                      jobs={job['name']: {'state': 'pending'} for job in jobs})
        try:
            for job in jobs:
                name = job['name']
                mark = logs / f'{name}.done.json'
                if mark.exists():
                    status['jobs'][name] = dict(state='completed', resumed=True)
                    continue
                start = time.time()
                status['jobs'][name] = dict(state='running', started_at=start)
                with (logs / f'{name}.log').open('a') as stream:
                    command = [sys.executable, *job['argv']]
                    child = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
                    while child.poll() is None:
                        status.update(current_job=name, child_pid=child.pid, updated_at=time.time())
                        status['jobs'][name]['elapsed_seconds'] = time.time()-start
                        atomic_json(root / 'scheduler.json', status)
                        time.sleep(10)
                status['jobs'][name].update(state='completed' if child.returncode == 0 else 'failed',
                                           seconds=time.time()-start, returncode=child.returncode)
                if child.returncode:
                    raise RuntimeError(f'{name} failed; see {logs / (name + ".log")}')
                atomic_json(mark, dict(command=command, seconds=time.time()-start))
            status['state'] = 'completed'
        except BaseException as error:
            status.update(state='failed', error=str(error))
            raise
        finally:
            status['updated_at'] = time.time()
            atomic_json(root / 'scheduler.json', status)


if __name__ == '__main__':
    main()
