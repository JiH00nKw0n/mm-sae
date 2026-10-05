"""Wait for an already authorized run, collect immutable outputs, and build figures.

This is a foreground collector, not a scheduled service. It never starts experiments.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shlex
import subprocess
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--host', required=True)
    p.add_argument('--remote-root', required=True)
    p.add_argument('--local-root', type=Path, required=True)
    p.add_argument('--baseline', type=Path, required=True)
    p.add_argument('--diagnosis-baseline', type=Path, required=True)
    p.add_argument('--slide-builder', type=Path, required=True)
    a = p.parse_args()
    a.local_root.mkdir(parents=True, exist_ok=True)
    status = a.local_root/'collection.json'
    ssh = ['ssh', '-o', 'ConnectTimeout=15', '-o', 'ServerAliveInterval=10', '-o', 'ServerAliveCountMax=2', a.host]
    failures = 0
    while True:
        try:
            run = subprocess.run(ssh+['cat '+shlex.quote(a.remote_root+'/scheduler.json')], check=True,
                                 capture_output=True, text=True, timeout=50)
            state = json.loads(run.stdout)
            failures = 0
        except (subprocess.SubprocessError, ValueError) as e:
            failures += 1
            status.write_text(json.dumps(dict(state='connection_retry', failures=failures, error=str(e), time=time.time())))
            print(f'Connection retry {failures}', flush=True)
            if failures >= 10:
                raise
            time.sleep(30)
            continue
        status.write_text(json.dumps(dict(state=state['state'], remote=state, time=time.time()), indent=2))
        complete = sum(r['state']=='completed' for r in state['jobs'].values())
        print(json.dumps(dict(time=time.time(), state=state['state'], stages_completed=complete,
                              stages_total=len(state['jobs']))), flush=True)
        if state['state']=='failed':
            raise RuntimeError(state.get('error','Remote experiment failed'))
        if state['state']=='completed':
            break
        time.sleep(30)
    subprocess.run(['rsync','-az','--timeout=120','-e','ssh -o ConnectTimeout=15',
                    a.host+':'+a.remote_root+'/',str(a.local_root)+'/'],check=True)
    # Independently verify every downloaded source file before making tables or slides.
    script='''import pathlib,hashlib,json
root=pathlib.Path(%r)
print(json.dumps({str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
                 for p in root.rglob('*') if p.is_file()}))
''' % a.remote_root
    remote = subprocess.run(ssh+['python3 -'],input=script,text=True,capture_output=True,check=True,timeout=120)
    hashes=json.loads(remote.stdout)
    for name, expected in hashes.items():
        actual=hashlib.sha256((a.local_root/name).read_bytes()).hexdigest()
        if actual!=expected:
            raise RuntimeError(f'Checksum differs: {name}')
    (a.local_root/'download_verified.json').write_text(json.dumps(dict(files=len(hashes), sha256=hashes),indent=2))
    subprocess.run([sys.executable,'scripts/summarize_selection_comparison.py',str(a.local_root),
                    '--baseline',str(a.baseline),'--diagnosis-baseline',str(a.diagnosis_baseline)],check=True)
    subprocess.run([sys.executable,str(a.slide_builder)],check=True)
    status.write_text(json.dumps(dict(state='slides_built_awaiting_visual_review', files_verified=len(hashes), time=time.time()),indent=2))
    print('Results verified and slides built. Visual and scientific review remain.',flush=True)


if __name__=='__main__':
    main()
