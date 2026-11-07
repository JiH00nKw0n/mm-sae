import time,subprocess,json,traceback
from pathlib import Path
ROOT=Path('/mnt/working/mm-sae');OUT=ROOT/'runs/rotation-mass-audit-2026-10-11'
try:
 while not (OUT/'baselines-complete.json').exists():
  if (OUT/'error-baseline.json').exists():raise RuntimeError('Baseline evaluation failed')
  time.sleep(30)
 for module,args in [('rotation_mass_audit_20261011',['--phase','mass']),('rotation_mass_support_20261011',[]),('rotation_mass_audit_20261011',['--phase','sparse'])]:
  subprocess.run([str(ROOT/'.venv/bin/python'),'-u','-m','experiments.'+module,*args],cwd=ROOT,check=True)
except Exception as e:
 (OUT/'error-continuation.json').write_text(json.dumps(dict(error=repr(e),traceback=traceback.format_exc())))
 raise
