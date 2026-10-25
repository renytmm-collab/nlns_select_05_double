import importlib.util
import os
import subprocess
import sys
import unittest
from pathlib import Path
from .contracts import require
from .io import write
from .fixtures import fixture


def verify(out,require_neural=False):
    root=Path(__file__).resolve().parent.parent; out=Path(out).resolve()
    require(out.is_relative_to(root),'verify output must stay inside project workspace')
    require(not out.exists(),'verification output must be new')
    out.mkdir(parents=True); os.environ['RELATION_TEST_ROOT']=str(out)
    os.environ['PYTHONDONTWRITEBYTECODE']='1'
    os.environ['TMP']=os.environ['TEMP']=os.environ['TMPDIR']=str(out)
    os.environ['TORCH_HOME']=str(out/'torch_cache'); os.environ['XDG_CACHE_HOME']=str(out/'cache')
    suite=unittest.defaultTestLoader.discover(str(root/'tests'))
    with (out/'tests.txt').open('w',encoding='utf-8') as stream:
        result=unittest.TextTestRunner(stream=stream,verbosity=2).run(suite)
    neural=importlib.util.find_spec('torch') is not None and importlib.util.find_spec('numpy') is not None
    fixture(out/'fixture'); pipeline_code=None
    if neural and result.wasSuccessful():
        argv=[sys.executable,'-B','-m','relation_transfer.pipeline','run','--config',str(out/'fixture'/'config.json'),'--data-dir',str(out/'fixture'),'--run-dir',str(out/'pipeline'),'--synthetic']
        with (out/'pipeline.log').open('w',encoding='utf-8') as stream:
            pipeline_code=subprocess.run(argv,cwd=root,stdout=stream,stderr=subprocess.STDOUT,check=False).returncode
        if pipeline_code==0:
            require(all((out/'pipeline'/name).exists() for name in ('REPORT.md','summary.json','status.json')),'missing pipeline outputs')
    success=result.wasSuccessful() and (pipeline_code==0 if neural else not require_neural)
    summary=dict(python=sys.version,tests=result.testsRun,failures=len(result.failures),errors=len(result.errors),skipped=[dict(test=str(t),reason=r) for t,r in result.skipped],neural_available=neural,pipeline_exit_code=pipeline_code,require_neural=require_neural,passed=success)
    write(out/'verification.json',summary); print(__import__('json').dumps(summary,ensure_ascii=False,indent=2))
    return 0 if success else 2
