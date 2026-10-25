"""CLI; imports torch only after read-only preflight and explicit execution gate."""
import argparse
import concurrent.futures
import importlib.metadata
import importlib.util
import multiprocessing
import os
import platform
import sys
import time
from pathlib import Path
from .contracts import digest, require
from .io import read,write,file_hash
from .data import load
from .budget import Budget,BudgetExhausted,snapshot


def identity_with_code(identity):
    root=Path(__file__).resolve().parent.parent
    identity=dict(identity)
    identity['code']=digest({str(p.relative_to(root)):file_hash(p) for folder in ('relation_transfer','scripts') for p in sorted((root/folder).glob('*')) if p.is_file()})
    identity['environment']=dict(python=platform.python_version(),platform=platform.platform(),torch=importlib.metadata.version('torch') if importlib.util.find_spec('torch') else None,numpy=importlib.metadata.version('numpy') if importlib.util.find_spec('numpy') else None)
    return identity


def preflight(config,data_dir):
    c=read(config); loaded=load(c,data_dir); identity=identity_with_code(loaded[4])
    return c,loaded,identity


def execute(args):
    c,loaded,identity=preflight(args.config,args.data_dir)
    ids,states,labels,roles,_,manifest,estimate=loaded
    require(c['kind']=='synthetic' if args.synthetic else c['kind']=='experiment' and c['budgets_confirmed'] is True and args.confirm_real_run,'execution gate: experiment + budgets_confirmed + --confirm-real-run (or bounded --synthetic)')
    run=Path(args.run_dir); resume=args.command=='resume' or (args.command=='diagnostic' and run.exists())
    if resume:
        require(run.is_dir(),'resume/diagnostic needs existing run')
        require(read(run/'identity.json')==identity,'resume refuses changed code/config/data/environment')
        require(read(run/'status.json')['status']!='COMPLETE' or args.command=='diagnostic','already complete')
    else:
        require(not run.exists(),'new run directory must not exist'); run.mkdir(parents=True)
        write(run/'config.json',c); write(run/'identity.json',identity); write(run/'data_manifest.json',manifest); write(run/'budget_estimate.json',estimate)
        Budget(run/'budget.sqlite').create(c)
    lock=run/'coordinator.lock'
    # Exclusive lock: never silently delete stale lock after an unclean process exit.
    with lock.open('x',encoding='utf-8') as f: f.write(str(os.getpid()))
    budget=Budget(run/'budget.sqlite'); active=False; status=dict(status='RUNNING',stage='initializing',exit_code=None)
    def update(stage):
        status['stage']=stage; status['updated_unix']=time.time(); write(run/'status.json',status)
    try:
        budget.start(); active=True
        require(importlib.util.find_spec('torch') is not None and importlib.util.find_spec('numpy') is not None,'missing dependency: install explicitly before neural run')
        from .training import fit_priors,policy_worker,configure,load_checkpoint
        from .diagnostics import prior_quality,paid_candidates,sensitivity
        from .evaluation import evaluate
        from .report import analyze
        configure(c)
        update('prior_and_geometry')
        prior,geometry=fit_priors(c,ids,labels,roles,run,identity,budget)
        update('prior_diagnostic'); prior_quality(c,ids,labels,roles,prior,geometry,run)
        if args.command=='diagnostic':
            update('diagnostic_dependency_training')
            for group in ('A','D'):
                policy_worker(c,group,c['train_seeds'][0],ids,states,roles,str(run),identity)
            update('paid_candidate_diagnostic'); paid_candidates(c,ids,states,roles,run,identity,budget)
            status.update(status='INCOMPLETE',exit_code=0); update('diagnostic_only')
            analyze(run,run,allow_existing=True); return 0
        if c['stop_after_prior'] and args.command!='resume':
            status.update(status='INCOMPLETE',exit_code=0); update('intentional_stop_after_prior'); return 0
        update('policy_training')
        # Spawn explicit callable arguments; no shell interpolation; bounded workers.
        with concurrent.futures.ProcessPoolExecutor(max_workers=c['workers'],mp_context=multiprocessing.get_context('spawn')) as pool:
            pending=[pool.submit(policy_worker,c,g,s,ids,states,roles,str(run),identity) for g in c['groups'] for s in c['train_seeds']]
            try:
                for future in concurrent.futures.as_completed(pending):
                    future.result(); budget.check()
            except BaseException:
                budget.abort()
                for future in pending: future.cancel()
                raise
        update('paid_candidate_diagnostic'); paid_candidates(c,ids,states,roles,run,identity,budget)
        update('serial_development_evaluation'); evaluate(c,ids,states,roles,run,identity,budget)
        update('sensitivity'); sensitivity(c,ids,states,roles,run,identity,budget)
        budget.check(); status.update(status='COMPLETE',exit_code=0); update('report')
        analyze(run,run,allow_existing=True); update('complete'); return 0
    except BaseException as exc:
        code=3 if isinstance(exc,BudgetExhausted) else 130 if isinstance(exc,KeyboardInterrupt) else 2
        status.update(status='BUDGET_EXHAUSTED' if code==3 else 'INCOMPLETE',exit_code=code,error=type(exc).__name__+': '+str(exc)); update(status.get('stage','failed'))
        try:
            from .report import analyze
            analyze(run,run,allow_existing=True)
        except Exception as analysis_exc:
            # Never silent: record failure so status/summary cannot look consistent with a stale report.
            message=type(analysis_exc).__name__+': '+str(analysis_exc)
            status['analysis_error']=message; update(status.get('stage','failed'))
            write(run/'analysis_error.json',dict(error=message,run_error=status['error'],updated_unix=time.time()))
            print('analysis failed: '+message,file=sys.stderr)
        print(status['error'],file=sys.stderr); return code
    finally:
        if active: budget.finish()
        lock.unlink()
        # Final report includes closed session wall clock, not worker-time sums.
        if (run/'summary.json').exists():
            summary=read(run/'summary.json'); summary['budget']=snapshot(run/'budget.sqlite')
            if 'analysis_error' in status:
                summary['stale']=dict(reason='latest analyze failed; this summary predates the current status.json',analysis_error=status['analysis_error'],current_status=status['status'])
            write(run/'summary.json',summary)


def main(argv=None):
    parser=argparse.ArgumentParser(); sub=parser.add_subparsers(dest='command',required=True)
    for name in ('preflight','run','resume','diagnostic'):
        p=sub.add_parser(name); p.add_argument('--config',required=True); p.add_argument('--data-dir',required=True)
        if name!='preflight':
            p.add_argument('--run-dir',required=True); p.add_argument('--confirm-real-run',action='store_true'); p.add_argument('--synthetic',action='store_true')
    p=sub.add_parser('status'); p.add_argument('--run-dir',required=True)
    p=sub.add_parser('analyze'); p.add_argument('--run-dir',required=True); p.add_argument('--out-dir',required=True)
    p=sub.add_parser('verify'); p.add_argument('--out-dir',required=True); p.add_argument('--require-neural',action='store_true')
    args=parser.parse_args(argv)
    try:
        if args.command=='preflight':
            c,loaded,identity=preflight(args.config,args.data_dir)
            print(__import__('json').dumps(dict(status='PREFLIGHT_OK',budget=loaded[-1],roles={r:len(v) for r,v in loaded[3].items()},identity=identity),ensure_ascii=False,indent=2)); return 0
        if args.command=='status':
            run=Path(args.run_dir); budget=snapshot(run/'budget.sqlite'); c=read(run/'config.json')
            tasks=[g+'_'+str(s) for g in c['groups'] for s in c['train_seeds']]
            print(__import__('json').dumps(dict(status=read(run/'status.json'),budget=budget,
                completed_policy_tasks=sum(budget['progress'].get(t,0)>=c['policy']['steps'] for t in tasks),total_policy_tasks=len(tasks)),ensure_ascii=False,indent=2)); return 0
        if args.command=='analyze':
            from .report import analyze
            analyze(args.run_dir,args.out_dir); return 0
        if args.command=='verify':
            from .verify import verify
            return verify(args.out_dir,args.require_neural)
        return execute(args)
    except Exception as exc:
        print(type(exc).__name__+': '+str(exc),file=sys.stderr); return 2


if __name__=='__main__': sys.exit(main())
