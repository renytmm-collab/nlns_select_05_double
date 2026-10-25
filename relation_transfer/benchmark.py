"""Read-only throughput probe: untrained models, no budget ledger, no files written.

Measures, on the target machine, seconds per policy update (A/R/D; S and G share D's feature path)
and completed proposals per timed search, then projects the policy-training wall clock against
global_max_wall_seconds. Untrained weights do not change per-proposal compute. Run before a formal run:

    python -B -m relation_transfer.benchmark --config configs/xml100_default.json --data-dir /abs/xml100_v3
"""
import argparse
import json
import time
import torch
from .data import load
from .neural import make_prior, make_policy, get_q, neighbors
from .search import search
from .training import Trainer, configure
from .vrp import ordinary_state


class FreeBudget:
    """Probe-only stand-in: never touches budget.sqlite."""
    path=None
    def charge(self,category): pass
    def check(self): pass


class ProbeTrainer(Trainer):
    def payload(self): raise RuntimeError('benchmark never checkpoints')


def chooser(model,q,inst,seed):
    generator=torch.Generator().manual_seed(seed)
    @torch.no_grad()
    def choose(routes):
        h=model.encoder(ordinary_state(inst,routes))
        return model.decode(h,10,generator,q=q,nb=neighbors(routes,len(inst['demands'])-1))[0]
    return choose


def run(c,ids,states,roles,updates=3,searches=3):
    configure(c); prior=make_prior(c).eval().requires_grad_(False); geometry=torch.zeros(2,device=c['device'])
    budget=FreeBudget(); out={}
    for group in ('A','R','D'):
        trainer=ProbeTrainer(c,group,c['train_seeds'][0],ids,states,roles,prior if group=='D' else None,geometry,budget,None)
        trainer.update()  # first call excluded (allocator/thread warmup)
        start=time.perf_counter()
        for _ in range(updates): trainer.update()
        per_update=(time.perf_counter()-start)/updates
        model=trainer.model.eval(); proposals=[]
        for key in roles['development'][:searches]:
            inst=ids[key]; q=get_q(group,inst,prior,geometry,c)
            result=search(inst,states[key][0]['routes'],lambda:chooser(model,q,inst,1),budget,'probe','time',seconds=c['evaluation']['seconds'])
            proposals.append(result['completed'])
        out[group]=dict(seconds_per_update=per_update,timed_proposals=proposals,mean_timed_proposals=sum(proposals)/len(proposals))
    steps,seeds,workers=c['policy']['steps'],len(c['train_seeds']),c['workers']
    per_task={g:out['D' if g in ('S','G') else g]['seconds_per_update']*steps for g in c['groups']}
    training=sum(per_task.values())*seeds/workers
    out['projection']=dict(policy_training_seconds_estimate=training,global_max_wall_seconds=c['global_max_wall_seconds'],
        training_fraction_of_wall_cap=training/c['global_max_wall_seconds'],
        note='S/G use D timing; excludes prior/geometry fit, evaluation, diagnostics and SQLite fsync per Repair; ideal 2-worker packing.')
    return out


def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument('--config',required=True); p.add_argument('--data-dir',required=True)
    p.add_argument('--updates',type=int,default=3); p.add_argument('--searches',type=int,default=3)
    a=p.parse_args(argv)
    with open(a.config,encoding='utf-8') as f: c=json.load(f)
    ids,states,_,roles,*_=load(c,a.data_dir)
    print(json.dumps(run(c,ids,states,roles,a.updates,a.searches),indent=2))


if __name__=='__main__': main()
