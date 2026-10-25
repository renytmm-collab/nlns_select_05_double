import os
import random
import time
from pathlib import Path
import torch
from .contracts import digest, instance_hash, require
from .io import read, write, file_hash, append
from .neural import make_policy, make_prior, get_q, neighbors
from .training import load_checkpoint
from .search import search,random_factory
from .vrp import ordinary_state


class Evaluator:
    def __init__(self,c,run,identity):
        self.c,self.run,self.identity=c,Path(run),identity
        self.models={}; self.prior=None
        self.geometry=torch.tensor(read(self.run/'geometry.json')['parameters'],device=c['device'])
        self.startup=[]

    def load(self,group,seed):
        if group=='random': return None
        if (group,seed) not in self.models:
            start=time.perf_counter(); model=make_policy(self.c,group,seed)
            model.load_state_dict(load_checkpoint(self.run/(group+'_'+str(seed))/'final.pt',self.identity)['model']); model.eval()
            self.models[group,seed]=model
            if group in ('D','S') and self.prior is None:
                self.prior=make_prior(self.c); self.prior.load_state_dict(load_checkpoint(self.run/'prior_final.pt',self.identity)['model']); self.prior.eval().requires_grad_(False)
            self.startup.append(dict(group=group,seed=seed,seconds=time.perf_counter()-start))
        return self.models[group,seed]

    def factory(self,group,seed,inst,rng_seed):
        model=self.load(group,seed)
        def setup():
            if group=='random': return random_factory(inst,rng_seed)
            q=get_q(group,inst,self.prior,self.geometry,self.c)
            if self.c['device']=='cuda': torch.cuda.synchronize()
            generator=torch.Generator().manual_seed(rng_seed)
            @torch.no_grad()
            def choose(routes):
                h=model.encoder(ordinary_state(inst,routes))
                return model.decode(h,10,generator,q=q,nb=neighbors(routes,len(inst['demands'])-1))[0]
            return choose
        return setup


def task_order(c,states,roles):
    tasks=[]; e=c['evaluation']
    for group in c['groups']+['random']:
        for seed in c['train_seeds'] if group!='random' else [None]:
            for key in roles['development']:
                for state in states[key]:
                    for es in e['seeds']:
                        for mode in ('single','iterations','time'):
                            for repeat in range(e['single_repeats'] if mode=='single' else 1):
                                tasks.append(dict(group=group,train_seed=seed,instance_id=key,state_hash=digest(state),eval_seed=es,mode=mode,repeat=repeat))
    random.Random(e['order_seed']).shuffle(tasks)
    return tasks


def evaluate(c,ids,states,roles,run,identity,budget):
    run=Path(run); directory=run/'evaluation'; directory.mkdir(exist_ok=True)
    tasks=task_order(c,states,roles); order=run/'evaluation_order.json'
    if order.exists(): require(read(order)==tasks,'evaluation order identity')
    else: write(order,tasks)
    evaluator=Evaluator(c,run,identity)
    # All model loading outside online clock; uniform no-warmup policy.
    for g in c['groups']:
        for s in c['train_seeds']: evaluator.load(g,s)
    write(run/'startup.json',dict(loads=evaluator.startup,policy='no explicit warmup; Q recomputed per independent repetition',system_load=os.getloadavg() if hasattr(os,'getloadavg') else None))
    for task in tasks:
        target=directory/(digest(task)+'.json')
        if target.exists():
            row=read(target); require(row['task']==task and row['identity']==identity,'evaluation row identity'); continue
        budget.check(); inst=ids[task['instance_id']]
        state=next((s for s in states[inst['id']] if digest(s)==task['state_hash']),None)
        require(state is not None,'evaluation state missing for task')
        rng_seed=int(digest([task['instance_id'],task['state_hash'],task['eval_seed'],task['mode'],task['repeat']])[:15],16)
        factory=evaluator.factory(task['group'],task['train_seed'],inst,rng_seed)
        result=search(inst,state['routes'],factory,budget,'evaluation_'+task['mode'],task['mode'],c['evaluation']['iterations'],c['evaluation']['seconds'])
        write(target,dict(**task,instance_hash=instance_hash(inst),task=task,identity=identity,**result))
    write(run/'evaluation_complete.json',dict(identity=identity,count=len(tasks),files={p.name:file_hash(p) for p in sorted(directory.glob('*.json'))}))
