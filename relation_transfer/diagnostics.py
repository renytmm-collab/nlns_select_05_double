import random
import statistics
import time
from pathlib import Path
import torch
from .contracts import digest, require
from .io import write, read
from .neural import get_q, static_input, neighbors, permute
from .vrp import repair, distances, route_cost, ordinary_state
from .statistics import quality, quantiles
from .evaluation import Evaluator


def prior_quality(c,ids,labels,roles,prior,geometry,run):
    rows=[]; density=read(Path(run)/'geometry.json')['density']; started=time.perf_counter()
    for key in roles['policy_train']:
        inst=ids[key]; n=len(inst['demands'])-1
        matrices={'prior':get_q('D',inst,prior,geometry,c).cpu().tolist(), 'geometry':get_q('G',inst,None,geometry,c).cpu().tolist(), 'constant':[[density]*n for _ in range(n)]}
        targets=[labels[key][i][j] for i in range(n) for j in range(i+1,n)]
        for name,q in matrices.items():
            rows.append(dict(instance_id=key,predictor=name,**quality([q[i][j] for i in range(n) for j in range(i+1,n)],targets,n)))
    paired={}
    for base in ('geometry','constant'):
        for metric in ('bce','brier','average_precision','precision_at_n'):
            ds=[]
            for key in roles['policy_train']:
                a=next(r[metric] for r in rows if r['instance_id']==key and r['predictor']=='prior')
                b=next(r[metric] for r in rows if r['instance_id']==key and r['predictor']==base)
                if a is not None and b is not None: ds.append((b-a) if metric in ('bce','brier') else (a-b))
            rng=random.Random(c['bootstrap_seed']); ci=quantiles([statistics.mean(rng.choices(ds,k=len(ds))) for _ in range(c['bootstrap_samples'])]) if ds else [None,None]
            paired[base+'_'+metric]=dict(mean=statistics.mean(ds) if ds else None,ci95=ci,n=len(ds),differences=ds)
    supported=all(paired[b+'_bce']['ci95'][0]>0 for b in ('geometry','constant'))
    result=dict(rows=rows,paired=paired,status='PRIOR_DESCRIPTIVE_ADVANTAGE' if supported else 'PRIOR_NOT_SUPPORTED',seconds=time.perf_counter()-started,
        rule='Descriptive instance-bootstrap BCE advantage over both controls; no checkpoint selection and no automatic stop.',ranking='Threshold-group AP ties; precision@N fractional boundary ties; AP=null with no positives.')
    write(Path(run)/'prior_diagnostic.json',result)


def candidate_summary(rewards):
    return dict(mean=statistics.mean(rewards),best=max(rewards),improvement_fraction=sum(x>0 for x in rewards)/len(rewards),distribution=rewards)


def paid_candidates(c,ids,states,roles,run,identity,budget):
    run=Path(run); directory=run/'paid_diagnostic'; directory.mkdir(exist_ok=True)
    ev=Evaluator(c,run,identity); seed=c['train_seeds'][0]; ev.load('A',seed); ev.load('D',seed)
    rng=random.Random(20261023); keys=rng.sample(sorted(roles['policy_train']),c['diagnostic']['instances'])
    for key in keys:
        target=directory/(digest(key)+'.json')
        if target.exists(): require(read(target)['identity']==identity,'diagnostic identity'); continue
        inst=ids[key]; routes=states[key][0]['routes']; d=distances(inst)
        choose=ev.factory('A',seed,inst,61001)(); records=[]
        for position in range(max(c['diagnostic']['positions'])+1):
            if position in c['diagnostic']['positions']:
                old=route_cost(routes,d); pool_rng=random.Random(int(digest([key,position,73001])[:15],16))
                pool=[pool_rng.sample(list(range(1,len(inst['demands']))),10) for _ in range(c['diagnostic']['candidates'])]
                rewards=[]
                for action in pool:
                    budget.charge('paid_candidate_random'); new=route_cost(repair(inst,routes,action,d),d); rewards.append((old-new)/max(old,1e-12))
                comparison={}
                for group in ('A','D'):
                    policy=ev.factory(group,seed,inst,82001+position)(); values=[]
                    for _ in range(c['diagnostic']['policy_actions']):
                        action=policy(routes); budget.charge('paid_candidate_policy'); values.append((old-route_cost(repair(inst,routes,action,d),d))/max(old,1e-12))
                    comparison[group]=candidate_summary(values)
                records.append(dict(position=position,routes=routes,initial_cost=old,pool=pool,random=candidate_summary(rewards),policies=comparison))
            if position<max(c['diagnostic']['positions']):
                action=choose(routes); budget.charge('paid_candidate_state_generation'); new=repair(inst,routes,action,d)
                if route_cost(new,d)<route_cost(routes,d): routes=new
        write(target,dict(identity=identity,instance_id=key,records=records,name='有限候选最优值/paid candidate diagnostic'))


@torch.no_grad()
def sensitivity(c,ids,states,roles,run,identity,budget):
    ev=Evaluator(c,run,identity); result=[]
    for group in ('R','D','S','G'):
        for seed in c['train_seeds']:
            model=ev.load(group,seed)
            for key in roles['development'][:c['diagnostic']['sensitivity_states']]:
                budget.check(); inst=ids[key]; routes=states[key][0]['routes']; nb=neighbors(routes,c['customers'])
                q=get_q(group,inst,ev.prior,ev.geometry,c); h=model.encoder(ordinary_state(inst,routes))
                generator=torch.Generator().manual_seed(44001)
                sequence=model.decode(h,10,generator,q=q,nb=nb)[0]
                prefix=sequence[:c['diagnostic']['prefix_length']]
                # Include one forced legal next action solely to expose its pre-action distribution.
                forced=sequence[:len(prefix)+1]
                raw=model.decode(h,len(forced),forced=forced,q=q,nb=nb)[2][-1]
                legal=torch.isfinite(raw); p=raw[legal].exp(); base_stats=model.last_stats[-1]
                constant=torch.full_like(q,read(Path(run)/'geometry.json')['density']); constant.fill_diagonal_(0)
                for variant,qq,off in (('normal',q,False),('off',q,True),('constant',constant,False),('permuted',permute(q,inst,991),False)):
                    logq=model.decode(h,len(forced),forced=forced,q=qq,nb=nb,disable=off)[2][-1]
                    require(torch.equal(torch.isfinite(logq),legal),'intervention legal masks changed')
                    v=logq[legal].exp(); m=(p+v)/2
                    js=.5*(torch.sum(p*(p.clamp_min(1e-30).log()-m.clamp_min(1e-30).log()))+torch.sum(v*(v.clamp_min(1e-30).log()-m.clamp_min(1e-30).log())))
                    result.append(dict(group=group,seed=seed,instance_id=key,prefix=prefix,variant=variant,js=float(js),l1=float((p-v).abs().sum()),argmax_changed=int(raw.argmax())!=int(logq.argmax()),normal=base_stats,intervention=model.last_stats[-1]))
    write(Path(run)/'sensitivity.json',dict(rows=result,repair_calls=0,note='Shared forced prefix and legal candidates; JS in nats. Input replacement may be out of distribution and is not causal proof.'))
