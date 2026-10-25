import math
import random
import statistics
from collections import defaultdict
from .contracts import require


def quality(scores,targets,k):
    require(len(scores)==len(targets)>0 and all(y in (0,1) for y in targets),'binary quality inputs')
    require(all(math.isfinite(s) and 0<=s<=1 for s in scores) and type(k) is int and k>0,'probability/ranking inputs')
    pairs=sorted(zip(scores,targets),reverse=True); positives=sum(targets)
    tp=count=0; ap=0.; precision_k=0.; pos=0
    while pos<len(pairs):
        end=pos+1
        while end<len(pairs) and pairs[end][0]==pairs[pos][0]: end+=1
        size=end-pos; p=sum(y for _,y in pairs[pos:end]); tp+=p; count+=size
        if positives: ap+=(p/positives)*(tp/count)
        take=max(0,min(end,k)-pos)
        precision_k+=take*p/size
        pos=end
    eps=1e-12
    clipped=[min(1-eps,max(eps,s)) for s in scores]
    bce=sum(-y*math.log(s)-(1-y)*math.log(1-s) for s,y in zip(clipped,targets))/len(scores)
    return dict(bce=bce,
        brier=sum((s-y)**2 for s,y in zip(scores,targets))/len(scores),average_precision=ap if positives else None,
        precision_at_n=precision_k/min(k,len(scores)),positive_count=positives)


def quantiles(values):
    v=sorted(values)
    def q(p):
        x=(len(v)-1)*p; lo=int(x); hi=min(lo+1,len(v)-1)
        return v[lo]+(v[hi]-v[lo])*(x-lo)
    return [q(.025),q(.975)]


def paired(rows,left,right,mode,c):
    maps={g:{} for g in (left,right)}
    for row in rows:
        if row['group'] not in maps or row['mode']!=mode: continue
        key=tuple(row[k] for k in ('instance_id','state_hash','train_seed','eval_seed','repeat'))
        require(key not in maps[row['group']],'duplicate pairing key')
        maps[row['group']][key]=row
    a,b=maps[left],maps[right]
    require(a and set(a)==set(b),'missing pair keys')
    costs=defaultdict(lambda:[[],[]])
    for key in a:
        require(a[key]['initial_cost']==b[key]['initial_cost'] and a[key]['instance_hash']==b[key]['instance_hash'],'pair initial/data mismatch')
        require(math.isfinite(a[key]['cost']) and math.isfinite(b[key]['cost']) and b[key]['cost']>0,'invalid costs')
        costs[(key[0],key[2])][0].append(a[key]['cost']); costs[(key[0],key[2])][1].append(b[key]['cost'])
    matrix={key:(statistics.mean(v[1])-statistics.mean(v[0]))/statistics.mean(v[1]) for key,v in costs.items()}
    instances=sorted({k[0] for k in matrix}); seeds=c['train_seeds']
    require(set(matrix)=={(i,s) for i in instances for s in seeds},'unbalanced seed matrix')
    by_instance={i:statistics.mean(matrix[i,s] for s in seeds) for i in instances}
    by_seed={str(s):statistics.mean(matrix[i,s] for i in instances) for s in seeds}
    rng=random.Random(c['bootstrap_seed']); boot=[]; two=[]
    for _ in range(c['bootstrap_samples']):
        sampled=rng.choices(instances,k=len(instances)); sampled_seeds=rng.choices(seeds,k=len(seeds))
        boot.append(statistics.mean(by_instance[i] for i in sampled))
        two.append(statistics.mean(matrix[i,s] for i in sampled for s in sampled_seeds))
    values=list(by_instance.values()); tied=[math.isclose(v,0,abs_tol=1e-12) for v in values]
    return dict(mean=statistics.mean(values),median=statistics.median(values),win=sum(v>0 and not t for v,t in zip(values,tied)),tie=sum(tied),
        loss=sum(v<0 and not t for v,t in zip(values,tied)),
        instance_ci95=quantiles(boot),exploratory_instance_seed_ci95=quantiles(two),by_seed=by_seed,by_instance=by_instance)


def supported(x,threshold=0):
    return x['mean']>=threshold and x['mean']>0 and x['instance_ci95'][0]>0 and sum(v>0 for v in x['by_seed'].values())>=4


def conclusions(comparisons,c):
    def positive(key,threshold=0):
        require(key in comparisons,'missing decision comparison '+key)
        return supported(comparisons[key],threshold)
    return dict(solver_gain='SOLVER_GAIN_GO' if positive('D_vs_A_time',c['min_mean_relative_cost_advantage']) else 'NOT_SUPPORTED',
        relation_increment='RELATION_INCREMENT_SUPPORTED' if positive('D_vs_R_time') and positive('D_vs_S_time') else 'NOT_SUPPORTED',
        reference_attribution='REFERENCE_ATTRIBUTION_SUPPORTED' if positive('D_vs_G_time') else 'KNOWLEDGE_ATTRIBUTION_INCONCLUSIVE')


def equal_proposal_evidence(comparisons,c):
    """Descriptive same rules on fixed-iteration (equal proposal count) mode; never overrides time-mode decisions."""
    result={}
    for other in ('A','R','S','G'):
        key='D_vs_'+other+'_iterations'
        if key in comparisons:
            threshold=c['min_mean_relative_cost_advantage'] if other=='A' else 0
            result[key]='SAME_RULE_POSITIVE' if supported(comparisons[key],threshold) else 'SAME_RULE_NOT_POSITIVE'
    return result


def throughput(rows,groups):
    """Mean completed proposals per timed search, and ratio to group A; separates implementation cost from action quality."""
    means={}
    for g in groups:
        values=[r['completed'] for r in rows if r['group']==g and r['mode']=='time']
        if values: means[g]=statistics.mean(values)
    base=means.get('A')
    return {g:dict(mean_completed=m,ratio_to_A=(m/base if base else None)) for g,m in means.items()}
