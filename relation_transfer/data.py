import random
from pathlib import Path
from .contracts import digest, require, validate_dataset, validate_routes, instance_hash, build_relations
from .io import read, file_hash
from .config import validate, budget_estimate


def load(c, directory):
    validate(c)
    directory=Path(directory)
    data,states,refs=[read(directory/name) for name in ('data.json','states.json','references.json')]
    validate_dataset(data)
    ids={i['id']:i for i in data}
    for i in data:
        require(len(i['demands'])==c['customers']+1 and i['distribution']==c['distribution'], 'size/distribution mismatch')
        require(c['kind']!='synthetic' or i.get('synthetic') is True, 'synthetic data required')
    train=sorted(i['id'] for i in data if i['split']=='train')
    require(len(train)==c['counts']['prior_train']+c['counts']['policy_train'], 'train count')
    # Group-preserving subset sum: never split a family to meet a requested count.
    families={}
    for key in train: families.setdefault(ids[key].get('family_id',key),[]).append(key)
    groups=sorted(families.values(),key=lambda a:a[0]); random.Random(c['split_seed']).shuffle(groups)
    target=c['counts']['prior_train']; solutions={0:[]}
    for group in groups:
        for n,chosen in list(solutions.items()):
            if n+len(group)<=target and n+len(group) not in solutions: solutions[n+len(group)]=chosen+group
    require(target in solutions,'family groups cannot satisfy exact role counts')
    prior=set(solutions[target])
    roles={'prior_train':sorted(prior),'policy_train':sorted(set(train)-prior),
        'development':sorted(i['id'] for i in data if i['split']=='validation'),
        'historical_test':sorted(i['id'] for i in data if i['split']=='test')}
    role_of={key:r for r,keys in roles.items() for key in keys}
    seen_family={}
    for r,keys in roles.items():
        require(len(keys)==c['counts'][r], 'role count '+r)
        for key in keys:
            family=ids[key].get('family_id',key)
            require(family not in seen_family or seen_family[family]==r,'family crosses roles')
            seen_family[family]=r
    by_state={key:[] for key in ids}; seen=set()
    for s in states:
        require(set(s)=={'instance_id','instance_hash','routes'},'state fields')
        i=ids[s['instance_id']]
        require(s['instance_hash']==instance_hash(i),'state identity')
        validate_routes(i,s['routes']); h=digest(s)
        require(h not in seen,'duplicate state'); seen.add(h); by_state[i['id']].append(s)
    for key,i in ids.items(): require(len(by_state[key])==(c['train_states'] if i['split']=='train' else 1),'state count')
    require(set(refs)==set(train),'references must cover only original train')
    labels={}
    for key in train:
        rr=refs[key]; require(len(rr)==1,'first version requires M=1')
        rc=dict(M=1,duplicates='keep_equal_weight',seeds=[rr[0]['seed']],budget=rr[0]['budget'],solver=rr[0]['solver'])
        labels[key]=build_relations(ids[key],rr,rc)['p']
    estimate=budget_estimate(c); require(estimate['remaining_for_timed_search']>0,'fixed work exceeds Repair budget')
    identity=dict(config=digest(c),data=digest(data),states=digest(states),references=digest(refs))
    manifest=dict(roles=roles,instances=[dict(id=k,role=role_of[k],original_split=i['split'],instance_hash=instance_hash(i),state_hashes=[digest(s) for s in by_state[k]],reference_hash=digest(refs[k]) if k in refs else None) for k,i in ids.items()],
        files={name:file_hash(directory/name) for name in ('data.json','states.json','references.json')},
        limitation='Cannot recognize arbitrary node-renumbering or geometric equivalence; family_id must disclose variants.')
    return ids,by_state,labels,roles,identity,manifest,estimate
