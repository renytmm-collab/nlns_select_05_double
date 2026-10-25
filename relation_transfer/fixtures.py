import random
from pathlib import Path
from .config import defaults
from .contracts import instance_hash, validate_routes
from .vrp import initial_solution
from .io import write


def fixture(directory):
    directory=Path(directory); directory.mkdir(parents=True,exist_ok=False)
    c=defaults(); c.update(kind='synthetic',customers=12,distribution='synthetic',train_states=1,
        counts=dict(prior_train=2,policy_train=2,development=2,historical_test=1),train_seeds=[31],
        model=dict(width=16,heads=4,layers=1),threads_per_worker=1,workers=2,checkpoint_interval=1,
        prior=dict(seed=101,steps=2,batch_size=2,lr=1e-4,grad_clip=1.),
        policy=dict(steps=2,batch_size=2,rollouts=2,lr=1e-4,grad_clip=1.),geometry=dict(steps=2,lr=.01),
        evaluation=dict(seeds=[51],single_repeats=1,iterations=2,seconds=.02,order_seed=20261023),
        diagnostic=dict(instances=1,positions=[0,1,2],candidates=2,policy_actions=2,sensitivity_states=1,prefix_length=3),
        bootstrap_samples=30,global_max_repair_calls=10000,global_max_wall_seconds=600)
    data=[]; states=[]; refs={}
    for j in range(7):
        rng=random.Random(500+j)
        i=dict(id='synthetic_'+str(j),split='train' if j<4 else 'validation' if j<6 else 'test',family_id='family_'+str(j),
            distribution='synthetic',synthetic=True,coordinates=[[rng.random(),rng.random()] for _ in range(13)],demands=[0]+[1]*12,capacity=4,distance='euclidean_float64')
        routes=initial_solution(i); data.append(i); states.append(dict(instance_id=i['id'],instance_hash=instance_hash(i),routes=routes))
        if j<4:
            refs[i['id']]=[dict(instance_id=i['id'],instance_hash=instance_hash(i),routes=routes,cost=validate_routes(i,routes),status='success',source='synthetic fixture, not solver evidence',seed=101,solver=dict(name='fixture',version='1',binary_sha256='0'*64),budget=dict(kind='iterations',limit=1))]
    for name,value in (('config.json',c),('data.json',data),('states.json',states),('references.json',refs)): write(directory/name,value)
    return c,data,states,refs
