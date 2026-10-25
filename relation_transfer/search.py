import random
import time
from .contracts import require, validate_routes
from .vrp import distances, repair, route_cost


def random_factory(instance,seed):
    rng=random.Random(seed); customers=list(range(1,len(instance['demands'])))
    return lambda routes:rng.sample(customers,10)


def search(instance,initial,factory,budget,category,mode,iterations=100,seconds=2.,clock=time.perf_counter):
    require(mode in ('single','iterations','time'),'search mode')
    start=clock(); choose=factory()  # Q, shuffle/geometry and feature setup are inside clock.
    validate_routes(instance,initial); d=distances(instance)
    current=[r[:] for r in initial]; old=cost=route_cost(current,d)
    attempted=completed=late=0; trace=[dict(seconds=0.,iteration=0,cost=cost)]
    limit=1 if mode=='single' else iterations
    while (attempted<limit if mode!='time' else clock()-start<seconds):
        action=choose(current); require(len(action)==10,'exact deletion count')
        budget.charge(category); attempted+=1
        candidate=repair(instance,current,action,d); new=route_cost(candidate,d); elapsed=clock()-start
        if mode=='time' and elapsed>seconds: late+=1; break
        completed+=1
        if mode=='single' or new<cost:
            current,cost=candidate,new; trace.append(dict(seconds=elapsed,iteration=completed,cost=cost))
    elapsed=clock()-start; budget.check()
    return dict(cost=cost,initial_cost=old,gain=old-cost,relative_gain=(old-cost)/max(old,1e-12),routes=current,
        attempted=attempted,completed=completed,late_discarded=late,overshoot=max(0.,elapsed-seconds) if mode=='time' else 0.,elapsed=elapsed,trace=trace)
