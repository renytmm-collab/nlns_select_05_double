import math
from .contracts import require


def defaults():
    return dict(schema=1, kind='experiment', budgets_confirmed=True,
        groups=['A','R','D','S','G'], train_seeds=[31,32,33,34,35], split_seed=20261023,
        shuffle_seed=73, customers=100, destroy_size=10, distribution='XML100-mixed-sample-v1',
        counts=dict(prior_train=100, policy_train=100, development=50, historical_test=100),
        train_states=4, model=dict(width=64,heads=4,layers=2),
        prior=dict(seed=101,steps=3000,batch_size=8,lr=1e-4,grad_clip=1.0),
        geometry=dict(steps=1000,lr=0.01),
        policy=dict(steps=3000,batch_size=8,rollouts=8,lr=1e-4,grad_clip=1.0),
        device='cpu', threads_per_worker=4, workers=2, checkpoint_interval=100, milestones=[],
        checkpoint_rule='last_step', normalization='all_nodes_center_rms_radius_clamp_1e-12',
        global_max_repair_calls=10000000,global_max_wall_seconds=259200,
        evaluation=dict(seeds=[51,52,53],single_repeats=8,iterations=100,seconds=2.0,order_seed=20261023),
        diagnostic=dict(instances=30,positions=[0,50,100],candidates=64,policy_actions=8,sensitivity_states=4,prefix_length=3),
        bootstrap_samples=5000,bootstrap_seed=20261023,min_mean_relative_cost_advantage=0.005,
        stop_after_prior=False)


def validate(c):
    require(set(c)==set(defaults()), 'unknown/missing configuration keys')
    require(c['groups']==['A','R','D','S','G'], 'five prespecified groups required')
    require(c['destroy_size']==10 and c['customers']>=10, 'fixed ten-point action')
    require(c['checkpoint_rule']=='last_step', 'last-step required')
    require(c['normalization']==defaults()['normalization'], 'normalization contract')
    require(c['kind'] in ('synthetic','experiment'), 'kind')
    require(c['device'] in ('cpu','cuda'), 'device')
    for section in ('model','counts','evaluation','diagnostic'):
        require(set(c[section])==set(defaults()[section]),section+' keys')
    for key in ('workers','threads_per_worker','checkpoint_interval','global_max_repair_calls','global_max_wall_seconds','bootstrap_samples'):
        require(type(c[key]) is int and c[key]>0, key)
    for section in ('prior','policy','geometry'):
        require(set(c[section])==set(defaults()[section]), section+' keys')
        require(all(type(v) in (int,float) and math.isfinite(v) and v>0 for v in c[section].values()), section)
        require(type(c[section]['steps']) is int,section+' integer steps')
    for section in ('prior','policy'):
        require(type(c[section]['batch_size']) is int,'integer batch')
    require(type(c['policy']['rollouts']) is int,'integer rollouts')
    require(all(type(v) is int and v>0 for v in c['counts'].values()),'role counts')
    require(all(type(v) is int and v>0 for v in c['model'].values()) and c['model']['width']%c['model']['heads']==0,'model shape')
    e=c['evaluation']; d=c['diagnostic']
    require(all(type(e[k]) is int and e[k]>0 for k in ('single_repeats','iterations')) and type(e['seconds']) in (int,float) and math.isfinite(e['seconds']) and e['seconds']>0,'evaluation budgets')
    require(all(type(d[k]) is int and d[k]>0 for k in ('instances','candidates','policy_actions','sensitivity_states')),'diagnostic budgets')
    require(d['positions']==sorted(set(d['positions'])) and d['positions'][0]==0 and all(type(x) is int and x>=0 for x in d['positions']),'diagnostic positions')
    require(type(d['prefix_length']) is int and 0<=d['prefix_length']<10,'forced prefix length')
    require(type(c['train_states']) is int and c['train_states']>0,'state count')
    require(c['policy']['rollouts']>=2, 'LOO needs K>=2')
    require(c['policy']['batch_size']<=c['counts']['policy_train'] and c['prior']['batch_size']<=c['counts']['prior_train'], 'batch size')
    require(len(set(c['train_seeds']))==len(c['train_seeds'])>0, 'training seeds')
    require(len(set(c['evaluation']['seeds']))==len(c['evaluation']['seeds'])>0, 'evaluation seeds')
    require(c['diagnostic']['instances']<=c['counts']['policy_train'], 'diagnostic count')
    require(type(c['stop_after_prior']) is bool and type(c['budgets_confirmed']) is bool, 'boolean flags')
    for key in ('split_seed','shuffle_seed','bootstrap_seed'):
        require(type(c[key]) is int, key+' integer seed')
    require(type(e['order_seed']) is int, 'evaluation order_seed integer')
    require(all(type(s) is int for s in c['train_seeds']) and all(type(s) is int for s in e['seeds']), 'integer seeds')
    t=c['min_mean_relative_cost_advantage']
    require(type(t) in (int,float) and math.isfinite(t) and 0<t<1, 'min_mean_relative_cost_advantage finite in (0,1)')
    m=c['milestones']
    require(type(m) is list and all(type(x) is int and 0<x<=c['policy']['steps'] for x in m) and len(set(m))==len(m), 'milestones: unique integer steps in 1..policy.steps')
    if c['kind']=='synthetic':
        require(c['customers']<=12 and sum(c['counts'].values())<=12 and c['prior']['steps']<=4 and c['policy']['steps']<=4 and c['policy']['rollouts']<=4 and c['geometry']['steps']<=4, 'synthetic cap')
        require(len(c['train_seeds'])<=5 and e['iterations']<=10 and e['seconds']<=1 and e['single_repeats']<=4 and max(d['positions'])<=10 and d['candidates']<=64 and d['policy_actions']<=8 and c['global_max_repair_calls']<=10000,'synthetic evaluation cap')


def budget_estimate(c):
    runs=len(c['groups'])*len(c['train_seeds'])
    train=runs*c['policy']['steps']*c['policy']['batch_size']*c['policy']['rollouts']
    e,d=c['evaluation'],c['diagnostic']
    fixed=(runs+1)*c['counts']['development']*len(e['seeds'])*(e['single_repeats']+e['iterations'])
    diag=d['instances']*(max(d['positions'])+len(d['positions'])*(d['candidates']+2*d['policy_actions']))
    return dict(policy_training=train,fixed_evaluation=fixed,paid_diagnostic=diag,
        remaining_for_timed_search=c['global_max_repair_calls']-train-fixed-diag,
        timed_searches=(runs+1)*c['counts']['development']*len(e['seeds']),
        warning='Timed proposal counts are unknown; shared hard cap stops incomplete runs.',
        estimated_disk_bytes=2_000_000_000)
