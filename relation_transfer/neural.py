import math
import random
import torch
from torch import nn
from torch.nn import functional as F
from .base import Policy
from .contracts import require, digest, instance_hash
from .vrp import action_mask


def static_input(instance):
    return {k:instance[k] for k in ('coordinates','demands','capacity')}


def normalized(state, device='cpu'):
    require(set(state)=={'coordinates','demands','capacity'}, 'static predictor rejects routes/reference/IDs')
    xy=torch.tensor(state['coordinates'],dtype=torch.float32,device=device)
    xy=xy-xy.mean(0)
    scale=xy.square().sum(-1).mean().sqrt().clamp_min(1e-12)
    xy=xy/scale
    demand=torch.tensor(state['demands'],dtype=xy.dtype,device=device)/state['capacity']
    return torch.cat((xy,demand[:,None]),-1), torch.cdist(xy[1:],xy[1:])


class Prior(nn.Module):
    def __init__(self,width=64,heads=4,layers=2):
        super().__init__()
        self.input=nn.Linear(3,width)
        self.layers=nn.ModuleList([nn.TransformerEncoderLayer(width,heads,2*width,dropout=0,batch_first=True) for _ in range(layers)])
        self.pair=nn.Sequential(nn.Linear(2*width,width),nn.ReLU(),nn.Linear(width,1))

    def forward(self,state):
        x,_=normalized(state,self.input.weight.device)
        h=self.input(x)[None]
        for layer in self.layers: h=layer(h)
        h=h[0,1:]; a,b=h[:,None],h[None,:]
        return self.pair(torch.cat((a+b,a*b),-1)).squeeze(-1)


def pair_loss(logits,target):
    ix=torch.triu_indices(len(logits),len(logits),offset=1,device=logits.device)
    y=torch.as_tensor(target,dtype=logits.dtype,device=logits.device)
    return F.binary_cross_entropy_with_logits(logits[ix[0],ix[1]],y[ix[0],ix[1]])


def make_prior(c):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(c['prior']['seed']); model=Prior(**c['model'])
    return model.to(c['device'])


def neighbors(routes,n):
    result=[[] for _ in range(n+1)]
    for route in routes:
        for pos,j in enumerate(route):
            result[j]=([route[pos-1]] if pos else [])+([route[pos+1]] if pos+1<len(route) else [])
    return result


def neighbor_mask(nb,n,device='cpu'):
    """(n,n) bool customer-neighbor matrix from one-based route neighbors; built once per decode."""
    require(len(nb)==n+1,'neighbor list length')
    near=torch.zeros((n,n),dtype=torch.bool,device=device)
    pairs=[(j-1,v-1) for j in range(1,n+1) for v in nb[j]]
    if pairs:
        rows,cols=zip(*pairs); near[list(rows),list(cols)]=True
    return near


def relation_features(q,near,selected,k,residual_only=False):
    # Vectorized; semantics identical to the former per-customer loop (see tests/test_neural.py reference).
    # Customer matrix uses zero-based indices; route/action IDs remain one-based; row 0 (depot) stays zero.
    n=len(q); require(tuple(near.shape)==(n,n) and near.dtype==torch.bool,'neighbor mask shape')
    dt=q.dtype; zero=q.new_zeros(n)
    eye=torch.eye(n,dtype=torch.bool,device=q.device); other=~near&~eye
    chosen=torch.zeros(n,dtype=torch.bool,device=q.device)
    if selected: chosen[torch.tensor([v-1 for v in selected],device=q.device)]=True
    near_count=near.sum(1); other_count=other.sum(1)
    f=[zero]*9
    f[2]=(near_count>0).to(dt); f[4]=(other_count>0).to(dt)
    f[7]=torch.full_like(zero,len(selected)/k)
    f[8]=(near&chosen[None]).sum(1).to(dt)/near_count.clamp_min(1).to(dt)
    if not residual_only:
        f[0]=q.masked_fill(~near,0).sum(1)/near_count.clamp_min(1).to(dt)
        f[1]=torch.where(near_count>0,q.masked_fill(~near,torch.inf).amin(1),zero)
        m=min(3,n); top=q.masked_fill(~other,-torch.inf).topk(m,dim=1).values
        take=other_count.clamp(max=m)
        keep=torch.arange(m,device=q.device)[None]<take[:,None]
        f[3]=top.masked_fill(~keep,0).sum(1)/take.clamp_min(1).to(dt)
        valid=chosen[None]&~eye; valid_count=valid.sum(1)
        f[5]=torch.where(valid_count>0,q.masked_fill(~valid,-torch.inf).amax(1),zero)
        f[6]=q.masked_fill(~valid,0).sum(1)/valid_count.clamp_min(1).to(dt)
    return torch.cat((q.new_zeros((1,9)),torch.stack(f,-1)),0)


class TransferPolicy(Policy):
    def __init__(self,group,seed,**spec):
        super().__init__(**spec)
        self.group=group
        if group!='A':
            with torch.random.fork_rng(devices=[]):
                torch.default_generator.manual_seed(seed+90001)
                self.residual=nn.Sequential(nn.Linear(2*spec['width']+9,64),nn.ReLU(),nn.Linear(64,1))
                nn.init.zeros_(self.residual[-1].weight); nn.init.zeros_(self.residual[-1].bias)

    def decode(self,h,k,generator=None,greedy=False,forced=None,*,q=None,nb=None,disable=False):
        require(1<=k<len(h),'action length')
        require(forced is None or len(forced)==k,'forced length')
        hidden=torch.tanh(self.initial(h[1:].mean(0))); previous=h[0]; keys=self.key(h)
        selected=[]; logs=[]; distributions=[]; stats=[]
        use_residual=self.group!='A' and not disable
        near=neighbor_mask(nb,len(h)-1,h.device) if use_residual else None
        for step in range(k):
            hidden=self.gru(previous[None],hidden[None])[0]
            base=10*torch.tanh((keys@self.query(hidden))/math.sqrt(h.shape[-1]))
            delta=torch.zeros_like(base)
            if use_residual:
                features=relation_features(q,near,selected,10,self.group=='R')
                delta=self.residual(torch.cat((h,hidden.expand_as(h),features),-1)).squeeze(-1)
            mask=torch.tensor(action_mask(len(h)-1,selected),device=h.device)
            legal=delta[~mask]; b=base[~mask]
            stats.append(dict(centered_delta_std=float(legal.detach().std(unbiased=False)),delta_abs_mean=float(legal.detach().abs().mean()),base_centered_std=float(b.detach().std(unbiased=False))))
            logp=torch.log_softmax((base+delta).masked_fill(mask,-torch.inf),-1)
            if forced is not None:
                node=forced[step]; require(type(node) is int and 0<=node<len(h) and not bool(mask[node]),'forced action')
            elif greedy: node=int(logp.argmax())
            else:
                require(generator is not None,'explicit RNG')
                node=int(torch.multinomial(logp.detach().exp().cpu(),1,generator=generator))
            selected.append(node); logs.append(logp[node]); distributions.append(logp); previous=h[node]
        self.last_stats=stats
        return selected,torch.stack(logs).sum(),distributions


def make_policy(c,group,seed):
    with torch.random.fork_rng(devices=[]):
        torch.default_generator.manual_seed(seed); model=TransferPolicy(group,seed,**c['model'])
    return model.to(c['device'])


def permute(q,inst,seed):
    order=list(range(len(q)))
    random.Random(int(digest([instance_hash(inst),seed]),16)).shuffle(order)
    ix=torch.tensor(order,device=q.device)
    return q[ix][:,ix]


@torch.no_grad()
def get_q(group,inst,prior,geometry,c):
    n=len(inst['demands'])-1
    if group in ('A','R'): return torch.zeros((n,n),device=c['device'])
    if group=='G':
        _,dist=normalized(static_input(inst),c['device']); q=torch.sigmoid(geometry[0]+geometry[1]*dist)
    else:
        q=prior(static_input(inst)).sigmoid()
        if group=='S': q=permute(q,inst,c['shuffle_seed'])
    q.fill_diagonal_(0)
    return q.detach()
