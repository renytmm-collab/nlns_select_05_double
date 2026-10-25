import os
import random
import time
from pathlib import Path
import torch
from .contracts import require, digest
from .io import append, write, read, file_hash
from .neural import make_prior, make_policy, pair_loss, static_input, normalized, get_q, neighbors
from .vrp import distances, route_cost, repair, ordinary_state, leave_one_out
from .budget import Budget, snapshot


def configure(c):
    torch.set_num_threads(c['threads_per_worker'])
    torch.use_deterministic_algorithms(True)
    if c['device']=='cuda':
        require(torch.cuda.is_available(),'CUDA unavailable')
        require(os.environ.get('CUBLAS_WORKSPACE_CONFIG') in (':4096:8',':16:8'),'set CUBLAS_WORKSPACE_CONFIG before CUDA')


def save_checkpoint(path,payload):
    path=Path(path); tmp=path.with_suffix('.pending')
    with tmp.open('wb') as f:
        torch.save(payload,f); f.flush(); os.fsync(f.fileno())
    os.replace(tmp,path)
    write(path.with_suffix('.sha256.json'),dict(sha256=file_hash(path)))


def load_checkpoint(path,identity):
    path=Path(path)
    require(read(path.with_suffix('.sha256.json'))['sha256']==file_hash(path),'checkpoint content hash mismatch or interrupted commit')
    p=torch.load(path,map_location='cpu',weights_only=True)
    require(p['identity']==identity,'checkpoint identity mismatch')
    return p


def norm(parameters):
    return sum(float(p.grad.detach().square().sum()) for p in parameters if p.grad is not None)**0.5


def optimizer_to(optimizer,device):
    for state in optimizer.state.values():
        for k,v in state.items():
            if isinstance(v,torch.Tensor) and k!='step': state[k]=v.to(device)


def fit_priors(c,ids,labels,roles,run,identity,budget):
    configure(c); run=Path(run)
    keys=roles['prior_train']; model=make_prior(c)
    opt=torch.optim.Adam(model.parameters(),lr=c['prior']['lr']); rng=random.Random(c['prior']['seed']+10001)
    start_step=0; path=run/'prior_latest.pt'; started=time.perf_counter()
    if path.exists():
        cp=load_checkpoint(path,identity); model.load_state_dict(cp['model']); opt.load_state_dict(cp['optimizer']); optimizer_to(opt,c['device'])
        rng.setstate(cp['data_rng']); start_step=cp['step']
    for step in range(start_step,c['prior']['steps']):
        budget.check(); opt.zero_grad(set_to_none=True)
        batch=rng.sample(keys,c['prior']['batch_size'])
        loss=torch.stack([pair_loss(model(static_input(ids[k])),labels[k]) for k in batch]).mean()
        require(bool(torch.isfinite(loss)),'nonfinite prior loss'); loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),c['prior']['grad_clip'],error_if_nonfinite=True); opt.step()
        budget.progress('prior',step+1)
        append(run/'prior_metrics.jsonl',dict(step=step+1,loss=float(loss.detach()),instances=len(batch)))
        if (step+1)%c['checkpoint_interval']==0 or step+1==c['prior']['steps']:
            save_checkpoint(path,dict(identity=identity,model=model.state_dict(),optimizer=opt.state_dict(),data_rng=rng.getstate(),step=step+1))
    save_checkpoint(run/'prior_final.pt',dict(identity=identity,model=model.state_dict(),step=c['prior']['steps']))
    model.eval().requires_grad_(False)
    model.zero_grad(set_to_none=True)
    geo=torch.zeros(2,device=c['device'],requires_grad=True); go=torch.optim.Adam([geo],lr=c['geometry']['lr']); gp=run/'geometry_latest.pt'; gs=0
    if gp.exists():
        cp=load_checkpoint(gp,identity)
        with torch.no_grad(): geo.copy_(cp['parameters'].to(c['device']))
        go.load_state_dict(cp['optimizer']); optimizer_to(go,c['device']); gs=cp['step']
    for step in range(gs,c['geometry']['steps']):
        budget.check(); go.zero_grad(set_to_none=True)
        # Backprop each weighted instance avoids keeping all pair-head graphs alive.
        value=0.
        for k in keys:
            _,d=normalized(static_input(ids[k]),c['device'])
            loss=pair_loss(geo[0]+geo[1]*d,labels[k])/len(keys)
            loss.backward(); value+=float(loss.detach())
        require(bool(torch.isfinite(geo.grad).all()),'nonfinite geometry gradient'); go.step()
        append(run/'geometry_metrics.jsonl',dict(step=step+1,loss=value,instances=len(keys)))
        budget.progress('geometry',step+1)
        if (step+1)%c['checkpoint_interval']==0 or step+1==c['geometry']['steps']:
            save_checkpoint(gp,dict(identity=identity,parameters=geo.detach(),optimizer=go.state_dict(),step=step+1))
    ix=torch.triu_indices(c['customers'],c['customers'],1)
    density=sum(sum(labels[k][i][j] for i,j in zip(ix[0].tolist(),ix[1].tolist()))/ix.shape[1] for k in keys)/len(keys)
    artifact=dict(parameters=geo.detach().cpu().tolist(),density=density,fit_role='prior_train',fit_ids=keys,steps=c['geometry']['steps'],lr=c['geometry']['lr'],normalization=c['normalization'],seconds_this_session=time.perf_counter()-started)
    write(run/'geometry.json',artifact)
    return model,geo.detach()


class Trainer:
    def __init__(self,c,group,seed,ids,states,roles,prior,geometry,budget,identity):
        self.c,self.group,self.seed=c,group,seed; self.ids,self.states=ids,states
        self.keys=roles['policy_train']; self.budget,self.identity=budget,identity
        self.model=make_policy(c,group,seed); self.optimizer=torch.optim.Adam(self.model.parameters(),lr=c['policy']['lr'])
        self.data_rng=random.Random(seed+10001); self.action_rng=torch.Generator().manual_seed(seed+20001)
        self.order=list(self.keys); self.data_rng.shuffle(self.order); self.cursor=0; self.step=0
        start=time.perf_counter()
        self.q={k:get_q(group,ids[k],prior,geometry,c) for k in self.keys}
        self.cache_seconds=time.perf_counter()-start
        self.cache_bytes=sum(t.numel()*t.element_size() for t in self.q.values())

    def update(self):
        self.budget.check(); start=time.perf_counter(); batch=[]
        # Fixed full unique-instance batches: tail is not silently shortened.
        if self.cursor+self.c['policy']['batch_size']>len(self.order):
            self.data_rng.shuffle(self.order); self.cursor=0
        keys=self.order[self.cursor:self.cursor+self.c['policy']['batch_size']]; self.cursor+=len(keys)
        for k in keys: batch.append((k,self.data_rng.choice(self.states[k])['routes']))
        self.optimizer.zero_grad(set_to_none=True); losses=[]; rewards_all=[]; stats=[]
        residual=list(self.model.residual.parameters()) if self.group!='A' else []
        before=[p.detach().clone() for p in residual]
        for key,routes in batch:
            inst=self.ids[key]; d=distances(inst); old=route_cost(routes,d)
            h=self.model.encoder(ordinary_state(inst,routes)); nb=neighbors(routes,len(inst['demands'])-1)
            logs=[]; rewards=[]
            for _ in range(self.c['policy']['rollouts']):
                action,logp,_=self.model.decode(h,10,self.action_rng,q=self.q[key],nb=nb)
                self.budget.charge('policy_'+self.group+'_'+str(self.seed))
                new=route_cost(repair(inst,routes,action,d),d)
                rewards.append((old-new)/max(old,1e-12)); logs.append(logp); stats.extend(self.model.last_stats)
            advantages=h.new_tensor(leave_one_out(rewards))
            losses.append(-(advantages.detach()*torch.stack(logs)).mean()); rewards_all+=rewards
        loss=torch.stack(losses).mean(); require(bool(torch.isfinite(loss)),'nonfinite RL loss'); loss.backward()
        residual_grad=norm(residual); main_grad=norm([p for name,p in self.model.named_parameters() if not name.startswith('residual.')])
        total=torch.nn.utils.clip_grad_norm_(self.model.parameters(),self.c['policy']['grad_clip'],error_if_nonfinite=True)
        self.optimizer.step(); self.step+=1
        update=sum(float((p.detach()-b).square().sum()) for p,b in zip(residual,before))**0.5
        return dict(step=self.step,loss=float(loss.detach()),mean_reward=sum(rewards_all)/len(rewards_all),repair_calls=len(rewards_all),
            residual_gradient_norm=residual_grad,main_gradient_norm=main_grad,residual_update_norm=update,total_gradient_norm=float(total),
            **{k:sum(s[k] for s in stats)/len(stats) for k in stats[0]},nonfinite=False,seconds=time.perf_counter()-start)

    def payload(self):
        return dict(identity=self.identity,model=self.model.state_dict(),optimizer=self.optimizer.state_dict(),step=self.step,
            data_rng=self.data_rng.getstate(),action_rng=self.action_rng.get_state(),order=self.order,cursor=self.cursor,
            python_rng=random.getstate(),torch_rng=torch.get_rng_state(),cuda_rng=torch.cuda.get_rng_state_all() if self.c['device']=='cuda' else [],
            charged_budget=snapshot(self.budget.path),cache_seconds=self.cache_seconds,cache_bytes=self.cache_bytes)

    def restore(self,path):
        cp=load_checkpoint(path,self.identity); self.model.load_state_dict(cp['model']); self.optimizer.load_state_dict(cp['optimizer']); optimizer_to(self.optimizer,self.c['device'])
        self.step,self.order,self.cursor=cp['step'],cp['order'],cp['cursor']; self.data_rng.setstate(cp['data_rng']); self.action_rng.set_state(cp['action_rng'])
        random.setstate(cp['python_rng']); torch.set_rng_state(cp['torch_rng'])
        if self.c['device']=='cuda': torch.cuda.set_rng_state_all(cp['cuda_rng'])


def policy_worker(c,group,seed,ids,states,roles,run,identity):
    configure(c); run=Path(run); task=group+'_'+str(seed); directory=run/task; directory.mkdir(exist_ok=True)
    budget=Budget(run/'budget.sqlite'); final=directory/'final.pt'
    if final.exists(): load_checkpoint(final,identity); return task
    prior=None
    if group in ('D','S'):
        prior=make_prior(c); prior.load_state_dict(load_checkpoint(run/'prior_final.pt',identity)['model']); prior.eval().requires_grad_(False)
    geometry=torch.tensor(read(run/'geometry.json')['parameters'],device=c['device'])
    trainer=Trainer(c,group,seed,ids,states,roles,prior,geometry,budget,identity)
    latest=directory/'latest.pt'
    if latest.exists(): trainer.restore(latest)
    previous=snapshot(budget.path)['progress'].get(task,0)
    if previous>trainer.step:
        append(directory/'replay_audit.jsonl',dict(previous_observed_step=previous,restored_step=trainer.step,
            note='Previously charged work remains charged; replay is extra compute. Partial updates may also have consumed calls.'))
    write(directory/'cache.json',dict(seconds=trainer.cache_seconds,bytes=trainer.cache_bytes,online_reuse=False))
    while trainer.step<c['policy']['steps']:
        metrics=trainer.update(); append(directory/'metrics.jsonl',metrics); budget.progress(task,trainer.step)
        if trainer.step%c['checkpoint_interval']==0 or trainer.step==c['policy']['steps']:
            save_checkpoint(latest,trainer.payload())
        if trainer.step in c['milestones']: save_checkpoint(directory/('step_%d.pt'%trainer.step),trainer.payload())
    save_checkpoint(final,trainer.payload())
    return task
