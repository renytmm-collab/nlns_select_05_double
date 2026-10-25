import importlib.util
import os
import unittest
from pathlib import Path

AVAILABLE=importlib.util.find_spec('torch') is not None and importlib.util.find_spec('numpy') is not None


def reference_relation_features(q,nb,selected,k,residual_only=False):
    """Former per-customer loop implementation, kept only as the equivalence oracle."""
    import torch
    n=len(q); out=q.new_zeros((n+1,9)); chosen=set(selected)
    for j in range(1,n+1):
        near=nb[j]; other=[v for v in range(1,n+1) if v!=j and v not in near]
        out[j,2]=bool(near); out[j,4]=bool(other); out[j,7]=len(selected)/k
        out[j,8]=sum(v in chosen for v in near)/len(near) if near else 0
        if residual_only: continue
        if near:
            values=q[j-1,torch.tensor([v-1 for v in near])]
            out[j,0]=values.mean(); out[j,1]=values.min()
        if other:
            values=q[j-1,torch.tensor([v-1 for v in other])]
            out[j,3]=values.topk(min(3,len(other))).values.mean()
        valid=[v for v in selected if v!=j]
        if valid:
            values=q[j-1,torch.tensor([v-1 for v in valid])]
            out[j,5]=values.max(); out[j,6]=values.mean()
    return out


@unittest.skipUnless(AVAILABLE,'PyTorch/NumPy unavailable: neural behavior not verified')
class Neural(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        from relation_transfer.fixtures import fixture
        from relation_transfer.data import load
        from relation_transfer.training import configure
        cls.root=Path(os.environ['RELATION_TEST_ROOT'])/'neural';cls.c,_,_,_=fixture(cls.root)
        cls.ids,cls.states,cls.labels,cls.roles,cls.identity,*_=load(cls.c,cls.root);configure(cls.c)

    def test_static_symmetric_reject_routes(self):
        import torch
        from relation_transfer.neural import make_prior,static_input,get_q
        p=make_prior(self.c);inst=next(iter(self.ids.values()));logits=p(static_input(inst))
        self.assertTrue(torch.equal(logits,logits.T))
        with self.assertRaises(ValueError):p(dict(static_input(inst),routes=[]))
        q=get_q('D',inst,p,None,self.c);self.assertEqual(float(q.diagonal().sum()),0.)

    def test_base_initialization_and_zero_residual(self):
        import torch
        from relation_transfer.neural import make_policy,neighbors
        from relation_transfer.vrp import ordinary_state
        key=next(iter(self.ids));inst=self.ids[key];r=self.states[key][0]['routes'];base=make_policy(self.c,'A',31)
        for group in ('R','D','S','G'):
            model=make_policy(self.c,group,31)
            for name,p in base.state_dict().items():self.assertTrue(torch.equal(p,model.state_dict()[name]),name)
            state=ordinary_state(inst,r);q=torch.zeros(12,12);nb=neighbors(r,12)
            a=base.decode(base.encoder(state),10,torch.Generator().manual_seed(1),q=q,nb=nb)
            b=model.decode(model.encoder(state),10,torch.Generator().manual_seed(1),q=q,nb=nb)
            self.assertEqual(a[0],b[0]);self.assertTrue(all(torch.equal(x,y) for x,y in zip(a[2],b[2])))

    def test_features_and_permutation_isolation(self):
        import torch
        from relation_transfer.neural import relation_features,neighbors,neighbor_mask,permute
        q=torch.ones(12,12);q.fill_diagonal_(999); nb=neighbors([[1],[2,3],[4,5,6,7,8,9,10,11,12]],12); near=neighbor_mask(nb,12)
        f=relation_features(q,near,[2],10);self.assertTrue(bool(torch.isfinite(f).all()));self.assertEqual(float(f[1,0]),0.);self.assertEqual(float(f[3,8]),1.)
        r=relation_features(q,near,[2],10,True);self.assertEqual(float(r[:,[0,1,3,5,6]].abs().sum()),0.)
        state=torch.get_rng_state();inst=next(iter(self.ids.values()))
        self.assertTrue(torch.equal(permute(q,inst,73),permute(q,inst,73)));self.assertTrue(torch.equal(state,torch.get_rng_state()))

    def test_vectorized_features_match_reference_loop(self):
        import random,torch
        from relation_transfer.neural import relation_features,neighbors,neighbor_mask
        rng=random.Random(7);g=torch.Generator().manual_seed(7)
        for n in (10,12,30,100):
            for trial in range(6):
                customers=list(range(1,n+1));rng.shuffle(customers);routes=[];i=0
                while i<n:
                    size=rng.choice((1,1,2,3,5,9));routes.append(customers[i:i+size]);i+=size
                q=torch.rand((n,n),generator=g,dtype=torch.float64);q=(q+q.T)/2
                if trial%3==0:q=torch.zeros((n,n),dtype=torch.float64)
                if trial%3==1:q=(q*4).round()/4  # ties for topk/min/max
                q.fill_diagonal_(0);nb=neighbors(routes,n);near=neighbor_mask(nb,n)
                selected=rng.sample(customers,rng.randint(0,min(9,n-1)))
                for residual_only in (False,True):
                    expected=reference_relation_features(q,nb,selected,10,residual_only)
                    actual=relation_features(q,near,selected,10,residual_only)
                    self.assertEqual(actual.shape,expected.shape)
                    self.assertTrue(torch.allclose(actual,expected,rtol=0,atol=1e-12),(n,trial,residual_only,(actual-expected).abs().max()))

    def test_rl_gradient_frozen_prior_and_resume(self):
        import torch
        from relation_transfer.neural import make_prior
        from relation_transfer.training import Trainer,save_checkpoint
        from relation_transfer.budget import Budget,snapshot
        prior=make_prior(self.c).eval().requires_grad_(False);before={k:v.clone() for k,v in prior.state_dict().items()}
        b=Budget(self.root/'train.sqlite');b.create(self.c);b.start()
        t=Trainer(self.c,'D',31,self.ids,self.states,self.roles,prior,torch.zeros(2),b,self.identity)
        metrics=t.update();self.assertGreater(metrics['residual_gradient_norm'],0.)
        path=self.root/'checkpoint.pt';save_checkpoint(path,t.payload());t.update()
        other=Trainer(self.c,'D',31,self.ids,self.states,self.roles,prior,torch.zeros(2),b,self.identity);other.restore(path);other.update()
        for k,v in t.model.state_dict().items():self.assertTrue(torch.equal(v,other.model.state_dict()[k]),k)
        for k,v in prior.state_dict().items():self.assertTrue(torch.equal(v,before[k]),k)
        self.assertTrue(all(p.grad is None for p in prior.parameters()));self.assertEqual(snapshot(b.path)['repair_calls'],12);b.finish()

    def test_geometry_role_and_candidate_direction(self):
        from relation_transfer.training import fit_priors
        from relation_transfer.budget import Budget
        from relation_transfer.io import read
        from relation_transfer.diagnostics import candidate_summary
        b=Budget(self.root/'prior.sqlite');b.create(self.c);b.start()
        fit_priors(self.c,self.ids,self.labels,self.roles,self.root,self.identity,b);b.finish()
        self.assertEqual(read(self.root/'geometry.json')['fit_ids'],self.roles['prior_train'])
        self.assertEqual(candidate_summary([-.1,.2,.1])['best'],.2)
