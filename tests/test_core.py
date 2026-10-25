import copy
import multiprocessing
import os
import random
import subprocess
import sys
import importlib.util
import unittest
from pathlib import Path
from relation_transfer.fixtures import fixture
from relation_transfer.data import load
from relation_transfer.io import read,write
from relation_transfer.contracts import instance_hash,validate_routes
from relation_transfer.vrp import repair,insert_sequence,distances,action_mask
from relation_transfer.budget import Budget,BudgetExhausted,snapshot
from relation_transfer.search import search,random_factory
from relation_transfer.statistics import quality,paired


def charge_many(path,n):
    b=Budget(path)
    for _ in range(n): b.charge('parallel')


class Core(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(os.environ['RELATION_TEST_ROOT'])/'core'; cls.c,cls.data,cls.states,cls.refs=fixture(cls.root)

    def test_repair_order_and_coverage(self):
        inst=self.data[0]; routes=self.states[0]['routes']; seq=list(range(1,11))
        out=repair(inst,routes,seq); validate_routes(inst,out)
        partial=[[j for j in r if j not in seq] for r in routes]
        self.assertEqual(out,insert_sequence(inst,partial,seq))
        self.assertNotEqual(out,repair(inst,routes,list(reversed(seq))))
        self.assertTrue(any(x!=round(x) for row in distances(inst) for x in row))
        with self.assertRaises(ValueError): repair(inst,routes,[1,1])
        with self.assertRaises(ValueError): repair(inst,routes,[0])

    def test_mask_and_existing_route_tie(self):
        self.assertEqual(action_mask(3,[2]),[True,False,True,False])
        i=copy.deepcopy(self.data[0]); i['coordinates']=[[0.,0.]]*13;i['capacity']=12
        self.assertEqual(insert_sequence(i,[[1],[2]],[3]),[[3,1],[2]])

    def test_data_roles_and_reference_leak(self):
        loaded=load(self.c,self.root); roles=loaded[3]
        self.assertFalse(set(roles['prior_train'])&set(roles['policy_train']))
        self.assertEqual(roles,load(self.c,self.root)[3])
        modified=copy.deepcopy(self.refs); modified[self.data[4]['id']]=self.refs[self.data[0]['id']]
        path=self.root/'references.json'; write(path,modified)
        try:
            with self.assertRaises(ValueError): load(self.c,self.root)
        finally: write(path,self.refs)

    def test_family_cross_role_rejected(self):
        data=copy.deepcopy(self.data); data[4]['family_id']=data[0]['family_id']; write(self.root/'data.json',data)
        try:
            with self.assertRaises(ValueError): load(self.c,self.root)
        finally: write(self.root/'data.json',self.data)

    def test_ranking_ties_and_no_positives(self):
        q=quality([.5,.5,.1],[1,0,0],1)
        self.assertAlmostEqual(q['average_precision'],.5); self.assertAlmostEqual(q['precision_at_n'],.5)
        self.assertIsNone(quality([.5,.5],[0,0],1)['average_precision'])
        self.assertEqual(quality([.9,.1],[1,0],1)['average_precision'],1.)

    def test_continuous_random(self):
        p=random_factory(self.data[0],12); a=p([]); b=p([])
        self.assertNotEqual(a,b); self.assertEqual(a,random_factory(self.data[0],12)([])); self.assertEqual(len(set(a)),10);self.assertNotIn(0,a)

    def test_shared_budget(self):
        path=self.root/'parallel.sqlite'; b=Budget(path); c=dict(self.c,global_max_repair_calls=20); b.create(c);b.start()
        ctx=multiprocessing.get_context('spawn'); procs=[ctx.Process(target=charge_many,args=(str(path),10)) for _ in range(2)]
        for p in procs:p.start()
        for p in procs:p.join(30);self.assertEqual(p.exitcode,0)
        self.assertEqual(snapshot(path)['repair_calls'],20)
        with self.assertRaises(BudgetExhausted):b.charge('overflow')
        b.finish();b.start();self.assertEqual(snapshot(path)['repair_calls'],20);b.finish()

    def test_time_includes_setup_and_discards_late(self):
        class B:
            def charge(self,c):pass
            def check(self):pass
        now=[0.]
        def factory():
            now[0]+=.3
            def choose(r):now[0]+=.8;return list(range(1,11))
            return choose
        r=search(self.data[0],self.states[0]['routes'],factory,B(),'test','time',seconds=1.,clock=lambda:now[0])
        self.assertEqual(r['late_discarded'],1);self.assertEqual(r['cost'],r['initial_cost']);self.assertAlmostEqual(r['overshoot'],.1)

    def test_pair_fail_closed(self):
        rows=[]
        for group,cost in [('D',9.),('A',10.)]:
            rows.append(dict(group=group,cost=cost,mode='time',initial_cost=12.,instance_id='i',instance_hash='h',state_hash='s',train_seed=31,eval_seed=51,repeat=0))
        self.assertAlmostEqual(paired(rows,'D','A','time',self.c)['mean'],.1)
        for bad in (rows+rows[:1],rows[:1],[rows[0],dict(rows[1],initial_cost=13.)]):
            with self.assertRaises(ValueError):paired(bad,'D','A','time',self.c)

    def test_synthetic_cannot_bypass_size_gate(self):
        bad=dict(self.c,customers=100)
        with self.assertRaises(ValueError):load(bad,self.root)

    def test_config_rejects_mistyped_rules(self):
        from relation_transfer.config import validate
        validate(self.c)
        e=lambda **kw:dict(self.c['evaluation'],**kw)
        for bad in (dict(stop_after_prior='yes'),dict(stop_after_prior=1),dict(milestones='1'),dict(milestones=[0]),dict(milestones=[1,1]),
                dict(milestones=[self.c['policy']['steps']+1]),dict(min_mean_relative_cost_advantage=float('nan')),dict(min_mean_relative_cost_advantage=-.5),
                dict(min_mean_relative_cost_advantage=0),dict(bootstrap_seed='1'),dict(bootstrap_seed=True),dict(evaluation=e(order_seed='x')),dict(train_seeds=['31'])):
            with self.assertRaises(ValueError,msg=str(bad)):validate(dict(self.c,**bad))

    def test_tie_tolerance_evidence_and_throughput(self):
        from relation_transfer.statistics import equal_proposal_evidence,throughput,conclusions
        base=dict(initial_cost=12.,instance_id='i',instance_hash='h',state_hash='s',train_seed=31,eval_seed=51,repeat=0)
        rows=[dict(base,group='A',cost=10.,mode='time'),dict(base,group='D',cost=10.*(1+1e-15),mode='time')]
        x=paired(rows,'D','A','time',self.c);self.assertEqual((x['win'],x['tie'],x['loss']),(0,1,0))
        good=dict(mean=.01,instance_ci95=[.001,.02],by_seed={str(s):.01 for s in range(5)})
        comps={'D_vs_'+g+'_iterations':good for g in 'ARSG'}
        self.assertEqual(set(equal_proposal_evidence(comps,self.c).values()),{'SAME_RULE_POSITIVE'})
        with self.assertRaises(ValueError):conclusions(comps,self.c)
        speed=throughput([dict(group='A',mode='time',completed=40),dict(group='D',mode='time',completed=10),dict(group='D',mode='iterations',completed=100)],['A','D'])
        self.assertEqual(speed['D'],dict(mean_completed=10,ratio_to_A=.25))

    def test_preflight_read_only_and_gate(self):
        before={str(p):p.stat().st_mtime_ns for p in self.root.rglob('*') if p.is_file()}
        command=[sys.executable,'-B','-m','relation_transfer.pipeline']
        args=['--config',str(self.root/'config.json'),'--data-dir',str(self.root)]
        p=subprocess.run(command+['preflight']+args,capture_output=True,text=True)
        self.assertEqual(p.returncode,0,p.stderr)
        after={str(p):p.stat().st_mtime_ns for p in self.root.rglob('*') if p.is_file()}
        self.assertEqual(before,after)
        target=self.root/'unauthorized'
        p=subprocess.run(command+['run']+args+['--run-dir',str(target)],capture_output=True,text=True)
        self.assertEqual(p.returncode,2);self.assertFalse(target.exists())

    def test_unclean_session_refused(self):
        path=self.root/'unclean.sqlite';b=Budget(path);b.create(self.c);b.start()
        with self.assertRaises(ValueError):Budget(path).start()
        b.finish()

    def test_report_and_machine_summary(self):
        from relation_transfer.report import analyze
        directory=self.root/'report_fixture';directory.mkdir()
        write(directory/'config.json',self.c);write(directory/'identity.json',{'synthetic_fixture':True})
        write(directory/'status.json',{'status':'INCOMPLETE'})
        b=Budget(directory/'budget.sqlite');b.create(self.c)
        out=self.root/'analysis';summary=analyze(directory,out)
        self.assertEqual(summary['status'],'INCOMPLETE');self.assertTrue((out/'REPORT.md').exists());self.assertTrue((out/'summary.json').exists())

    def test_bad_reference_cost_and_state_hash(self):
        refs=copy.deepcopy(self.refs);refs[self.data[0]['id']][0]['cost']+=1
        write(self.root/'references.json',refs)
        try:
            with self.assertRaises(ValueError):load(self.c,self.root)
        finally:write(self.root/'references.json',self.refs)
        states=copy.deepcopy(self.states);states[0]['instance_hash']='wrong';write(self.root/'states.json',states)
        try:
            with self.assertRaises(ValueError):load(self.c,self.root)
        finally:write(self.root/'states.json',self.states)

    @unittest.skipIf(importlib.util.find_spec('torch') is not None and importlib.util.find_spec('numpy') is not None,'missing-dependency path requires missing dependency')
    def test_missing_dependency_exit_and_status(self):
        target=self.root/'missing_dependency'
        p=subprocess.run([sys.executable,'-B','-m','relation_transfer.pipeline','run','--config',str(self.root/'config.json'),
            '--data-dir',str(self.root),'--run-dir',str(target),'--synthetic'],capture_output=True,text=True)
        self.assertEqual(p.returncode,2)
        state=read(target/'status.json');self.assertEqual(state['status'],'INCOMPLETE');self.assertIn('missing dependency',state['error'])
        self.assertEqual(snapshot(target/'budget.sqlite')['repair_calls'],0)
