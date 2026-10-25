import itertools
import statistics
from pathlib import Path
from .contracts import require
from .io import read,write,file_hash
from .budget import snapshot
from .statistics import paired,conclusions,equal_proposal_evidence,throughput


def analyze(run,out,allow_existing=False):
    run,out=Path(run),Path(out)
    if not allow_existing:
        require(not out.exists(),'analysis output must be a new directory'); out.mkdir(parents=True)
    c=read(run/'config.json'); state=read(run/'status.json'); identity=read(run/'identity.json')
    complete_file=run/'evaluation_complete.json'; raw=[]; complete=complete_file.exists() and state['status']=='COMPLETE'
    if complete_file.exists():
        record=read(complete_file); require(record['identity']==identity,'evaluation identity')
        for name,h in record['files'].items(): require(file_hash(run/'evaluation'/name)==h,'evaluation artifact changed')
    for path in sorted((run/'evaluation').glob('*.json')):
        row=read(path); require(row['identity']==identity,'row identity'); raw.append(row)
    if complete:
        expected=read(run/'evaluation_order.json'); require(len(raw)==len(expected),'evaluation incomplete')
        require({tuple(sorted(r['task'].items())) for r in raw}=={tuple(sorted(t.items())) for t in expected},'evaluation task mismatch')
    comparisons={}; aggregates=[]
    for group in c['groups']+['random']:
        for mode in ('single','iterations','time'):
            subset=[r for r in raw if r['group']==group and r['mode']==mode]
            if subset:
                aggregates.append(dict(group=group,mode=mode,rows=len(subset),mean_cost=statistics.mean(r['cost'] for r in subset),mean_relative_gain=statistics.mean(r['relative_gain'] for r in subset),mean_proposals=statistics.mean(r['completed'] for r in subset),total_seconds=sum(r['elapsed'] for r in subset),late=sum(r['late_discarded'] for r in subset)))
    if complete:
        for mode in ('single','iterations','time'):
            for a,b in itertools.combinations(c['groups'],2):
                left,right=('D',a) if b=='D' else (a,b)
                comparisons[left+'_vs_'+right+'_'+mode]=paired(raw,left,right,mode,c)
    decisions=conclusions(comparisons,c) if complete else dict(solver_gain='INCOMPLETE',relation_increment='INCOMPLETE',reference_attribution='INCOMPLETE')
    auxiliary=equal_proposal_evidence(comparisons,c) if complete else {}
    speed=throughput(raw,c['groups']+['random'])
    if c['kind']=='synthetic':
        decisions={k:'SYNTHETIC_NO_RESEARCH_CONCLUSION' for k in decisions}; auxiliary={k:'SYNTHETIC_NO_RESEARCH_CONCLUSION' for k in auxiliary}
    prior=read(run/'prior_diagnostic.json') if (run/'prior_diagnostic.json').exists() else None
    sensitivity=read(run/'sensitivity.json') if (run/'sensitivity.json').exists() else None
    diagnostic=[read(p) for p in sorted((run/'paid_diagnostic').glob('*.json'))]
    budget=snapshot(run/'budget.sqlite')
    summary=dict(status='COMPLETE' if complete else 'INCOMPLETE',kind=c['kind'],decisions=decisions,equal_proposal_evidence=auxiliary,timed_throughput=speed,
        comparisons=comparisons,aggregates=aggregates,budget=budget,prior=prior,
        sensitivity=sensitivity,paid_candidate_diagnostic=diagnostic,identity=identity)
    write(out/'summary.json',summary)
    lines=['# 冻结参考结构迁移：开发实验报告','',f"运行状态：**{summary['status']}**；类型：**{c['kind']}**。",'',
        '## 研究问题与分组','独立预测参考结构，再通过当前状态相关残差特征接入动作打分，是否改善等时间搜索？',
        'A：纯 RL 基线；R：同容量残差但清零 Q 数值；D：冻结预测 Q；S：固定置换预测 Q；G：预设一维几何 logistic。Repair、十点有序动作、未截断奖励与严格改善接受固定。',
        '', '## 数据、身份与预算',f"角色数量：`{c['counts']}`。原 split 不修改。historical_test 默认不评价，confirm_test 尚未提供，当前结果仅为开发证据。",
        f"身份：`{identity}`",f"累计预算：`{budget}`",'实例任意重编号等价不能自动识别，需要可靠 family_id。训练 Q 缓存单独登记；在线每次独立评价重新生成 Q。',
        '', '## 自动开发判断']
    lines += [f'- {k}：**{v}**' for k,v in decisions.items()]
    lines += ['', '三维判断分别报告，未校正多重比较。D 优于 A 但不优于 R，不能归因于关系输入。DEVELOPMENT/GO 不等于正式研究成功。',
        '', '## 必读辅助证据：等提案数与等时间吞吐',
        '自动判断基于等时间（time）模式，包含 Q/特征/编码/采样的实现开销。下列等提案数（iterations，固定次数）比较使用相同规则，仅作描述，不覆盖上方判断；两者方向不一致时，应先检查吞吐差异，而不是直接解释为方法优劣。','']
    lines += [f'- {k}：**{v}**' for k,v in auxiliary.items()] or ['- 未完成，无等提案数比较。']
    lines += ['', '|组|time 模式平均完成提案|相对 A|','|---|---:|---:|']
    lines += [f"|{g}|{v['mean_completed']:.2f}|{'' if v['ratio_to_A'] is None else format(v['ratio_to_A'],'.3f')}|" for g,v in speed.items()]
    lines += ['', '## 先验预测质量',f"状态：{prior['status'] if prior else 'INCOMPLETE'}。诊断仅使用 policy_train 参考，不代表完整策略独立测试；不选择 checkpoint。"]
    if prior:
        lines += ['','| 对照与指标（正值有利 prior） | 配对均值 | 实例 CI |','|---|---:|---|']
        lines += [f"|{k}|{v['mean']}|{v['ci95']}|" for k,v in prior['paired'].items()]
    lines += ['', '## 随机、单步、固定迭代和等时间结果','随机组无训练 seed，不复制为五个独立训练重复。改善相对初始解，不是最优 gap。',
        '', '|组|模式|行数|平均成本|相对改善|平均完成提案|迟到|','|---|---|---:|---:|---:|---:|---:|']
    lines += [f"|{x['group']}|{x['mode']}|{x['rows']}|{x['mean_cost']:.6f}|{x['mean_relative_gain']:.6%}|{x['mean_proposals']:.2f}|{x['late']}|" for x in aggregates]
    lines += ['', '## 所有学习组配对比较','先在实例×训练 seed 内平均该格全部评价观测（eval seed × 状态 × 单步重复；time/iterations 模式每格观测数 = eval seed 数 × 状态数），再计算相对成本优势。|差|≤1e-12 记为平。',
        '', '|比较|均值|中位数|实例 CI95|探索性双维 CI95|胜/平/负|每 seed 均值|','|---|---:|---:|---|---|---|---|']
    lines += [f"|{k}|{v['mean']:.6%}|{v['median']:.6%}|{v['instance_ci95']}|{v['exploratory_instance_seed_ci95']}|{v['win']}/{v['tie']}/{v['loss']}|{v['by_seed']}|" for k,v in comparisons.items()]
    lines += ['', '## 有限候选最优值 / paid candidate diagnostic',f'已完成实例数：{len(diagnostic)}。随机候选池、早中后期状态、均值、最好值和改善比例见 summary.json 与 paid_diagnostic。',
        'best-of-64 支付 64 次 Repair，不能当一次动作的公平分数，也不是全局 oracle 上界。诊断不会自动改变超参数。',
        '', '## 残差与敏感性',f"固定前缀诊断行数：{len(sensitivity['rows']) if sensitivity else 0}。仅在共同合法动作上计算 JS（自然对数）和 L1，记录首选动作变化及中心化残差尺度。",
        '训练梯度、参数更新、base/delta 尺度见各策略 metrics.jsonl。分布变化不代表收益，收益须看正式配对结果；输入替换可能分布外，不构成因果证明。',
        '', '## 时间、完整性和局限',
        '模型加载见 startup.json；训练预计算见各 run/cache.json；训练更新时间见 metrics.jsonl；在线 Q/特征/编码/采样/Repair/校验计入。无显式预热，时间评价串行，任务顺序已保存。',
        '实例 CI 条件于给定训练 seed；双维重采样仅五个 seed 时不稳定。五个策略 seed 不保证统计功效，不覆盖单一 prior seed 随机性；开发选择偏差不由普通 CI 消除。',
        '区间跨零不是等效，CI 半宽不是完整 MDE。100/100 划分的负结果不能外推更大训练数据。未建立收益不等于关系信息无价值；有限候选诊断不能否定整个方向。',
        '不宣称严格 NDS 复现，不证明最优 gap、整个动作空间局部最优或论文方法成立。合成运行仅验证工程。',
        '', '## 原始文件索引','- identity.json、data_manifest.json、config.json：输入身份及角色。','- budget.sqlite：每次 Repair 持久扣费和累计 session 墙钟。','- prior_diagnostic.json、geometry.json：先验与几何。','- A_*/R_*/D_*/S_*/G_*/metrics.jsonl、latest.pt、final.pt：训练。','- evaluation_order.json、evaluation/*.json：逐任务原始行和 anytime trace。','- paid_diagnostic/*.json、sensitivity.json：诊断。','- status.json、summary.json、REPORT.md：状态和分析。']
    (out/'REPORT.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return summary
