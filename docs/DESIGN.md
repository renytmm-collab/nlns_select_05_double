# 研究和工程约定

静态 prior：普通坐标、需求/容量，实例内全节点坐标中心化，以全节点均方半径开根为尺度，clamp≥1e-12。静态 attention 和对称 pair MLP，无 depot/对角线标签。全部 i<j 无权 BCE，先实例内均值，再实例均值。几何 logistic 使用同尺度，b/w 零初始化、full-batch Adam 1000 步、lr=.01。常数密度按 prior_train 实例均值拟合。三者只用 prior_train 标签。单一 prior seed=101，不联合微调，不共享动态参数。

动态编码器与 pointer/GRU 继承旧独立实现。残差 MLP 输入 h/z/固定九特征，隐藏 64，最终层零初始化，独立 RNG。R 保留特征 3/5/8/9（邻居存在、非邻居存在、已选比例、邻居已选比例），将 1/2/4/6/7 清零，不加载 prior。邻居固定为本轮开始路线，已选集合逐步更新。Q 不阈值化，无对角线和 depot。S 由 instance_hash+73 的私有 RNG 置换 Q，不能消耗策略随机流。

九特征为整矩阵向量化实现：每次 decode 开始由路线邻居构造一次 (N,N) 布尔邻居矩阵，每步只更新已选向量，用 masked mean/amin/topk/amax 一次算出所有客户的特征。语义与旧逐客户 Python 循环完全一致（tests/test_neural.py 保留旧循环作为等价判据，含平分、零 Q、单点路线和 R 组）。原因：逐客户循环的 Python/小张量开销会计入等时间预算，使 D/S/G/R 的提案数系统性少于 A，把实现开销误判为方法差异，并可能使训练超过墙钟上限。

每个 batch 不同实例各随机选一个状态，K 完整序列全部参与 LOO，无 top-1、奖励不截断。固定 full batch；尾部不足一个 batch 时重新 shuffle（用于每个配置更新恰好 batch×K Repair，记录为工程选择）。当前默认 100/8 使用这个明确规则，不把 200 实例旧模型充当新基线。动作有序，不只是删除集合。

Repair 保留 float64 Python 欧氏距离；同时删除，按动作序列重插；最小 (增量,路线号,位置)；有可行既有路线就不开新路线。每完整序列只 Repair 一次。单步保留实际负收益，多步只严格改善接受。固定时间生成 Q、校验和候选全计时，最后迟到候选丢弃，记录 attempted/completed/late/overshoot。无显式预热；模型加载单列。训练两 worker 结束后评价串行，预先洗牌并保存任务顺序。

角色：排序 ID，私有 split RNG=20261023，按 family 分组做精确子集和划分；单实例 family 时等价于打乱列表取前 100。原 train 分 prior_train/policy_train，validation 为 development，test 为 historical_test（不评价）。拒绝内容重复、family 跨角色或不足数量。未实现未来 confirm_test 导入/确认流程，不能使用本项目开发报告声称独立确认。

开发结论分三维：D/A≥0.5%、实例 CI 下界>0、至少 4/5 seed 正为 SOLVER_GAIN_GO；D/R 与 D/S 各均值>0、CI下界>0、4/5正为 RELATION_INCREMENT_SUPPORTED；D/G 对应条件为 REFERENCE_ATTRIBUTION_SUPPORTED。所有学习组两两比较都报告。先按实例×seed平均重复成本，再算相对优势。实例 bootstrap 条件于给定 seed；双维 bootstrap 探索性，不覆盖 prior seed。阈值预设，未经多重校正，不保证功效。

自动判断仍只用等时间（time）模式。同一规则在固定迭代（iterations，等提案数）模式上的结果作为必读辅助证据写入 summary.json 的 equal_proposal_evidence，time 模式各组平均完成提案数及相对 A 的比例写入 timed_throughput；二者只描述，不覆盖判断。time 模式每个实例×seed 格含 eval seed 数×状态数个观测，不另设重复。相对优势 |差|≤1e-12 计为平局。

AP 按相同分数合并阈值组，召回增量乘该组累计 precision；无正例 AP=null。precision@N 在截断平分组用期望命中数，N 取客户数（默认 100，即在 4950 对中取前 100）。诊断不选 checkpoint、不自动调参。先验支持标记仅为对两简单对照 BCE 实例 CI 的描述性规则，不是通用信息价值门槛。

恢复：SQLite 独占事务逐 Repair 扣费，多进程共享，checkpoint 不含可回滚账本。模型/优化器/独立 RNG/数据游标均保存，临时文件原子替换并校验 SHA256；原子替换中断导致不一致就拒绝恢复。安全退出记录 session 活跃墙钟，离线停机不计入；不可判断的崩溃直接拒绝自动恢复。prior/geometry 更新单列进度和日志，无 Repair 费用；重算不返还实际账本。时间上限在安全边界检查，单次调用可能超限，最终 budget 显示实际超出。

默认报告不使用最优 gap，不把 best-of-64 当一次动作成绩，不将 null CI 解释为等效或方向无效。输入替换仅分布敏感性，不能独立作因果证明。
