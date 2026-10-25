# 当前验证记录（2026-10-24）

## 2026-10-24 修订

依据代码审查修改：`relation_features` 改为整矩阵向量化（每次 decode 构造一次邻居布尔矩阵）；报告与 summary 新增等提案数（iterations）同规则辅助证据和 time 模式吞吐比；配对平局改为 |差|≤1e-12；补齐配置校验（布尔开关、种子、阈值、milestones）；失败路径 analyze 出错不再静默；评价状态缺失报 ValueError；删除无引用代码；新增只读吞吐探针 `relation_transfer.benchmark`。

最新实际测试：test_artifacts/final_07/verification.json，共 23 项，17 通过、0 失败、0 错误、6 神经测试跳过（新增向量化特征与旧循环逐元素等价测试，同样因无 PyTorch 跳过）。**向量化特征、吞吐探针与训练/评价路径均未在 PyTorch 下实跑**；必须在目标环境执行 `verify --require-neural` 与 benchmark 后才能正式运行。代码身份已变化，旧 run 不能 resume。

## 2026-10-23 记录

环境：Windows，Python 3.11.1；当前解释器未安装 NumPy/PyTorch。没有安装依赖、运行真实训练或调用真实实例 Repair。旧项目只读。

最新实际测试：test_artifacts/final_04/verification.json，共 20 项组合测试，15 通过、0 失败、0 错误、5 神经测试跳过。严格验收真实子进程退出码为 2。之后仅调整 stop_after_prior 的显式 resume 继续条件；该神经阶段行为未实跑。

已实际执行：

- 真实 XML100 数据只读 preflight，通过 350 实例、950 状态、训练参考、100/100/50/100 角色计数及身份校验；只校验合法性和参考成本，不运行搜索或求解。
- 标准库合成测试：Repair 顺序/覆盖、mask/depot/重复、既有路线平局、float 距离、角色与参考泄漏、family 泄漏、排序 tie/无正例、随机连续 RNG、跨 spawn worker 预算及上限、Q setup 计时/迟到丢弃、配对缺失/重复/初始成本不一致拒绝、synthetic gate、preflight 不改文件、未确认执行 gate、unclean session 拒绝、部分中文报告与 summary、错误参考成本和状态身份。
- 无 PyTorch 时神经测试显式 skip；严格 --require-neural 命令返回非零。

未实际执行：

- PyTorch 模型初始化/采样等价、真实 RL 残差梯度、冻结 prior、神经 checkpoint 精确恢复、几何训练等 5 项组合神经测试。
- 完整神经合成流水线、25 run 调度与完整报告的端到端实跑。
- Linux shell 脚本实际执行、PyTorch 2.5.1 CPU 兼容实证、CUDA、多日故障恢复、正式预算吞吐。

verify 在依赖齐全的目标环境自动执行神经测试并运行完整合成流水线。服务器命令见 README。完整合成成功仍不是科学证据。不能将当前项目描述为已获神经验收或正式生产就绪。

本地具体日志保存在 test_artifacts 下，上传时排除；后续修改后的最终验收编号与结果以最新 verification.json 为准。早期日志保留，不覆盖历史。
