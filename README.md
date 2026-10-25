# 冻结参考结构与显式关系残差

独立可上传的 CVRP 研究项目。A/R/D/S/G 五组、5 个策略 seed，独立 prior，只用预测 Q，固定有序十点删除与原贪心 Repair。不是严格 NDS 复现。历史数据与旧工作区均不修改。

**当前验收边界：本机 Python 3.11.1，无 PyTorch/NumPy。标准库合成测试已执行；神经训练、完整合成流水线、Linux 和 PyTorch 2.5.1 CPU 尚未实际验证。正式运行前必须在目标环境执行严格验收。不能把源码完整或语法通过当作神经验证通过。**

## 最小上传清单

上传 `relation_transfer/`、`configs/`、`scripts/`、`tests/`、`docs/`、`README.md`、`requirements.txt`。不上传 `.git`、`__pycache__`、`test_artifacts`、旧数据、旧 checkpoint 或环境目录。三个数据 JSON 单独放入服务器 DATA_DIR，无需复制旧项目。源码不依赖旧工作区导入或软链接。

## 依赖（由用户执行）

已有 PyTorch 环境，先检查，不自动升级共享环境：

```bash
/absolute/path/to/python -B -c 'import sys,torch,numpy; print(sys.version,torch.__version__,numpy.__version__)'
# 若仅缺 numpy，确认使用该环境后再执行：
/absolute/path/to/python -m pip install 'numpy>=1.24,<3'
```

新建 CPU 环境（Linux，Python 3.11）：

```bash
python3.11 -m venv /absolute/path/to/relation-env
/absolute/path/to/relation-env/bin/python -m pip install 'numpy>=1.24,<3'
/absolute/path/to/relation-env/bin/python -m pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
```

默认 CPU，不需要 C++/HGS/NDS、联网数据或预训练权重。可将配置 device 改为 cuda，需已有兼容 PyTorch，设置 `CUBLAS_WORKSPACE_CONFIG=:4096:8`，仍须独立验收；不可修改进行中 run 的配置。

## 严格合成验收

在项目根目录（输出目录必须不存在）：

```bash
/absolute/path/to/python -B -m relation_transfer.pipeline verify --out-dir test_artifacts/server_acceptance --require-neural
```

输出 tests.txt、verification.json；依赖齐全才会实际运行完整合成流水线，输出 pipeline/REPORT.md、summary.json 和 status.json。依赖缺失明确 skip，严格命令退出 2。合成上限 N≤12、prior/policy≤4 步、K≤4，不接触真实数据。

## 只读预检

```bash
/absolute/path/to/python -B -m relation_transfer.pipeline preflight --config configs/xml100_default.json --data-dir /absolute/path/to/xml100_v3
```

预检不创建目录或缓存。检验旧 data.json/states.json/references.json、exact counts、family、内容 hash、状态/参考合法性；显示预算和保守磁盘预估（约 2 GB，非硬磁盘上限）。使用 `-B` 避免 Python 字节码缓存。

默认策略 Repair 4,800,000；固定评价 421,200；选择空间诊断（含生成状态与 A/D 比较）10,200，其中随机候选最多 5,760。等时间搜索 3,900 次，剩余 Repair 4,768,600，平均约 1,223 次/搜索，实际取决于吞吐。预算 10,000,000 不自动扩张；不足即 BUDGET_EXHAUSTED/INCOMPLETE，不缩短评价冒充完整。

## 吞吐与墙钟预估（只读，正式运行前执行）

```bash
/absolute/path/to/python -B -m relation_transfer.benchmark --config configs/xml100_default.json --data-dir /absolute/path/to/xml100_v3
```

使用未训练模型，不写文件、不扣预算。输出 A/R/D 每次策略更新秒数、2 秒等时间搜索完成提案数，以及策略训练墙钟相对 `global_max_wall_seconds` 的比例（S/G 按 D 计，未含 prior、评价、诊断及每次 Repair 的 SQLite fsync）。若训练比例已接近 1，正式运行很可能 BUDGET_EXHAUSTED，应先调整配置再新建 run。各组 time 模式完成提案数差异也会在最终报告“必读辅助证据”一节列出。

## 一条后台启动命令

先替换四个绝对路径。外层 RUN_DIR 和其 pipeline 子目录必须尚不存在：

```bash
nohup bash /absolute/path/to/project/scripts/run_server.sh /absolute/path/to/project /absolute/path/to/xml100_v3 /absolute/path/to/new_run /absolute/path/to/python </dev/null >/dev/null 2>&1 &
```

脚本中的正式入口带 `--confirm-real-run`，默认配置 `kind=experiment`、`budgets_confirmed=true`；执行此命令即确认列出的预算。日志写入 new_run/run_时间_PID.log，退出码写入 new_run/exit_code.txt。失败 run 不覆盖，重新开始另取目录。

也可直接前台一条命令：

```bash
python -B -m relation_transfer.pipeline run --config configs/xml100_default.json --data-dir /absolute/path/to/xml100_v3 --run-dir /absolute/path/to/new_pipeline --confirm-real-run
```

## 状态与恢复

```bash
bash /absolute/path/to/project/scripts/status.sh /absolute/path/to/project /absolute/path/to/new_run /absolute/path/to/python
```

状态包括阶段、每任务当前 step、累计 Repair、活跃墙钟和最近错误；不读取大日志、不改 run。

```bash
nohup bash /absolute/path/to/project/scripts/run_server.sh /absolute/path/to/project /absolute/path/to/xml100_v3 /absolute/path/to/new_run /absolute/path/to/python resume </dev/null >/dev/null 2>&1 &
```

恢复需要代码/配置/输入/关键环境版本身份完全一致，已提交 checkpoint hash 正确。环境（Python/PyTorch/NumPy 版本或 platform 字符串，含内核升级）一旦变化，resume 会被拒绝；此时唯一合法路径是对已有 run 执行 analyze 得到 INCOMPLETE 报告，或新建 run 重跑，不能修改 identity.json 绕过。72 小时运行期间请冻结环境。预算账本独立于模型；从 latest 重算的 Repair 再次付费，并留 replay_audit。每 100 步及最后一步写 latest，最后另存 final（final 未写完即中断时，resume 从最后一步 latest 直接写 final，不重算）；默认不保留全部步 checkpoint。若失败路径上的 analyze 本身出错，错误写入 status.json 的 analysis_error 与 analysis_error.json，旧 summary.json 标记 stale。

**SIGKILL、断电或 coordinator/worker 状态不明时 fail closed**：不自动清锁，不猜停机时间，不返还预算。只支持协调器能够收束 worker、结清 session 的安全恢复；不宣称任意故障严格恢复。暂停间隔不计活跃墙钟；各 worker 时间不相加。SQLite FULL 同步每次 Repair 先扣费，额外工程开销会进入时间，所有组一致。

`stop_after_prior=true` 会在首次 run 有意停止；之后显式执行同配置 resume 即继续剩余预算，无需改配置身份。

## 独立诊断和部分分析

可先单独执行诊断：新目录会自动完成 prior、geometry、先验诊断及 A/D 第一个 seed 的训练，再生成付费候选诊断；已有兼容目录会复用已提交结果。所有费用仍扣同一个总预算，之后 resume 补齐剩余流水线。

```bash
python -B -m relation_transfer.pipeline diagnostic --config configs/xml100_default.json --data-dir /absolute/path/to/xml100_v3 --run-dir /absolute/path/to/new_run/pipeline --confirm-real-run
python -B -m relation_transfer.pipeline analyze --run-dir /absolute/path/to/new_run/pipeline --out-dir /absolute/path/to/new_analysis
```

analyze 不改原 run，输出必须为新目录；部分结果醒目标 INCOMPLETE。报告最终位置为 `RUN_DIR/pipeline/REPORT.md` 与 `summary.json`（直接 CLI 则为 --run-dir 下）。

方法、来源及测试边界见 docs/DESIGN.md、docs/SOURCES.md、docs/VALIDATION.md。
