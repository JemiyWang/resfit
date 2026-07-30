# SHORE-RL 补充材料评审

评审对象：

- `paper/main.tex`
- `paper/aaai2027-unified-supp.tex`

## 总体结论

这份材料作为“内部审计稿/工作底稿”很好，但作为正式投稿的 Supplement
目前还不合适，也还不够。

方法部分已经比较完整；仿真部分缺少支撑核心结论的原始结果与复现信息；
真实机器人部分存在尚未解决的证据链冲突。当前版本不建议直接提交。

## 分章节判断

| 部分 | 当前质量 | 判断 |
|---|---|---|
| A.1--A.5 方法与优化细节 | 较强 | 基本保留 |
| B.1--B.3 任务、特征、超参数 | 较好 | 补少量关键细节 |
| B.4--B.7 baseline、结果、provenance | 不够 | 需要大幅补充 |
| B.8 仿真局限 | 合适 | 保留并适度压缩 |
| C.1--C.4 机器人平台与 world model | 信息很多，但偏内部审计 | 重组 |
| C.5--C.7 机器人训练与结果 | 存在硬冲突 | 提交前必须解决 |
| C.8 局限 | 诚实且有价值 | 解决冲突后保留 |

方法部分尤其值得保留，包括：

- residual 的归一化坐标和实际决策频率；
- waypoint 的训练、medoid goal 和双 value head；
- stage bonus 与 potential shaping 的严格区分；
- TD3/RLPD 的真实实现细节，包括特殊 critic loss 和 BC residual 坐标。

这些内容比主稿详细很多，确实起到了支撑作用。

## 提交前必须解决的 P0 问题

### 1. 真实机器人 artifact 与论文 claim 冲突

主稿写的是：

- waypoint-conditioned residual 进入 world-model imagination；
- 部署时是完整 SHORE-RL；
- SHORE-RL 在三个真实任务上优于 frozen base 和 ResFit。

补充材料前面同样写部署时运行 navigator。

但 `C.5` 明确审计出：当前保留的 Block artifact 关闭了 subgoal
conditioning、navigator 和在线 value 更新，本质上是 flat residual，不能支持
waypoint-conditioned SHORE-RL 结果。

这是当前最大问题，不能只把这段警告删掉。必须二选一：

1. 找到三个任务真正对应的 SHORE-RL launcher、checkpoint、配置、commit
   和训练记录，并验证评测使用的就是这些 artifact；
2. 如果找不到，就删除或降级主稿中的真实机器人 SHORE-RL claim，把它改成
   “flat residual + self-derived potential 的初步 imagination case study”。

在证据补齐之前，真实机器人结果不能作为完整 SHORE-RL 的主要贡献。

### 2. 消融实验种子数不一致

补充材料声称 Pouring 和 PieceAssembly 的每个消融配置都是 3 seeds，主稿
Figure 5 caption 也写三种子。

但当前 `paper/plot_ablation_bars_400k.py` 中：

- Pouring `subgoal_only`：2 个 run；
- Pouring `no_staged`：2 个 run；
- 其余多数配置：3 个 run。

必须：

1. 找到遗漏的第三个 run，并确认其配置完全一致；或者
2. 将正文、补充材料和图注改成实际种子数，并重新计算 mean、SEM 和相关表述。

同时建议对所有主表和所有图做一次统一 run-inventory 审计。

### 3. 所有正文可见的 TODO 必须消失

目前补充材料还有约 17 个实质性 TODO，集中在：

- 机器人 timeout、随机化和成功标准；
- 数据精确数量；
- world-model 定量验证；
- artifact provenance；
- paired evaluation、interleaving、blind scoring；
- failure-mode 分析。

最终投稿中不能保留 `\todo`。如果信息无法恢复，应删除相应表格或收缩
claim，而不是留下“待验证”。

### 4. Figure 1 的 horizon-collapse 证据没有进入 Supplement

主稿用四个短任务和四个长任务构造 motivation：

- 短任务平均成功轨迹 121 steps；
- 长任务平均成功轨迹 298 steps。

但补充材料没有说明：

- 八个任务分别是什么；
- 121/298 如何计算；
- 每个任务用了多少成功轨迹；
- 对应哪一个 ResFit run/seed；
- 分组标准是什么。

实际画图脚本显示：

- 短任务：Can、Square、CanSort、BoxCleanup；
- 长任务：PieceAssembly、Threading、LiftTray、Pouring；
- 每个任务目前固定到一个 W&B run。

建议增加下表：

| Task | Benchmark | Mean successful length | Max horizon | Base success | ResFit run/seed |
|---|---|---:|---:|---:|---|

否则“collapse 与 horizon 相关”这一开篇论据缺少可审查依据。

### 5. “stage shaping”消融同时删除了 stage-balanced replay

补充材料已经承认，`-stage` 不只删除奖励，还删除 stage-balanced replay。
所以这个实验不能单独归因于 reward shaping。

建议：

1. 全文统一改名为
   `w/o stage-derived branch (bonus + balanced replay)`；或者
2. 补两个干净消融：
   - stage reward on，balanced replay off；
   - stage reward off，balanced replay on。

如果不补实验，至少不要把结果解释成纯 reward effect。

## 建议添加的内容

### 1. 完整 run inventory 和逐种子结果

当前章节标题叫 “Results, Run Inventory, and Provenance”，但实际没有真正列出
run inventory。

建议增加：

| Task | Method | Seed | Run ID | Code commit | Final step | Final-window score |
|---|---|---:|---|---|---:|---:|

仓库里已经有 IBRL 的完整记录：
`paper/figure/IBRL_11_COMPLETED_WANDB_RUNS.md`。其他方法也应采用同样格式。

同时保存一份冻结的 CSV/JSON，避免绘图脚本在投稿后继续从 live W&B 拉取变化
的数据。

### 2. Baseline 超参数与公平性表

现在只详细写了 SHORE-RL，DSRL、IBRL、IQL 基本只有方法介绍。需要补：

- actor/critic 结构；
- learning rate、batch、UTD、discount；
- action/noise scale；
- demonstration 数据使用方式；
- BC anchor；
- execution horizon；
- tuning 搜索范围和最终选择规则；
- 是否获得相同的 dense reward。

这直接关系到主稿 Table 1 是否公平。

### 3. 精确的 stage detector 和 stage-balanced replay

主结果依赖手写 stage bonus，但 Supplement 只给了 \(K\)，没有给每个 stage 的
判定。

实际代码 `resfit/rl_finetuning/chunk_residual/stage_detectors.py` 中已经有完整
定义，建议整理成表格：

- PieceAssembly：双物体抓取 → piece 1 装好 → piece 2 装好；
- Pouring：抓杯 → 球入碗 → 抓碗 → 碗到 pad；
- LiftTray：一块上盘 → 两块上盘 → 抬盘成功；
- Threading：针和支架抓取 → 穿环成功。

还需写清 stage-balanced replay 的采样比例和空 stage 的退化规则。

### 4. Base policy 和数据表

每个任务应列出：

- demonstration 数量；
- dataset ID/version；
- ACT/π0 checkpoint/hash；
- base execution horizon；
- frozen-base success；
- feature 维度；
- goal/value/navigator checkpoint；
- residual seed。

目前只写“ACT 每任务训练、π0 冻结”，还不足以复现。

### 5. Waypoint 机制诊断

现在证明了完整系统有效，但对“waypoint 缩短有效决策时域”的直接证据仍弱。
建议至少增加一项：

- \(k=5/15/25\) 敏感性；
- random latent / shuffled waypoint 对照；
- navigator 预测 waypoint 与真实未来特征的距离；
- waypoint 对未来阶段到达率的提升；
- residual norm、Q calibration 或 late-training drift 分析。

这会显著增强论文最核心的科学解释。

### 6. 真实机器人 world-model 定量验证

仅有挑选出来的 qualitative video 不够。至少需要：

- held-out episode 数量和划分规则；
- 图像预测指标；
- action perturbation/negative probe；
- 真实未来与 imagined future 上的 potential trajectory 一致性；
- 各任务分别报告，而不只是 Block；
- dynamics checkpoint hash、训练配置和服务配置的不可变绑定。

尤其需要解决 `action_norm_json` 缺失、训练归一化与推理固定 `[-1,1]` 是否一致
的问题。

### 7. 真实机器人评测完整记录

应补充：

- 六个 checkpoint 各自的成功次数，而不只保留 0 和 500k；
- 每个 checkpoint 是否都是 50 trials；
- 每个方法有几个独立训练 seed；
- 初始状态集合和 reset 程序；
- trial 是否逐一配对；
- 条件是否交错执行；
- scorer 是否盲评；
- invalid/aborted trial 如何处理。

Wilson CI 表是合适的，建议保留。

## 建议删除或压缩

以下内容不适合以当前形式出现在最终 Supplement：

- 开头 `currently recoverable`、`unavailable fields are marked` 等内部审计式摘要；
- 所有 `TODO` 和 draft helper；
- 具体 GPU 编号、端口、内部进程分配；
- 多次重复的 `retained block command` 取证叙述；
- `eval/success_rate=0.0` placeholder 的长篇解释；
- 未 held-out、与真实成功率无明确关系的 Block proxy 表；
- 只有总失败数、mode breakdown 全是 TODO 的失败分析表。

但是，涉及 artifact 不匹配、world-model 未验证等问题，不能通过简单删除来
掩盖。应该先解决；解决不了就同步删除或弱化主稿 claim。

任务过程图片可以保留，但三组超宽序列图占用较大。可以压缩为每任务 3--4
帧，腾出空间给逐种子结果和 provenance 表。

## 推荐的新结构

这里列出的内容顺序，不代表一定要写成 10 个同级大章节。更推荐保留现有
A/B/C 框架，在内部重组。

### A. Method and Implementation Details

1. A.1 Residual Policy, Coordinates, and Decision Time Scale
2. A.2 Goal-Conditioned Value and Waypoint Navigator
3. A.3 Dense Progress Rewards: Stage Bonus and Learned Potential
4. A.4 Residual Actor--Critic Objective and Updates
5. A.5 Unified Training Loop, Architectures, and Shared Settings

### B. Simulation Experiments

1. B.1 Simulation Tasks, Datasets, and Success Criteria
2. B.2 Base Policies and Frozen Representations
3. B.3 Exact Stage Detectors and Stage-Balanced Replay
4. B.4 Training Configurations and Matched Baselines
5. B.5 Evaluation Protocol and Metrics
6. B.6 Main Results, Per-Seed Results, and Run Inventory
7. B.7 Component and Reward-Form Ablations
8. B.8 Horizon and Waypoint Diagnostics
9. B.9 Simulation Limitations

其中 B.6 应真正列出逐种子结果、Run ID、commit 和 final-window score；B.8
用于补 \(k\) 敏感性、waypoint 质量和 horizon-collapse 数据来源。

### C. Real-Robot Experiments

1. C.1 Platform, Tasks, and Data
2. C.2 Dynamics Model and Training Configuration
3. C.3 Held-Out Dynamics-Model Validation
4. C.4 Residual Training in Imagination
5. C.5 Physical Evaluation Protocol
6. C.6 Results and Artifact Provenance
7. C.7 Real-Robot Limitations

这里必须先解决真实机器人 artifact 是否真正启用了 waypoint。若无法解决，
C 部分应改为明确的 preliminary flat-residual case study，同时收缩主稿 claim。

### D. Reproducibility Summary（可选）

用一两张总表集中列出：

- dataset/version；
- base checkpoint；
- value/navigator checkpoint；
- seeds 与 run IDs；
- code commit；
- 训练与评测预算；
- artifact hash。

核心建议不是推翻现有 Supplement，而是保留当前 A 方法、B 仿真、C 真实机器人
三级结构，把缺失的实验支撑补进对应小节，并删掉内部审计式、重复或无效的内容。
这样比 10 个同级章节更自然，也更容易让审稿人从主稿 claim 快速定位到对应
证据。

## A 章节现有内容与推荐结构对比

对比后可以发现：推荐的前四节与现有前四节基本一一对应，主要是标题更贴近
主稿叙事；第五节则存在实质差别。若机械采用此前的
`A.5 Network Architecture and Hyperparameters`，会丢掉现有 A.5 中很有价值的
统一算法和双场景对照，因此需要修正。

| 现有章节 | 现在讲什么 | 推荐章节 | 差别 |
|---|---|---|---|
| A.1 Residual Coordinates and Decision Time Scale | actor/critic 输入、残差坐标、zero-init、combined action、仿真逐步 residual 与机器人 50-step chunk | A.1 Residual Policy, Coordinates, and Decision Time Scale | 内容基本相同，只需使标题更完整 |
| A.2 Waypoint Implementation Details | terminal medoid goal、bottleneck \(\phi\)、双 value head、HIQL 目标、navigator AWR、在线联合更新 | A.2 Goal-Conditioned Value and Waypoint Navigator | 完全对应，推荐标题更具体 |
| A.3 Potential and Stage-Bonus Implementations | stage bonus、单状态 \(V_p\)、potential scaling、terminal/truncation handling | A.3 Dense Progress Rewards: Stage Bonus and Learned Potential | 完全对应，推荐标题是更清晰的上位概括 |
| A.4 Residual Off-Policy Reinforcement Learning | TD3/RLPD、10 critics、n-step target、特殊 critic loss、demo-BC、探索噪声 | A.4 Residual Actor--Critic Objective and Updates | 完全对应，主要是标题聚焦 |
| A.5 Shared Update, Two Venue Instantiations | 统一算法、simulation/real-robot collector 差异、replay 差异、共享超参数表 | A.5 Unified Training Loop, Architectures, and Shared Settings | 保留现有功能，同时补网络结构 |

### A.1 的处理建议

现有 A.1 已经比原推荐标题包含得更多：

- residual actor/critic 的输入；
- base feature 只供 navigator 使用；
- actor 与 critic 使用独立的可训练视觉编码器；
- residual 的归一化坐标；
- zero initialization；
- simulation residual 每控制步决策；
- real-robot residual 每 50 步决策。

因此建议仅将标题改成：

> A.1 Residual Policy, Coordinates, and Decision Time Scale

主体内容不需要重写。

### A.2 的处理建议

现有 A.2 已经完整覆盖：

- goal medoid；
- goal-conditioned value；
- bottleneck representation；
- twin value heads；
- goal sampling；
- navigator AWR；
- offline pretraining；
- online joint finetuning；
- dual-arm 与 LIBERO 的差异。

建议标题改成：

> A.2 Goal-Conditioned Value and Waypoint Navigator

可以额外补充一个目前略缺的细节：在线轨迹，尤其失败轨迹，如何采样
current/future/goal，以及在线 value 和 navigator 分别使用哪些 mask。

### A.3 的处理建议

现有内容本质上就是 Dense Reward Construction，而且已经很好地区分：

- stage bonus 不是 potential-based；
- \(V_p\) 与 \(V_{\mathrm{gc}}\) 是两个独立网络；
- simulator terminal 与 imagination artificial truncation 的处理不同。

建议标题改成：

> A.3 Dense Progress Rewards: Stage Bonus and Learned Potential

精确的各任务 stage detector 不需要塞进这里。A.3 保留通用数学定义，具体
谓词放到 B.2。

### A.4 的处理建议

现有 A.4 已经是比较完整的 TD3/RLPD objective，包括很多通常 Supplement
会漏掉的实现细节。

建议标题改成：

> A.4 Residual Actor--Critic Objective and Updates

这里基本不需要重构。

### A.5 的处理建议

此前推荐的：

> A.5 Network Architecture and Hyperparameters

如果直接替换当前 A.5，会丢掉：

- 统一训练算法；
- simulation 与 imagination 的 collector 对照；
- reward source 和 replay sampler 差异；
- shared update 与 venue-specific instantiation 的关系。

这些内容有价值，不应删除。更合适的标题是：

> A.5 Unified Training Loop, Architectures, and Shared Settings

内部顺序可以调整为：

1. unified training algorithm；
2. simulation vs. real-robot instantiation table；
3. shared actor/critic/value architecture；
4. shared hyperparameters；
5. 指向 B.3 和 C.4 的 venue-specific 参数。

当前 A.5 的 shared hyperparameter 表已经覆盖 MLP、critic 数量和学习率等，
但还缺真正的视觉网络细节，建议补充：

- multi-view encoder 的具体结构；
- 输入图像尺寸和相机数量；
- augmentation；
- 各相机特征如何融合；
- actor/critic 是否共享 encoder；
- encoder 由哪些 loss 更新；
- proprioception、base action、waypoint 的拼接位置；
- 参数量。

### A 章节最终建议

最终建议的 A 章节为：

1. **A.1 Residual Policy, Coordinates, and Decision Time Scale**
2. **A.2 Goal-Conditioned Value and Waypoint Navigator**
3. **A.3 Dense Progress Rewards: Stage Bonus and Learned Potential**
4. **A.4 Residual Actor--Critic Objective and Updates**
5. **A.5 Unified Training Loop, Architectures, and Shared Settings**

本质上不需要重写 A 章节：

- 前四节主要改标题并补少量缺失细节；
- 第五节保留当前统一算法与双场景对照，同时补全网络结构；
- exact stage detector、per-task 参数等具体内容继续放在 B/C，而不是全部挤进 A。

当前 A 章节整体是 Supplement 中最成熟的部分，不需要大改结构。

## B 章节现有内容与推荐结构对比

现有 B.1--B.8 的骨架总体合理，不应该把 B.1 任务和 B.2 基策略强行合并。
真正缺失的是精确 stage 定义、实际逐种子结果和 waypoint/horizon 诊断。

| 现有章节 | 当前内容 | 推荐去向 | 判断 |
|---|---|---|---|
| B.1 Simulated Tasks and Success Criteria | DexMimicGen/LIBERO 任务、horizon、动作维度、成功标准、任务过程图 | B.1 Simulation Tasks, Datasets, and Success Criteria | 基本保留，补数据集信息 |
| B.2 Base Policies and Frozen Features | ACT/π0、冻结特征、特征维度、execution horizon | B.2 Base Policies and Frozen Representations | 应独立保留，不应并入 B.1 |
| B.3 Configuration and Hyperparameters | waypoint/value 配置、SHORE-RL 训练参数 | B.4 Training Configurations and Matched Baselines | 保留，与 baseline 参数放在相邻位置 |
| B.4 Baselines | ResFit、DSRL、IBRL、IQL 的概念说明 | B.4 Training Configurations and Matched Baselines | 需要补成可复现的配置表 |
| B.5 Ablation Design | 五种组件消融、reward-form 对照、coverage | B.7 Component and Reward-Form Ablations | 内容基本合适，移到主结果之后更自然 |
| B.6 Evaluation Protocol | evaluator、episode 数、预算、final-window 定义 | B.5 Evaluation Protocol and Metrics | 基本保留 |
| B.7 Results, Run Inventory, and Provenance | 种子数、预算、纳入标准、coverage caveat | B.6 Main Results, Per-Seed Results, and Run Inventory | 标题合理，但实际内容不够 |
| B.8 Limitations | 统计覆盖、特权 stage、LIBERO 可比性 | B.9 Simulation Limitations | 基本保留 |
| 当前缺失 | 精确 stage detector 与 balanced replay | 新 B.3 | 必须增加 |
| 当前缺失 | horizon/waypoint 机制诊断 | 新 B.8 | 建议增加 |

### B.1 的处理建议

当前 B.1 已经讲清楚：

- 五个 DexMimicGen 任务；
- 两个 LIBERO 任务；
- action dimension；
- episode horizon；
- stage 数 \(K\)；
- success criterion；
- 控制频率；
- 任务执行过程。

它不需要与 B.2 合并。建议标题改为：

> B.1 Simulation Tasks, Datasets, and Success Criteria

需要增加：

- dataset ID/version；
- demonstration 数量；
- train/eval 数据来源；
- Figure 1 中四个短任务和四个长任务的完整清单；
- 121/298 successful trajectory length 的计算方法和样本量。

五组任务过程图可以保留，但建议压缩，否则占用空间较大。

### B.2 的处理建议

当前 B.2 很有价值，已经解释：

- DexMimicGen 使用 ACT；
- LIBERO 使用 \(\pi_0\)；
- waypoint 使用冻结基策略特征；
- residual actor/critic 不使用该冻结特征；
- ACT feature 是 512-D mean-pooled token；
- \(\pi_0\) feature 是 2048-D prefix token；
- base execution horizon 为 10；
- privileged `eef_piece` 只用于 upper-bound ablation。

此前将它合并到 `Tasks, Datasets, and Base Policies` 不够合理。建议继续独立：

> B.2 Base Policies and Frozen Representations

需要补充：

- 每个 base checkpoint/hash；
- base 训练步数与优化器；
- base 使用的 demonstration 数量；
- 每个任务 frozen-base success；
- 图像预处理和 feature cache 的生成方式；
- LIBERO serve checkpoint、prompt 和 pooling 配置。

### 新增 B.3

当前文档只在 B.1 给出了 stage 数 \(K\)，在配置表中写了
`direct +1 per stage`，但没有定义每个 stage 如何判定。

建议新增：

> B.3 Exact Stage Detectors and Stage-Balanced Replay

内容应包括：

- PieceAssembly：双物体抓取 → piece 1 装好 → piece 2 装好；
- Pouring：抓杯 → 球入碗 → 抓碗 → 碗到 pad；
- LiftTray：一块上盘 → 两块上盘 → 抬盘成功；
- Threading：针和支架抓取 → 穿环成功；
- stage latch 的 `max-so-far` 规则；
- stage bonus 的触发时机；
- stage-balanced replay 的采样比例；
- 某个 stage 没有样本时如何处理；
- offline 和 online replay 是否使用相同 stage label。

这些定义已经存在于
`resfit/rl_finetuning/chunk_residual/stage_detectors.py`，可以直接整理成表格。

### 现有 B.3 与 B.4 的处理建议

当前 B.3 对 SHORE-RL 写得比较详细，但 B.4 对 baseline 只有概念介绍。
建议将它们组织为：

> B.4 Training Configurations and Matched Baselines

内部先后顺序：

1. SHORE-RL simulation-specific configuration；
2. ResFit 配置；
3. DSRL 配置；
4. IBRL 配置；
5. full-policy IQL 配置；
6. 公平性和调参预算。

需要为每个 baseline 列出：

- network；
- learning rate；
- batch、UTD、discount；
- action/noise scale；
- offline demonstration 是否使用；
- BC anchor；
- execution horizon；
- training budget；
- seed；
- evaluator；
- 超参数选择方法。

A.5 已经给出 shared architecture，B.4 应尽量只写 simulation-specific 参数和
baseline 差异，避免重复。

### 现有 B.6 的处理建议

当前 B.6 基本成熟，建议改名：

> B.5 Evaluation Protocol and Metrics

保留：

- 8 个并行环境；
- DexMimicGen 每次 50 episodes；
- LIBERO 每次 10 episodes；
- 500k/400k budget；
- final 20% checkpoint window；
- seed-level mean 后再计算 SEM；
- step-0 作为 frozen-base reference。

建议补充：

- numeric seed IDs；
- evaluation frequency；
- final-window 中具体包含哪些 checkpoint；
- 缺失 checkpoint 如何处理；
- curve aggregation 是否只使用共同 checkpoint；
- LIBERO DSRL 为什么不完全 matched。

### 现有 B.7 的处理建议

当前 B.7 是 B 章节最需要补强的部分。标题叫
`Results, Run Inventory, and Provenance`，但实际只有“有几个 seeds、跑到多少步”
的文字，没有：

- 逐种子分数；
- Run ID；
- code commit；
- base checkpoint；
- final-window checkpoint；
- artifact/hash；
- 主表数值的可复算来源。

建议改为：

> B.6 Main Results, Per-Seed Results, and Run Inventory

至少加入：

| Task | Method | Seed | Run ID | Commit | Final step | Final-window success |
|---|---|---:|---|---|---:|---:|

还需要解决当前 Pouring 消融两项只有两个 run ID、正文却写三个 seeds 的冲突。

### 现有 B.5 的处理建议

当前 B.5 对消融设计说明得比较清楚，建议放到主结果之后：

> B.7 Component and Reward-Form Ablations

保留：

- Full SHORE-RL；
- w/o stage branch；
- w/o waypoint；
- w/o waypoint + stage；
- waypoint only；
- stage bonus vs self-derived potential。

但要统一说明：`w/o stage` 删除的是 stage reward 和 stage-balanced replay
整个分支，不是只删除 reward shaping。

如果不增加 reward/replay 的正交消融，图例也应改为
`w/o stage-derived branch`。

### 新增 B.8

建议新增：

> B.8 Horizon and Waypoint Diagnostics

可包含：

- Figure 1 八任务的 horizon 数据来源；
- \(k=5/15/25\) 敏感性；
- shuffled/random waypoint；
- waypoint 与真实未来 feature 的距离；
- stage reach rate；
- residual norm 或 drift；
- critic/Q calibration；
- potential 与 stage progress 的相关性。

这不是纯复现信息，而是用于直接支撑
“effective horizon shortening”这一核心解释。

### 现有 B.8 的处理建议

当前 B.8 内容合理，建议保留为：

> B.9 Simulation Limitations

其中关于三种子覆盖、privileged stage structure、LIBERO DSRL 非严格匹配的
说明都应该保留。

### B 章节最终建议

最终建议的 B 章节为：

1. **B.1 Simulation Tasks, Datasets, and Success Criteria**
2. **B.2 Base Policies and Frozen Representations**
3. **B.3 Exact Stage Detectors and Stage-Balanced Replay**
4. **B.4 Training Configurations and Matched Baselines**
5. **B.5 Evaluation Protocol and Metrics**
6. **B.6 Main Results, Per-Seed Results, and Run Inventory**
7. **B.7 Component and Reward-Form Ablations**
8. **B.8 Horizon and Waypoint Diagnostics**
9. **B.9 Simulation Limitations**

现有 B 章节不需要推翻：

- B.1、B.2、B.6、B.8 基本保留；
- B.3 和 B.4 合并整理为完整的训练与 baseline 配置；
- B.5 移到主结果之后；
- B.7 补成真正的 per-seed results 与 run inventory；
- 新增精确 stage 定义和 waypoint/horizon diagnostics。

与 A 章节相比，B 章节需要的结构调整更大，主要原因不是现有内容写错，而是
核心实验的可复现证据尚未真正放进 Supplement。

## C 章节现有内容与推荐结构对比

总体看，现有 C 章的技术细节很多，但组织方式更像“内部审计记录”；推荐结构
则围绕真实机器人结论的证据链组织：

> 数据从哪里来 → dynamics model 是否可信 → residual 如何训练 →
> 实机怎样评测 → 结果对应哪些 artifact。

现有 C.1--C.8 与推荐 C.1--C.7 的对应关系如下。

| 现有章节 | 现在讲什么 | 推荐章节 | 处理方式 |
|---|---|---|---|
| C.1 Platform and Control | 机器人、相机、控制频率、动作维度、VLA、部署组件 | C.1 Platform, Tasks, and Data | 与现有 C.2、C.3 合并 |
| C.2 Tasks and Success Criteria | 三个任务、成功条件、随机化、二值评价 | C.1 Platform, Tasks, and Data | 合并，但补全 TODO |
| C.3 Demonstration and Rollout Data | 专家数据、base rollout、数据规模 | C.1 Platform, Tasks, and Data | 合并并解决数量冲突 |
| C.4 Dynamics Model | 模型架构、数据、训练配置、定性检查 | C.2 + C.3 | 拆为“模型训练”和“独立验证” |
| C.5 Residual RL in Imagination | imagined rollout、replay、reward、训练配置及 artifact 审计 | C.4 Residual Training in Imagination | 保留核心方法，但必须解决 artifact 冲突 |
| C.6 Evaluation Protocol | 50 trials、初始状态、统计方法 | C.5 Physical Evaluation Protocol | 基本对应，但信息尚未补齐 |
| C.7 Results and Failure Analysis | 0/500k 成功率、Wilson CI、失败总数 | C.6 Results and Artifact Provenance | 加入完整曲线、artifact 和 trial provenance |
| C.8 Limitations | world-model bias、potential bias、弱 action conditioning 等 | C.7 Real-Robot Limitations | 基本保留并压缩 |

### 现有 C.1--C.3 与推荐 C.1

推荐：

> C.1 Platform, Tasks, and Data

现有 C.1--C.3 分别介绍平台、任务和数据，逻辑本身没有问题，而且信息比较
完整。例如：

- 双臂 16 维关节动作；
- 三个 RGB 相机、30 Hz；
- 每次执行 50-step base-policy chunk；
- 三个实机任务及成功标准；
- 每任务约 1,000 条示范；
- frozen-base rollout 用于 dynamics-model finetuning。

推荐合并的主要原因是压缩篇幅，并把“实验对象和数据来源”集中说明，而不是
因为现有内容错误。合并后可以在 C.1 内使用三个小标题：

1. Platform and Observation/Action Interface
2. Tasks and Success Criteria
3. Demonstrations and Base-Policy Rollouts

但以下缺失仍需补齐：

- 每个任务的 timeout；
- 物体随机化范围和机器人初始姿态；
- 固定评测初始状态集如何构造；
- 精确 demonstration 数量，不能只写约 1,000；
- `约 1,000 demonstrations` 与 C.5 中 `295 demonstrations` 的关系；
- 每个任务的 base checkpoint、训练数据量和 frozen-base success；
- rollout 中成功与失败各有多少。

因此，这部分主要是“合并、压缩和补数据”，不是重新写方法。

### 现有 C.4 与推荐 C.2、C.3

这是结构调整最重要的地方。

现有 C.4 把两类性质不同的内容放在了一起：

- C.4.1--C.4.3：模型是什么、怎样训练；
- C.4.4：模型是否真的可以作为 RL 环境。

推荐拆成：

> C.2 Dynamics Model and Training Configuration  
> C.3 Held-Out Dynamics-Model Validation

#### 推荐 C.2

主要吸收现有 C.4.1--C.4.3，保留：

- LTX-Video backbone；
- 三视角输入；
- 4 帧 history、25 帧预测；
- action conditioning；
- pooled three-task training data；
- mixed successful/failure rollouts；
- frame-rate 和 caption shortcut controls；
- optimizer、batch、训练步数；
- dynamics checkpoint hash。

需要重点解决：

- 训练时 action normalization 与推理时固定 `[-1,1]` 是否一致；
- dynamics checkpoint、训练配置、数据版本和推理服务之间的不可变绑定；
- 三个任务是否确实使用同一个 step-18k checkpoint。

内部端口、具体 GPU 编号、服务进程分配等内容可以删除或大幅压缩。

#### 推荐 C.3

现有 C.4.4 目前不能算真正的 validation section。它主要包含：

- 三个挑选出来的 qualitative examples；
- 应该进行哪些验证的文字说明；
- 一个非 held-out 的 Block value-proxy 表；
- 大量尚未完成的 TODO。

推荐 C.3 应真正报告：

- held-out episode 数量和划分方法；
- 每个任务的 rollout prediction metric；
- action perturbation/negative probe；
- 不同 action 是否生成明显不同的未来；
- predicted future 与真实 future 上的 potential trajectory 一致性；
- 每个任务分别报告，而不只是 Block。

当前 Block proxy 没有真实未来作为对照，也不是 success metric，不能替代
held-out validation。它可以删除，或者降级为一小段 training diagnostic。

### 现有 C.5 与推荐 C.4

推荐：

> C.4 Residual Training in Imagination

现有 C.5 已经非常详细地说明了：

- chunk-level dynamics interaction；
- 两段 imagined rollout；
- 50/50 demonstration--imagination replay；
- critic warmup、UTD、discount、BC coefficient；
- potential-only reward；
- 哪些模型冻结、哪些网络训练。

这些内容可以保留并整理成配置表。

但这里存在整个 C 章最严重的问题：

- C.1 说部署时使用 navigator 和 waypoint；
- C.6、C.7 报告 SHORE-RL 实机结果；
- C.5 审计出的现有 Block artifact 却关闭了 subgoal、navigator 和在线 value
  更新，本质上是 flat residual。

因此，推荐 C.4 不能只是给现有 C.5 改名。必须先找到真正对应三个任务
SHORE-RL 结果的：

- launcher/config；
- residual checkpoint；
- navigator/value checkpoint；
- dynamics checkpoint；
- seed、commit 和 Run ID；
- 实机评测所加载的 artifact。

如果找不到，真实机器人部分只能称为 flat residual imagination case study，
不能支撑完整 SHORE-RL 的真实机器人结论。

### 现有 C.6 与推荐 C.5

推荐：

> C.5 Physical Evaluation Protocol

现有 C.6 已经规定了 50 trials、固定初始状态集和 Wilson interval，方向正确。

但现在仍缺：

- 每个任务固定初始状态集的具体组成；
- trial 是否一一配对；
- reset procedure；
- 三种条件是否交错执行；
- session 日期和顺序；
- scorer 是否盲评；
- invalid/aborted trial 的处理规则；
- 每次评测实际加载的 checkpoint；
- 是否每个 checkpoint 都进行了 50 次试验。

如果找不到 trial-level pairing 信息，就不要做 paired test，也不要声称这是
严格的 paired evaluation；只报告独立的 marginal success rate 和 interval。

### 现有 C.7 与推荐 C.6

推荐：

> C.6 Results and Artifact Provenance

现有 C.7 的优点是已经报告：

- 成功次数，而不只是百分比；
- Wilson 95% interval；
- frozen base、ResFit 和 SHORE-RL 三种条件。

但当前仅有 step 0 和 step 500k 的聚合结果。主稿声称有六个 checkpoint 的
曲线，因此需要补：

| Task | Method | Training seed | Eval checkpoint | Artifact/hash | Successes / 50 |
|---|---|---:|---:|---|---:|

这一节还应说明：

- 六个 checkpoint 的完整成功次数；
- 每个方法使用多少训练 seed；
- 每组实机结果对应哪个 residual checkpoint；
- base、dynamics、value、navigator 和 residual artifact 的组合；
- commit 和配置文件；
- trial/session provenance。

目前的 failure-analysis 表只有总失败数，具体 failure mode 全是 TODO。如果
不能重新查看视频并按预先定义的类别标注，建议整张表删除。`50-successes`
得到的失败总数并不能构成 failure analysis。

### 现有 C.8 与推荐 C.7

推荐：

> C.7 Real-Robot Limitations

现有 C.8 是 C 章中比较成熟的部分，以下内容都值得保留：

- RL 实际在 learned dynamics model 中训练；
- model bias 会转化为 policy bias；
- potential 不是真实物理 reward；
- action conditioning 较弱；
- 三任务共享一个 dynamics model；
- real-robot interaction 仅用于数据收集和评测。

建议压缩重复内容，并补充：

- 每个方法训练 seed 数量有限；
- 三个任务覆盖范围有限；
- dynamics model 尚未完成 held-out validation；
- 如果最终无法找到 SHORE artifact，应明确说明真实机器人结果只验证了
  flat residual。

### C 章节最终建议

现有 C 章不需要全部推翻。最合理的调整是：

1. 将现有 C.1--C.3 合并成实验平台、任务和数据；
2. 将现有 C.4 拆成“模型训练”和“独立验证”；
3. 保留 C.5 的 imagination-training 技术细节，但先解决 artifact 冲突；
4. 补全 C.6 的实机评测协议；
5. 将 C.7 改造成“结果 + artifact provenance”；
6. 保留并压缩 C.8。

推荐结构为：

1. **C.1 Platform, Tasks, and Data**
2. **C.2 Dynamics Model and Training Configuration**
3. **C.3 Held-Out Dynamics-Model Validation**
4. **C.4 Residual Training in Imagination**
5. **C.5 Physical Evaluation Protocol**
6. **C.6 Results and Artifact Provenance**
7. **C.7 Real-Robot Limitations**

结构变化不是当前 C 章的核心难点。真正决定这部分能否提交的是两个问题：
能否完成 dynamics-model 的 held-out validation，以及能否证明实机 SHORE-RL
结果确实来自启用了 waypoint/navigator 的对应 artifact。

## 其他一致性和排版问题

- 主稿标题是 `Effective Horizons`，补充材料目前是 `Effective Horizon`，需要统一。
- 当前 Supplement PDF 为 16 页。长度本身不是主要问题，问题是仿真证据偏少、
  真实机器人内部审计偏多。
- 当前编译没有 undefined citation/reference，但存在少量
  overfull/underfull；这些排版问题的优先级远低于上述内容和证据链问题。

## 最终判断

当前版本：

- 方法说明基本够；
- 仿真实验支撑不够；
- 真实机器人证据链尚不成立；
- 不适合直接作为最终 Supplement 提交。

优先补齐真实机器人 artifact、消融种子数、逐种子结果、baseline 配置、
stage detector 和 horizon-collapse provenance，再进行删减与排版整理。
