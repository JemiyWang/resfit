# AAAI AC 审稿意见：SHORE-RL

审阅对象仅为 `main.pdf`，未参考代码、补充材料或外部资料。

## AC 结论

- 建议：**Weak Reject（3/6，接近 borderline）**
- 信心：**4/5**

这篇论文的系统效果很强，问题也重要；但现有证据主要支持“waypoint、BC 约束与稠密奖励的组合有效”，尚不足以支持论文最核心的科学结论——“通过 waypoint 缩短目标时域，是避免后期崩溃的关键原因”。

## 论文概述

SHORE-RL 冻结 BC/VLA 基策略，并学习有界动作 residual。方法包含三个主要部分：

- 从基策略特征中预测未来 latent waypoint；
- 用 waypoint 条件化 residual actor/critic；
- 用手工阶段奖励或学习到的 potential，为 critic 提供稠密反馈。

仿真结果覆盖四个 DexMimicGen 长时程任务、两个 LIBERO 任务；真实机器人部分利用学习到的动力学模型进行 imagination training，并在三个任务上测试。

## 优点

- 研究问题重要。论文明确关注“短期改善、后期跌破 frozen base”的现象，比只报告峰值性能更有实际意义。
- `final-window success` 指标和完整学习曲线是合理的，减少了挑选最佳 checkpoint 的空间。
- 实验覆盖面较广，包括 ACT、π0、π0.5-style 基策略，多种 RL baseline，以及仿真和真实机器人。
- 完整方法的数值提升显著。Table 1 中 SHORE-RL 在四项长时程任务达到 `.91/.66/.85/.52`，远高于所列基线。
- potential-based shaping 替代手工阶段检测器的实验有价值：Fig. 7 中 Threading 为 `.52 vs .52`，PieceAssembly 为 `.70 vs .66`。
- 论文整体清楚，图表可读，作者也谨慎承认 critic error 和 residual drift 尚不是被独立证明的崩溃原因。

## 主要问题

### 1. 方法并未在形式上真正“缩短 objective horizon”

这是最根本的问题。

Eq. (3) 中 actor 仍然最大化 TD3 critic 的任务回报；Algorithm 1 的 n-step target 仍然沿完整剩余 episode bootstrap。Waypoint 没有对应的局部到达奖励、局部终止条件、option boundary 或独立局部价值函数。

因此，`z_t` 更准确地说是一个从示范中学到的条件特征或策略先验，而不是一个已经证明会缩短 Bellman/credit-assignment horizon 的机制。真正直接改变局部学习信号的是 dense reward shaping。

论文也没有提供以下关键证据：

- waypoint 距离 \(k\) 的敏感性或 horizon scaling；
- waypoint 可达率、预测质量或实际时间距离；
- local credit horizon 的测量；
- critic error、Q calibration、residual norm/drift 的诊断；
- random latent、普通未来特征或 oracle waypoint 对照。

所以标题和结论中的 “shortening the horizon” 目前更像解释性假说，而不是被验证的机制。

### 2. Fig. 2 不是严格的受控 horizon 实验

所谓短时程组和长时程组由不同任务构成，部分还来自不同 benchmark。任务难度、基策略初始质量、双臂协调、接触复杂度、奖励密度和状态分布都可能共同变化。

因此该实验不能“隔离” horizon，也不足以推出 temporal composition 是 central empirical factor。更可信的设计应在同一任务中只改变阶段数、episode 长度或任务串联次数，并保持奖励、基策略和动力学条件一致。论文甚至没有报告各任务的实际 horizon 或平均成功轨迹长度。

### 3. 主基线比较混合了多个额外优势

完整 SHORE-RL 同时使用：

- waypoint；
- demo-BC loss；
- 手工 per-stage dense reward。

正文没有明确说明 Table 1 中 ResFit、DSRL、IBRL 和 IQL 是否获得相同的 dense reward 与辅助监督。如果没有，Table 1 不能把差异归因于 waypoint 或 horizon shortening。

最接近公平对照的是 Fig. 6 的 `w/o waypoint`：保留 BC 和 stage shaping，只移除 waypoint。但它只在两个任务、400k budget 下报告：

- Pouring：完整方法 `.93`，无 waypoint `.92`，几乎相同；
- PieceAssembly：`.65` 对 `.28`，显示 waypoint 在该任务上有明显条件性贡献。

这说明 waypoint 的作用高度依赖任务和其他组件。要支撑主结论，应在全部四个任务、相同 500k budget 下报告 `ResFit + same BC + same shaping`，并最好完成三组件的完整 factorial ablation。

### 4. 复现信息不足

PDF 多次写明关键细节“deferred to the supplement”，但这份文件并不包含 supplement。缺失内容包括：

- \(k/H\)、waypoint bottleneck 和 latent 维度；
- residual action bounds、动作 chunk 合成方式；
- stage detector、奖励系数 \(b,c\) 及其调参方式；
- \(V_{\mathrm{gc}}\)、\(V_p\)、navigator 的架构与训练过程；
- online failed trajectories 如何用于 navigator/value 更新；
- baseline 架构、初始化与调参预算；
- dynamics model 架构、rollout 长度、buffer mixing 和模型误差评估。

尤其是 Algorithm 1 第 9 行“在 mixed completed trajectories 上更新 \(V_{\mathrm{gc}}\)、\(\pi_h\)”，与 Eq. (4) 的 demonstration-future objective 如何对应并不清楚。

### 5. 统计证据和真实机器人部分仍有限

仿真只有三种子，尚可作为机器人论文常见设置，但缺少 individual-seed 结果和显著性分析。

真实机器人每个表格单元有 50 次测试，却没有置信区间、训练随机种子或独立训练重复。Paper-Roll 上 SHORE-RL `.42` 对 ResFit `.28` 的差距，在只有一次训练实例时并不十分稳固。

此外：

- 没有报告动力学模型的预测质量或 OOD/action exploitation 检查；
- 不清楚 ResFit 是否使用了完全相同的 imagination data、reward 和 replay mixture；
- LIBERO 每个 checkpoint 每种子只有 10 episodes，且只与 DSRL 比较；
- “base-policy-agnostic”表述偏强，因为方法需要访问基策略内部 encoder、示范数据、终态 goal medoid 和任务相关 proprioception。

## 希望作者在 rebuttal 中回答

1. 除条件化 critic/actor 外，是否存在正文未说明的 waypoint 局部奖励或终止机制？若没有，严格意义上的 horizon shortening 如何定义和测量？
2. Table 1 的所有基线是否使用相同的 stage/potential reward、示范数据、BC loss 和视觉特征？
3. 能否提供四个主任务上保留所有设置、仅移除 waypoint 的结果？
4. Fig. 2 中各任务的实际 horizon、阶段数和 frozen-base success 分别是多少？
5. 真实机器人结果是否来自多个独立训练 run？动力学模型如何验证，ResFit 是否使用相同模型数据？
6. 论文所引用但未包含的 supplement 是否实际随投稿提交？

## 评分

- 技术可靠性：**3/5**
- 新颖性：**3/5**
- 潜在影响：**4/5**
- 表达清晰度：**4/5**
- 可复现性：**2/5**
- 总体评分：**Weak Reject（3/6）**
- 审稿信心：**4/5**

## 最终判断

这项工作有明显潜力，完整系统的经验表现也令人信服；但当前稿件把“有效的多组件系统”解释成了“waypoint 缩短时域，因此解决崩溃”，而现有实验尚不能完成这一步因果归因。

由于补充实验涉及同任务 horizon 控制、全任务公平消融和真实机器人重复，并非仅靠文字澄清即可解决，因此建议 **Weak Reject**，而不是 Weak Accept。

---
方法并未在形式上真正“缩短 objective horizon”  对于这个问题 我们从两方面缩短horizon  一方面 我们使用了latent 空间的waypoint特征 作为条件 输入到策略网络中 因此 策略网络看到相比最终状态
距离当前状态更近的状态  相当于他在更短的horizon任务上执行动作   另一方面  使用staged 或者dense reward  使得奖励信号距离初始状态更近  更容易传播回初始动作中  指导动作改进    这两种机制不
就是隐式缩短了任务的horizon嘛？


 对，如果把 horizon 定义为“有效决策 horizon”和“有效信用分配 horizon”，你们的解释是成立的。

  我之前的质疑不是说这两种机制没有缩短作用，而是需要区分：

  - 环境/任务 horizon
  - RL objective horizon
  - 有效决策 horizon
  - 有效信用分配 horizon

  你们缩短的是后两者，并没有改变环境本身的 episode 长度。

  ### 1. Latent waypoint 缩短有效决策 horizon

  你们的 waypoint 来自距离当前状态不超过 k 步的未来示范状态。于是 residual policy 不必直接考虑“如何从当前状态完成整个任务”，而是根据一个时间上更近的局部目标决定当前动作。

  可以理解为：

  当前状态 -> 最终目标

  被分解为：

  当前状态 -> 附近 waypoint -> 下一个 waypoint -> ... -> 最终目标

  所以，策略每次需要处理的目标距离确实缩短了。这可以称为：

  - decision-conditioning horizon shortening
  - temporal localization
  - implicit local subgoal decomposition

  但这里有一个严格性问题：当前方法中的 waypoint 只是 actor 和 critic 的条件输入。Actor 仍通过 TD3 critic 优化整个 episode 的累计回报；系统没有明确规定：

  - 必须在 k 步内到达 waypoint；
  - 到达 waypoint 后局部 episode 终止；
  - waypoint 有独立的局部到达奖励；
  - residual policy 只优化到 waypoint 为止的回报。

  因此，更准确的说法是：

  > Waypoint 隐式缩短了策略需要解析的目标距离，但没有在数学上把原始 Bellman objective 的 horizon 从 T 改成 k。

  这不意味着机制不成立，只意味着它目前属于一种有根据的 inductive bias，而不是严格的 MDP horizon reduction。

  ### 2. Staged/dense reward 缩短有效信用分配 horizon

  这一部分更加直接。

  Sparse reward 下，早期动作可能要经过很长时间才能获得成功或失败反馈。Stage reward 将反馈提前到各阶段完成时，dense potential reward 则在相邻状态之间提供 progress signal。

  因此：

  早期动作 -> 很久以后才收到成功奖励

  变成：

  早期动作 -> 较近的阶段奖励或 progress reward

  这确实缩短了动作与有效反馈之间的距离，使 critic 更容易学习，也使奖励更容易传播到早期动作。

  不过，对于 potential-based shaping，累计目标原则上仍与原任务等价。它没有改变最终要完成的任务，而是改变了 TD 学习时看到的局部反馈。因此最好称为：

  - feedback-horizon shortening
  - credit-assignment-horizon shortening

  而不是 task-horizon shortening。

  ### 最准确的总体表述

  你们实际上从两个互补方向缩短了“有效学习 horizon”：

  1. Waypoint 缩短策略单次决策需要处理的目标距离。
  2. Dense/staged reward 缩短动作与有效奖励之间的传播距离。

  论文可以直接写成：

  > SHORE-RL does not change the physical episode horizon. Instead, it shortens two complementary effective horizons: the decision-conditioning horizon through nearby latent waypoints, and
  the feedback horizon through dense progress rewards.

  中文可以写成：

  > SHORE-RL 不改变环境的物理时域，而是缩短两个互补的有效时域：latent waypoint 缩短 residual policy 在单次决策中需要解析的目标距离，稠密进度奖励缩短动作与有效反馈之间的信用传播距离。

  因此，我会把原来的审稿意见从：

  > 方法并未真正缩短 horizon。

  修正为：

  > 方法对有效决策时域和信用分配时域提供了合理的隐式缩短，但论文尚未形式化定义或直接测量这种缩短，因此不应将其表述为原始 MDP 或全局优化目标 horizon 的严格缩短。

  换句话说：你们的概念是成立的，主要问题是需要把“缩短了哪一种 horizon”说得更精确，并用实验验证 waypoint 确实承担了局部目标作用，而不只是一个额外的 learned feature。



既然可以改正文、但不能补实验，我建议采用“保留核心故事，但把 horizon 定义改严谨”的方案。目标不是放弃 shortening horizon，而是把它从容易被攻击的“严格 MDP horizon 缩短”，改成“两个互补的
  effective horizon 缩短”。

  有三种路线：

  1. 推荐：保留 SHORE-RL 和 horizon-shortening 主线，但全面改成 effective horizon
      - 优点：保留论文辨识度和核心贡献。
      - 代价：需要系统修改标题、摘要、引言、贡献和结论中的因果措辞。
      - 最适合当前不能补实验的情况。

  2. 更保守：改成 temporal localization + dense progress feedback
      - 科学表述最安全。
      - 但会弱化 SHORE 名称和创新卖点，容易被评价为现有组件组合。

  3. 不推荐：继续坚持 objective horizon 被严格缩短
      - Eq. (3) 仍优化全局累计回报，缺少局部终止和 waypoint reward，很容易继续被审稿人抓住。
      - 仅靠文字或形式推导很难完全说服。

## 推荐修改框架

在无法增加新实验的情况下，建议保留 SHORE-RL 和 horizon-shortening 的核心叙事，但将论文的核心概念统一为 **effective horizon shortening**，并明确区分：

- **Physical task horizon**：环境的 episode 长度，SHORE-RL 不改变它；
- **Global return objective**：TD3 critic 仍估计全局累计回报，SHORE-RL 不替换它；
- **Decision-conditioning horizon**：latent waypoint 缩短 residual policy 在单次决策中需要解析的目标距离；
- **Feedback/credit-assignment horizon**：staged 或 dense reward 缩短动作与有效反馈之间的传播距离；
- **Effective horizon shortening**：论文真正提出、且现有实验能够支持的核心概念。

不建议继续声称方法严格缩短了 MDP horizon 或 Bellman objective horizon。更准确的主张是：SHORE-RL 从决策条件和反馈传播两个互补方向，对长时程 residual learning 进行 temporal localization。

### 1. 修改标题

推荐标题：

> **SHORE-RL: Effective Horizon Shortening for Long-Horizon Residual Reinforcement Learning**

如果不希望改变原标题，至少应在摘要和正文首次解释 “shortening the horizon” 时明确写成：

> shortening the effective learning horizon

不建议继续不加限定地使用 “objective horizon”。

### 2. 在 Introduction 中定义三类 Horizon

建议在 Introduction 首次提出方法时加入：

> We distinguish the physical task horizon from two effective learning horizons. SHORE-RL does not change the episode length or replace the global return objective. Instead, it reduces (i) the decision-conditioning horizon by providing each residual correction with a nearby latent waypoint, and (ii) the feedback horizon by providing transition-level progress rewards.

其核心含义是：

> SHORE-RL 不改变环境的物理时域，也不替换全局回报目标；它通过附近的 latent waypoint 缩短决策条件时域，并通过 transition-level progress reward 缩短反馈传播时域。

这一段应尽早出现，以避免读者将 “horizon shortening” 理解为 episode truncation、option termination 或 Bellman horizon 的严格改变。

### 3. 重写摘要中的机制描述

原先类似：

> reducing the long horizon to a sequence of short ones

建议改成：

> SHORE-RL temporally localizes residual learning along two complementary dimensions: nearby latent waypoints reduce the goal distance represented at each decision, while dense progress rewards reduce the delay between an action and informative critic feedback.

随后增加限定：

> This formulation leaves the physical episode horizon and global return objective unchanged.

修改后的摘要不再声称原始 MDP 被严格拆成多个短 MDP，而是强调两种有效学习时域的缩短。

### 4. 降低 Introduction 中过强的因果措辞

将：

> Figure 2 isolates this horizon dependence.

改为：

> Figure 2 reveals a consistent association between extended temporal composition and late-training collapse across the evaluated tasks.

将：

> This pattern identifies extended temporal composition as the central empirical factor.

改为：

> This pattern motivates extended temporal composition as a plausible organizing factor in our setting, while task difficulty, reward delay, and coordination complexity are not independently controlled.

将：

> each residual correction faces a shorter objective horizon

改为：

> each residual correction is conditioned on a temporally closer target and receives feedback over a shorter effective credit-assignment distance.

这些修改承认 Fig. 2 是跨任务关联性证据，而不是严格控制其他变量后的因果识别。

### 5. 修改贡献列表

第一项不建议继续称为 “mechanism analysis”，因为论文没有直接测量 critic error、Q calibration 或 residual drift。建议改成：

> **An empirical long-horizon failure diagnosis.** We document a consistent late-training collapse pattern across the evaluated long-horizon tasks and contrast it with stable shorter-horizon controls. This observation motivates, rather than causally establishes, effective-horizon shortening.

第二项建议改成：

> **Two-axis effective-horizon shortening.** A learned latent waypoint temporally localizes residual action selection, while an independent progress reward reduces feedback delay for critic learning.

这样可以把论文的概念贡献组织为：

1. 发现并描述长时程 residual finetuning 的后期崩溃现象；
2. 提出决策条件时域与反馈时域两个互补的 temporal localization 机制；
3. 通过跨任务、跨基策略和真实机器人结果验证完整系统的有效性。

### 6. 在 Method 中主动解释 “Horizon Shortening” 的含义

建议在 Method 4.1 前后增加一个独立短段落：

> **Meaning of horizon shortening.** SHORE-RL does not truncate the TD target at the waypoint. The critic continues to estimate the global shaped return. “Horizon shortening” refers to two forms of temporal localization: the navigator represents a demonstrated future feature no more than k steps away, and the reward constructor provides progress feedback at the transition or stage level.

随后补充：

> The waypoint should therefore be interpreted as a learned local control context rather than a new terminal state.

主动说明“不在 waypoint 处截断 TD target”不会削弱方法，反而能够避免读者认为论文混淆了局部条件输入与 Bellman horizon。

Waypoint 机制应解释为：

- waypoint 由距离当前状态不超过 k 步的未来示范特征监督；
- residual policy 无需仅依赖最终任务目标推断完整动作链；
- waypoint 为当前 residual correction 提供时间上更近的控制上下文；
- 这是对 action selection 的隐式 temporal localization，而不是显式 option termination。

Reward 机制应解释为：

- stage reward 将反馈提前到子阶段完成时；
- potential difference 在相邻 transition 上提供 progress feedback；
- 两者缩短的是动作与有效 critic signal 之间的距离；
- potential-based shaping 保留全局任务目标，但改变 TD 学习所见的局部反馈结构。

### 7. 重新解释 Fig. 6 的组件消融

不建议声称 waypoint 在所有任务上都是独立且必要的。建议改为：

> On Pouring, stage shaping and demonstration anchoring largely compensate for removing the waypoint. On PieceAssembly, the substantial gap between the full method and the no-waypoint variant indicates that local waypoint context becomes more important when the task contains object- and contact-specific transitions. Thus, the results support task-dependent complementarity rather than a universal independent waypoint effect.

这一解释与现有数值一致：

- Pouring：完整方法 `.93`，无 waypoint `.92`；
- PieceAssembly：完整方法 `.65`，无 waypoint `.28`。

因此，合理结论不是 “waypoint 在每个任务上都不可缺少”，而是：

> waypoint、demo-BC 和 reward shaping 存在明显的任务依赖互补性；waypoint 在需要复杂对象交互和接触转换的任务上贡献更明显。

### 8. 重写 Conclusion

不建议写：

> temporal localization is key to stable long-horizon RL finetuning

建议改为：

> The results support effective temporal localization as a promising design principle for stable long-horizon residual finetuning.

同时加入限制：

> Our experiments do not independently identify physical episode length as the cause of collapse, nor do they establish that waypoint conditioning mathematically shortens the Bellman horizon. Instead, they show that combining local waypoint context with shorter-delay progress feedback consistently stabilizes the evaluated long-horizon tasks.

建议结论围绕以下逻辑组织：

1. 现有方法在所测长时程任务上表现出后期崩溃；
2. SHORE-RL 从决策条件和反馈传播两个方向进行 temporal localization；
3. 完整方法在所测任务和基策略上获得稳定提升；
4. 这些结果支持 effective horizon shortening 是一种有前景的设计原则；
5. 现有实验尚未证明物理 episode 长度是崩溃的唯一原因，也未声称改变 Bellman horizon。

### 9. 建议全文统一替换的措辞

建议优先使用：

- effective learning horizon
- decision-conditioning horizon
- feedback horizon
- credit-assignment distance
- temporally local target
- local control context
- temporal localization
- consistent with the horizon hypothesis
- motivates rather than causally establishes
- task-dependent complementarity

建议避免或限制使用：

- shorter objective horizon
- isolates horizon dependence
- proves that horizon causes collapse
- decomposes the MDP into short-horizon MDPs
- waypoint is universally necessary
- temporal localization is the key factor

### 修改后的核心主张

建议全文统一到以下主张：

> SHORE-RL does not change the physical episode horizon or replace the global return objective. It shortens two complementary effective learning horizons: nearby latent waypoints temporally localize residual action selection, while dense progress rewards reduce the delay between actions and informative critic feedback. Across the evaluated tasks, this combination substantially improves the stability of long-horizon residual finetuning.

对应的中文表述为：

> SHORE-RL 不改变环境的物理时域，也不替换全局回报目标。它缩短两个互补的有效学习时域：附近的 latent waypoint 对 residual action selection 进行时间局部化，稠密进度奖励则缩短动作与有效 critic 反馈之间的传播距离。在所评估任务上，这一组合显著提高了长时程 residual finetuning 的稳定性。
