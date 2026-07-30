# HiRes-RL 消融实验方案 (ablation spec)

> 目的:为 AAAI 论文的实验/消融部分定一套**能挡审稿人、且算力可控**的消融矩阵。
> 全部开关已在 `resfit/rl_finetuning/chunk_residual/` 代码里核过(见 §2),命令可跑。
> 约定:每条消融**只动一个变量、其余全锁**(seed / action_scale / state_mode / data_source /
> budget / task / **代码库**一律固定),否则结论有混淆。

---

## 0. 前置决策(必须先定,影响 ②和论文 framing)

**已核实的事实:所谓"一值两用"其实是两个独立训练、输入不同的 value 网络。**

| 用途                  | 网络                                                | 输入                 | 训练脚本 / 加载                                                            | flag                  |
| --------------------- | --------------------------------------------------- | -------------------- | -------------------------------------------------------------------------- | --------------------- |
| 出子目标 z(where)     | goal-conditioned**V(s, φ([g,s]))**,10 维瓶颈 | state**+ goal**      | `train_hiql_gc_value.py` / `load_gc_value`                             | `--gc_value_ckpt`   |
| 出势函数 Φ(how well) | 单状态**V(s)**                                | **只有 state** | `train_hiql_value.py` / `hiql_value.py::ValueMLP` → `HiqlPotential` | `--hiql_value_ckpt` |

- `hiql_gc_value.py:3` 文件头自己写了「与单任务 V-as-Φ 的 hiql_value.py **正交**」——作者本就当两个网络。
- 两者**共享**:同一套冻结基座特征(同 `state_mode`/ψ)、同一批离线 demo、同类 HIQL/IQL 训练法。
  **不共享**:网络权重、输入(goal-cond vs 单状态)。所以"自导 / 无特权 / 零奖励工程"成立,唯"**一个** value"不成立。

**投稿前必须三选一(影响 abstract / intro 第4段 / 贡献3 / §一值两用消融的措辞):**

- **A. 改 framing(最省)**:诚实写"两个自导 value,共享同一基座表征与离线数据"。贡献3 降级为"两个信号都自导于基座特征、无特权、无额外奖励工程"。
- **B. 改实现让 claim 变真(最强、最自洽)**:势函数改成 gc_value 的单状态投影 Φ(s)=V(s, g=固定任务目标)。固定外生 goal 恰好=安全性分析里 telescoping 安全的条件,把"一值两用"和"安全准则"缝成一条线。代价:改码 + 重训重评 HiRes。
- **C. 重构贡献3**:主打"两个互补自导信号(where + how-well)",承认两网络。

> 建议:时间紧走 A 先锁投稿;要最强故事走 B。**下面的消融设计对 A/B/C 都成立**,只是 ②的标题措辞随之变。

---

## M. 主实验对比方案（external baselines，与 §1–§9 消融并列）

> §1–§9 全是**方法内部消融**（自己和自己比）。本节是**外部方法对比**——主表
> (Table 2) 真正"和别人比"的那部分。现状:主表严格意义只比了 resfit 一个外部方法,
> DSRL 只挂在 LIBERO;AAAI 会砍"baseline 不足 / comparison only to one prior method"。
> 本节把要补的外部 baseline 定清楚。

### M.1 baseline → 回答的审稿人问题

| baseline                                                   | 回答"审稿人会问"                                    | 类别         | 代码来源 / 可行性                         |
| ---------------------------------------------------------- | --------------------------------------------------- | ------------ | ----------------------------------------- |
| **BC base**(ACT / π0,no RL)                         | RL 到底帮没帮上忙?                                  | 地板         | 已有(每 run step-0 首 eval 点),零成本;取值协议见 M.3 |
| **Residual RL / resfit**(flat,demo-anchored)         | 现有残差方法能做到哪?                               | prior 主对手 | 已在`train_chunk_residual` flat 臂,已跑 |
| **DSRL**                                             | 为什么用动作空间 raw 残差,不用 latent / noise 残差? | prior        | LIBERO 已跑通(`chj/dsrl_pi0`),迁双臂    |
| **IBRL**                                             | 你这套残差凭什么比别的 BC-bootstrapped RL 强?       | prior        | 需接入(见 M.3)                            |
| **全策略 off-policy 微调**(RLPD/IQL不冻基座)⟨可选⟩ | 为什么要 residual 而非直接调整个策略?               | 隔离         | 需搭;预期更快塌,反衬 residual 结构        |
| DPPO / Q-chunking / Cal-QL 等 ⟨只引用⟩                   | BC-RL 微调这一大类你没漏                            | related work | 不跑,引用 + 一句定位                      |

> **shaping / 安全性那条线(消融③)的 baseline 你已经齐了**(none / staged /
> single-state / goal-cond 全在内部消融里),缺的只是"BC 微调 / residual RL 这一大类
> 的横向对手"——就是本节。

### M.2 公平性锁定(所有外部 baseline 与我方臂必须一致)

> 这是 DSRL-on-LIBERO 那条 caveat 的教训:DSRL 用了 10 episodes/eval、base 每 20 步
> requery,和我方 horizon 10/30 不一致 → 结果只能写成 "indicative rather than
> controlled"。双臂上补跑时**必须消掉这个漏洞**。

- **同 base**:双臂统一 ACT;LIBERO 统一 π0。
- **同离线 demo**:锚 / 预训练数据来源、划分一致。
- **同 eval harness**:同 episode 数(50)、同 replan / execution 协议(我方 horizon)、
  同 success 判定。**跑 DSRL/IBRL 时套我方 eval,不要用它们各自的 harness**,否则又变
  不可控对比。
- **同预算 / 种子**:同 env-steps(500k)、同 seed 集(旗舰任务 ≥3)。
- **步长语义对齐**:声明各方法的等价"锚定 / 步长"(我方 action_scale、DSRL 的噪声尺度、
  IBRL 的 exploration σ),避免被说"调参赢"。

### M.3 各 baseline 具体跑法要点

- **BC base(取值协议 · 定稿 2026-07-14,已按代码核实)**:BC base = **冻结基座策略自己的
  成功率**,是一个**常数 / 水平参照线**,不是曲线。取法 = 每个 run 曲线**最左那个 eval 点**
  (env_steps≈0)的 `eval/success_rate`。
  - **为什么 step-0 就是纯 base**(三条代码依据,当前配置全满足):
    ① 残差末层初始化即 0:`actor_last_layer_init_scale=0.0`
       (`resfit/rl_finetuning/config/residual_td3.py:115`,注释 "imp for residual")
       → 初始残差 μ≈0,再乘 action_scale → 执行动作 = base + ≈0 ≈ 纯 base;
    ② eval 确定性、无探索噪声:`agent.act(obs, eval_mode=True, stddev=0.0)`
       (`resfit/rl_finetuning/utils/evaluate_dexmg.py:199`)→ base 不被探索噪声拉低;
    ③ 首个 eval 发生在任何 actor 更新之前:`next_eval=0`
       (`resfit/rl_finetuning/chunk_residual/train_chunk_residual.py:1178`),actor 更新被
       `env_steps >= learning_starts`(几千步)挡着 → 曲线最左点时 actor 仍停在初始化(残差=0)。
  - **样本量 / 口径**:每次 eval 跑 **50 局**(`--eval_num_episodes` 默认 50,
    `train_chunk_residual.py:601`);success 判据 `reward==1.0` 在 episode 末
    (`evaluate_dexmg.py:245`),与 RL 各评估点**完全同 env、同协议、同判定**。
  - **跨臂 pool 拿 seed**:step-0 时残差恒 0,和该 run 开没开 subgoal/potential/BC **无关**,
    所以同一任务下**所有臂**的 step-0 点都≈同一个 base → 把该任务**全部臂**的首 eval 点一起
    平均,白捡 seed、误差棒更紧。报 `mean±s.e.`。
    (**顺带自检**:同一任务所有臂曲线的最左点应≈重合成一条线;若某臂明显偏低 = 那条 run 有问题。)
  - **画法**:Fig.2 每个任务子图画成一条**水平地板线**(非曲线),强调"所有 RL 臂都要跨过它"。
  - **不要做**:① 单独训 BC 再每 10k 评、画波动曲线——base 是冻结的、没有"训练曲线",纯属重复劳动;
    ② 用 `resfit/rl_finetuning/scripts/eval_pi05_base.py` 取数——它只是跨进程 smoke、
    **不聚合成功率**(自己注释写 "not success rate")。
  - **仅当**要比 "50 局 × N seed" 更紧的误差棒时,才值得单独写 base-eval 脚本;但必须**逐字复刻**
    `eval_env` 的 env / 相机 / max_steps 配置,否则又引入协议错位——这本身就是"优先用 step-0"的理由。
  - **别混**:BC base(step-0,残差恒 0,纯基座) ≠ flat residual(残差已训练、只关
    subgoal/BC/potential,是另一条 run 的后段,§9 #2)。Table 2 里是两行。
- **Residual RL / resfit(flat)**:= 内部消融的 `flat, bc0.1` 臂(§9 #2);它同时**也是
  外部 prior 方法的代表**(published recipe + demo 锚)。注意区分:naive resfit(bc0,塌)
  = §9 #1 与 Table 1,别混。
- **DSRL(双臂)**:复用 `chj/dsrl_pi0` 那套 noise-space RL,env 从 LIBERO 换成双臂
  DexMimicGen 任务(Pouring 优先),base 换成同一 ACT/π0;**eval 套我方 harness**
  (见 M.2)。最高性价比的"第二个外部方法",且正好和我方 raw 残差形成 latent-vs-raw
  对照(呼应 intro"attack the horizon")。
- **IBRL(接入)**:IBRL = RLPD 底座 + BC-proposal bootstrapping——每步对 {a_BC, a_RL} 用
  Q 集成取 argmax(**acting 与 TD target 两处都要**)。我方已有 RLPD/TD3 + demo replay,
  两条路:
  - (a) 直接用官方 IBRL repo,换我方 base / task / eval;
  - (b) 在现有底座加一个**全动作** actor(非残差头)+ BC-proposal 选择逻辑。
  - 公平性:BC policy 用同一 base,Q 集成规模 / UTD / demo 比例与我方一致。
  - **关键对照点**:IBRL 是**软锚**(Q 一旦高估就丢弃 BC、用被高估的全动作),我方是**硬锚**
    (小 action_scale 结构上走不远)。若 IBRL 在长程也塌而我方不塌,"锚定结构带来稳定"
    就立住;若 IBRL 也稳,得承认稳定不只来自残差结构。两种结果都有信息量。
- **全策略 RLPD 微调(可选隔离)**:同 RLPD 底座,actor 输出整动作、从 base 初始化直接微调,
  不加 residual / subgoal / shaping。至少 Pouring 跑 1 条,预期更快塌 → 隔离出"稳定来自
  residual 结构本身"。

### M.4 Table 2 骨架(已落地)

主表已重排成**每任务两组**:上组 prior methods(flat residual / DSRL / IBRL,后两者
`\todo{run}` 占位),`\cmidrule` 隔开,下组 HiRes-RL 及消融;BC 地板 = 每任务 step-0 首 eval 点
(取值协议见 M.3),在主表列为**地板行**、在 Fig.2 画成**水平线**。
DSRL / IBRL 行等本节跑完回填;IBRL 的 bib(`hu2023imitation`)与 Related Work 一句定位
已加。

> 若只在 Pouring 跑外部 baseline、其余任务不跑,记得删掉用不到的 `\todo` 行(别占坑不填)。

### M.5 算力预算(增量,叠加在 §10 之上)

- **DSRL**:Pouring 3 seed + 其余 3 长程任务各 1 seed(佐证)。→ ~6 run。
- **IBRL**:同上量级;算力紧则**至少 Pouring 3 seed**。
- **全策略 RLPD**(可选):Pouring 1 条(信息量在"塌"本身)。
- **只够加一个** → 选 **DSRL 双臂**(复用 harness + 叙事最贴)。**能加两个** → + IBRL。

---

## 1. 每条消融对应的 claim

| 编号 | 消融                                                    | 支撑的贡献 / 挡的质疑                                   | 优先级     |
| ---- | ------------------------------------------------------- | ------------------------------------------------------- | ---------- |
| ①   | bc 锚 × subgoal(2×2)                                  | 防塌方到底是 bc 锚还是子目标?**最可能被拒的混淆** | ★必做     |
| ②   | subgoal × potential(2×2)                              | 两种用途各自够不够、是否互补(贡献3)                     | ★必做     |
| ③   | 单状态 V(s) vs 目标条件 V(s,z) 当势函数 + 漂移图        | value-as-potential 安全准则(贡献4,最锋利 novelty)       | ★必做     |
| ④   | 表征 state_mode:eef_piece(特权) vs act_feat vs pi0_feat | 无特权→sim-to-real(贡献2)                              | ★必做     |
| ⑤   | joint 联合微调 on/off                                   | 拆开 Table 2 "staged+joint" 的双变量混淆                | ☆强烈建议 |
| ⑥   | action_scale 扫 {0.1,0.2,0.4}                           | 锚定强度稳健性                                          | ○加分     |
| ⑦   | subgoal 视野 / base 类型 / renorm                       | 灵敏度、泛化                                            | ○可选     |

---

## 2. 已核实的开关(file:line)

`resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`:

- `--subgoal_conditioned`(605):注入潜在子目标 z(where)
- `--reward_shaping {none,staged,potential}`(644)
- `--potential_source {stage,hiql,hiql_subgoal}`(650):`hiql`=单状态 V(s)*scale(安全);`hiql_subgoal`=V(s,z)*scale(不安全,A2)
- `--gc_value_ckpt`(607):子目标用的 goal-cond value(`--subgoal_conditioned` 需)
- `--high_actor_ckpt`(608):在线提 z 的 high_actor
- `--hiql_value_ckpt`(653):势函数用的**单状态** value.pt(`--potential_source hiql` 需)
- `--demo_bc_coef`(629):**BC 锚系数**(= "bc01" 的 0.1;需 `--offline_fraction>0` 供 demo)
- `--bc_coef_final`(632):bc 衰减末值(默认 None=不衰减)
- `--offline_fraction`(687):离线 demo 混采比例(锚需 >0)
- `--action_scale`(582,默认 0.2):残差锚定强度
- `--online_finetune_value` / `--online_finetune_high_actor`(708/710):Phase3 联合微调(joint)
- `--renorm_subgoal`(612,默认 True)、`--subgoal_way_steps`(609,默认 25)
- `--base_policy_type {act,pi05}`(664)、`--task`(565)、`--dataset`(580)、`--env_family`(567)

离线 value 训练:

- `train_hiql_gc_value.py --state_mode {eef_piece,act_feat,pi0_feat}`(:77)→ 产 `gc_value.pt`
- `train_hiql_high_actor.py --state_mode {…}`(:58)→ 产 `high_actor.pt`
- `train_hiql_value.py --state_mode {eef,eef_piece,act_feat}`(:310)→ 产**单状态** `value.pt`

---

## 3. 消融① — bc 锚 × subgoal(2×2,去混淆)

**动机**:vanilla 基线是 `demo_bc_coef 0`,你的方法臂是 `0.1`;直接比就同时动了 bc 和 subgoal,审稿人第一刀。用 2×2 一次隔离 **bc 主效应 / subgoal 主效应 / 交互**,且四格**全在 `train_chunk_residual` 内跑**(消掉"原版↔框架"的 codebase 混淆)。

|                 | subgoal 关  | subgoal 开     |
| --------------- | ----------- | -------------- |
| **bc=0**  | flat, bc0   | subgoal, bc0   |
| **bc0.1** | flat, bc0.1 | subgoal, bc0.1 |

- **读法**:左列上→下 = bc 单独作用;每行左→右 = subgoal 单独作用;两者差异 = 交互(**很可能塌得越狠的任务 subgoal 越不可替代**)。
- **bc 两边都锁**:因为要隔离 subgoal 就必须固定 bc;锁在 0.1(非 0)是因为那是真实方法档位。bc="0.1"若含衰减,则四格 `--bc_coef_final` 整条曲线锁一致。
- **关键 flag 差异**:
  - flat:不加 `--subgoal_conditioned`,`--reward_shaping none`
  - subgoal:`--subgoal_conditioned --gc_value_ckpt … --high_actor_ckpt …`,`--reward_shaping none`
  - bc0 / bc0.1:`--demo_bc_coef 0.0` / `0.1`(均 `--offline_fraction 0.5`)
- **顺带验一个要命的问题**:框架内 `flat+bc0` 是否也塌?
  - 若也塌 → Table 1 塌方是 bc=0 的锅,故事干净;
  - 若不塌 → 塌方部分来自原版实现差异,**宁可自己先知道**。
- **已知信号**:Table 2 里 Pouring 的 `flat residual` ss≈.76(vs vanilla bc0 的 .07)。若那行确是 bc0.1,说明**Pouring 上防塌大头是 bc 锚**,主 claim 得诚实调成"bc 锚防崩溃、subgoal/势函数进一步提稳态与峰值"。**需先核实该行出处**。

---

## 4. 消融② — subgoal × potential(2×2,两种用途拆解)

**动机**:证明 value 的两种用途(出子目标 / 出势函数)各自有效且互补(贡献3)。

| 臂             | 子目标 z     | 势函数 V(s)  | flag                                                                                                 |
| -------------- | ------------ | ------------ | ---------------------------------------------------------------------------------------------------- |
| flat(都关)     | 关           | 关           | `--reward_shaping none`(不加 subgoal)                                                              |
| subgoal-only   | **开** | 关           | `--subgoal_conditioned … --reward_shaping none`                                                   |
| potential-only | 关           | **开** | `--reward_shaping potential --potential_source hiql --hiql_value_ckpt …`(不加 subgoal)            |
| HiRes(都开)    | **开** | **开** | `--subgoal_conditioned … --reward_shaping potential --potential_source hiql --hiql_value_ckpt …` |

- **读法**:flat→subgoal-only 隔离"去哪";flat→potential-only 隔离"做得好不好";两单用→HiRes 看互补/超加性。
- **可能结论(决定 claim)**:subgoal-only≈HiRes 且 potential-only 弱 → 整形冗余,诚实降级;两单用皆部分有效、HiRes 明显更好 → 互补成立,贡献3 最硬。
- **caveat(§0)**:此处 subgoal 用 gc_value、potential 用单状态 value,是两网络;走 Option B 后才真是"一网络两用"。

---

## 5. 消融③ — value-as-potential 安全性(最锋利 novelty)

**对照**(两臂都 `--subgoal_conditioned`,只差 `--potential_source`):

- **安全 = HiRes**:`--reward_shaping potential --potential_source hiql --hiql_value_ckpt …`(单状态 V(s),telescoping)
- **不安全**:`--reward_shaping potential --potential_source hiql_subgoal --gc_value_ckpt …`(V(s,z),z 内生 → farming)

  - 依赖(代码断言):`hiql_subgoal` 需同时 `--subgoal_conditioned` + `renorm_subgoal=True` + `--gc_value_ckpt`。
- **已有证据**:Pouring 稳态 0.84→0.06(unsafe 臂),已足够立"塌方"。
- **补 Fig.2 漂移图(几乎白捡,离线算)**:从已存 checkpoint 画 V(s,z_t) 随 t 的**漂移** vs 单状态 V(s_t) 的**telescoping**,直观展示 farming 机制。**不用重训**,只需前向已存 value/gc_value。

---

## 6. 消融④ — 无特权表征(sim-to-real 命根)

**对照**:同一任务,`state_mode` 三选一,重训对应 value 后跑同配置 HiRes(或 subgoal-only):

- `eef_piece`(**特权** eef+物体 rel):上界参照
- `act_feat`(冻结 ACT encoder 特征):免特权
- `pi0_feat`(冻结 π0 prefix 特征):免特权
- **代价**:每个 state_mode 要**重训 `gc_value` + `high_actor`(+ 单状态 `value`)**,再跑主训。
- **claim**:扔掉特权状态不掉点或掉很少 → 支撑贡献2 + 真机迁移。
- 建议选一个有 stage 结构的任务(ThreePiece)做,顺带覆盖 stage 相关路径。

---

## 7. 消融⑤ — joint 联合微调(拆 staged+joint 混淆)

**动机**:Table 2 的 "staged+joint" 行**同时动了 staged shaping 和 joint 在线微调两个变量**,无法归因。
**对照**(其余全锁):

- HiRes,**冻结**离线 value/high_actor(不加 joint flag)
- HiRes,**+joint**:`--online_finetune_value --online_finetune_high_actor`
- 只在有 stage 结构的任务(ThreePiece / Threading)上做才和 staged 行可比。
- joint 与 potential/staged 正交:此处固定 `--potential_source hiql`,只切 joint on/off。

---

## 8. Tier-2 可选

- **⑥ action_scale 剂量-反应**:flat 那一行扫 `--action_scale {0.1,0.2,0.4}`,看"锚越紧越不塌但可能越不涨"。
- **⑦a subgoal 视野**:`--subgoal_way_steps {15,25,40}`(注意 high_actor 的 `--way_steps` 与主训 `--subgoal_way_steps` 两处须一致)。
- **⑦b base 泛化**:`--base_policy_type act` vs `pi05`(支撑"ACT 和 π0 两种基座")。
- **⑦c renorm**:`--renorm_subgoal` on/off(已知非命门,做成"负结果"稳健性)。
- OAC 探索:与本文 claim 无关,**这篇不放**。

---

## 9. 共享臂总表(一个任务上,①②③ 只需 7 臂)

**别把 ①②③ 当三批独立实验跑**;union 如下(全 `--offline_fraction 0.5`,bc0.1=`--demo_bc_coef 0.1`):

| # | 臂                      | subgoal | potential    | bc  | 覆盖      |
| - | ----------------------- | ------- | ------------ | --- | --------- |
| 1 | flat, bc0               | 关      | none         | 0   | ①        |
| 2 | flat, bc0.1             | 关      | none         | 0.1 | ①②      |
| 3 | subgoal-only, bc0       | 开      | none         | 0   | ①        |
| 4 | subgoal-only, bc0.1     | 开      | none         | 0.1 | ①②      |
| 5 | potential-only, bc0.1   | 关      | hiql         | 0.1 | ②        |
| 6 | **HiRes**, bc0.1  | 开      | hiql         | 0.1 | ②③-安全 |
| 7 | **unsafe**, bc0.1 | 开      | hiql_subgoal | 0.1 | ③-不安全 |

---

## 10. 任务范围 & 算力预算

- **完整 7 臂矩阵只挑 1 个代表任务** = **Pouring**(塌方最干净 0.84→0.06;`pouring_ablation_matrix_0706` 已在跑部分减法臂,别重起)。理想 3 seed → ~21 run(含在跑的)。
- **其余 3 个长程任务**(ThreePiece / Threading / LiftTray):Fig.1 已有 vanilla / subgoal-only / HiRes / [staged] / unsafe;**只需补 `flat+bc0.1` 与 `potential-only(hiql)` 两格**做跨任务佐证。1–2 seed。→ 约 +6 run × (1–2 seed)。
- **短程任务(CanSort 等)不参与本组消融**(任何格都不塌 = 零信息);它们在 Fig.1 当"塌方长程特有"的对照即可。
- **④表征**:选 1 任务 × 3 state_mode,含重训 value/high_actor。
- **⑤joint**:1 stage 任务 × 2 臂。
- **总量级**:约 **1 深(Pouring 7 臂×3seed)+ 3 任务各补 2 格 + ④/⑤ 少量**,而非 4 任务 × 全格 × 3seed 的全矩阵。

---

## 11. 报告规范

- 指标沿用主表:**base / peak / ss(末 20% 均值) / fin**;主 claim 看 **ss/fin**(稳态)非 peak。
- 主结果与①③核心臂:**3 seed + mean±s.e.**;消融格子可先 1–2 seed 铺满、核心再补。
- 交叉任务表:每任务给 `flat+bc0.1` vs `subgoal-only` vs `HiRes` vs `unsafe` 的 ss,凸显"越难越离不开 subgoal"。

---

## 12. 待办 / 依赖

- [ ] **定 §0 的 A/B/C 方向**(决定 ②标题与 abstract/intro/贡献3 措辞;B 需改码 spec)。
- [ ] 核实 Table 2 中 Pouring `flat residual` 行的 bc 档位与 run 出处(决定①主 claim 写法)。
- [ ] 每任务准备好 `gc_value.pt` / `high_actor.pt` / 单状态 `value.pt`(④还要按 state_mode 重训)。
- [ ] Fig.2 漂移图:从已存 checkpoint 离线出图(无需重训)。
- [ ] **外部 baseline(§M)**:DSRL 双臂(Pouring 3 seed + 3 长程任务各 1 seed,eval 套我方 harness)、IBRL 接入(官方 repo 或底座加 BC-proposal,对齐 eval)、(可选)全策略 RLPD;跑完回填 Table 2 的 DSRL / IBRL 行。
- [ ] (只引用)Related Work 补 DPPO / Q-chunking 等一句话定位,补全 BC-RL 微调这一类。
- [ ] GPU 授权后按 §9/§10 起批;offcache 复用规则见项目记忆(action_scale 进签名→变则重建)。

---

*flags 核验日期与代码位置见 §2;若代码更新,重新 grep `add_argument` 校准。*
