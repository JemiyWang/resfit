# chunk_residual — Handoff (2026-06-02 PM, staged 稠密奖励 + A/B + per-stage 诊断 → 根因锁定)

> 接续 `HANDOFF_2026-06-02.md`(stage 工具链 + ThreePiece 负结果)的 §7 下一步 item 1+2。
> 本轮:把 stage2→3 拆细 + staged 稠密奖励做完(TDD),跑了 staged / 降幅 / potential 的 A/B,
> 并用新写的 **per-stage 到达率仪表**把崩溃根因一锤定音。

---

## 0. 一句话现状
staged 稠密奖励(+ potential-based shaping,后者由**并行线**实现)**成功把 critic 从恒 0 救活**
(target_q 长出按 stage 结构),但 **ThreePiece chunk-RL 仍早期塌**——staged / staged-降幅 /
potential **三种配置全塌**。per-stage 诊断证明:**残差选择性摧毁最终精插入(stage3→4),早期
stage 保留** → 根因 = **残差漂离 identity + chunk-20 开环放大成基座破坏**,**与奖励信号无关**
(potential 这个"不可刷分"的对照也塌,排除了 reward-hacking)。下一步 = **advantage-gating /
identity 锚定**(需与并行线协调文件归属)。

---

## 1. 本轮新增/改动

### 本 thread 做的(stage 拆细 + staged 奖励 + per-stage 仪表)
- `stage_detectors.py` — threepiece 检测器 **4→5 段**:0 起步 / 1 piece1抓 / 2 piece1放好释放 /
  **3 piece2抓起** / 4 成功(用 `env.piece_2`,属性名经真环境探针确认;`NUM_STAGES=5`)。
- `chunk_env_wrapper.py` — `staged_bonus(start,end,bonus)=bonus·max(0,end−start)` 纯函数 +
  构造参 `stage_reward_bonus`;`step` 内 add 前折进 reward(自动进 n-step),不设 done。
- `train_chunk_residual.py` — `--staged_reward` / `--stage_reward_bonus`(默认1.0);
  训练 env 接 bonus,**eval env 恒 bonus=0(不污染指标)**。
- `stage_reach.py` ★新 — 纯函数 `stage_reach_rates(episode_max_stages, num_stages)`:每 stage k 的
  到达率(到过 stage≥k 的 episode 占比)。TDD 5 测(`tests/test_stage_reach.py`)。
- `eval_stage_reach.py` ★新 — per-stage 到达率诊断:base + 受控 L2 范数随机残差(num_envs=1
  逐 episode 记最高 stage),看残差幅度如何侵蚀各 stage 到达率。
- `verify_stage5_instant.py` ★新(早些)— 真环境探针确认 5 段 + env.piece_2。
- 测试:`test_staged_reward.py`★新、`test_stage_reach.py`★新、`test_stage_detector.py` 扩。
- spec/plan:`docs/superpowers/specs/2026-06-02-threepiece-staged-reward-design.md`、
  `docs/superpowers/plans/2026-06-02-threepiece-staged-reward.md`。

### 并行线做的(⚠️ 不是本 thread,但已进同一批文件)
- **PBS(potential-based shaping)**:`--reward_shaping {none,staged,potential}`;纯函数
  `shaping_reward(start,end,*,mode,bonus,gamma,done)`;potential=`bonus·(γ·Φ'−Φ)`,Φ=stage,
  **done 时 Φ'=0**(Ng1999,不改最优策略);别名 `--staged_reward` 经 `resolve_shaping_mode` 解析。
- `--output_dir`(默认 outputs_chunk;**并行 run 必须给不同目录,否则抢同一 best.pt**)。
- 全套 **77 passed**(并行 72 + 本线 stage_reach 5)。

---

## 2. 实验与结论(本轮核心产出)

### A/B/诊断(均 ThreePiece,raw actor,stage_balanced,seed 0,base=本地 `policy_step_199999`)
| run | 配置 | eval success 轨迹 | 结局 | 日志 |
|---|---|---|---|---|
| #1 staged 控制 | bonus=1.0, as=0.2 | 0.24→0.32→0→0.02→0.04→0→0→0→0→0.02(到90k) | 塌 | `outputs_chunk/threepiece_staged_balanced.log` |
| #2 A 降幅 | bonus=0.2, as=0.1 | 0.44→0.30→0.08→0.14→0.04→0.06(到50k) | 软塌 | `outputs_chunk/threepiece_staged_b0.2_as0.1.log` |
| potential | mode=potential, bonus=1.0 | 0.26→0.18→0→0→0(到40k+) | 塌 | `outputs_chunk/threepiece_potential_balanced.log`(#4,GPU6,**仍在跑**) |
| raw(今早老 run) | 无 shaping | 0.30→0→…→0(满300k) | 塌 | `outputs_chunk/threepiece_raw_balanced.log` |

### 三条结论
1. **critic 醒了**:staged 与 PBS 都把 `target_q` 从 ≈0 救活。staged 反向(stage0 最高~2,净加
   bonus 可收);potential 呈 −bonus·Φ 线性下移(stage0~+0.19 / stage3~−2.81),还原 Q_orig≈平的
   ~0.2 = 真实成功率 → **证实 PBS 实现正确**。
2. **全塌 + potential 对照** → **reward-hacking 排除**(potential 不可刷分也塌)、**shaping 种类无关**、
   **降 bonus/action_scale 无效**(只软化不阻止)。
3. **per-stage 诊断(`eval_stage_reach`,norm 扫描,n=10/norm)**:
   ```
   resid_norm | reach1 reach2 reach3 success(reach4)
      0.00    |  1.00   0.90   0.80   0.30      ← 基座
      0.50    |  0.90   0.90   0.70   0.20
      1.00    |  0.70   0.70   0.50   0.00      ← 训练崩溃残差幅度
      1.50    |  0.80   0.50   0.40   0.10
   ```
   → **success 选择性崩到 0、reach1-3 保留**(=假设 a):残差毁的是**最终精插入 stage3→4**,不是
   早期粗动作。连 norm0.5 都把 success 0.3→0.2 → **残差在本任务几乎只帮倒忙**。
   caveat:n=10 噪声 ±~0.15,定性结论稳(精确数字可 n=25 重跑)。

---

## 3. 根因 & 下一步

**根因锁定**:残差漂离 identity(residual_norm 0→~1.5)+ **chunk-20 开环**把残差误差放大成
基座破坏;尤其精插入是闭环动作,开环 chunk 中途无法修正。与奖励信号无关。

**两个正交旋钮**:
- **一阶(现在为什么崩)**:残差幅度/方向 → **advantage-gating / identity 锚定**。← 先做这个,
  诊断直接指向它,且不推翻 chunk 假设。
- **二阶(好残差的天花板)**:重决策频率/闭环 → 缩短 chunk / receding-horizon 重规划 /(极端)
  per-step。注意:per-step = 原版单步残差,**已栽过**(稀疏奖励 critic≡0);现在 staged 或能救,
  但未验证,且放弃 chunk-flow 假设。

**优先级**:
1. **advantage-gating / identity 锚定**(改 QAgent/actor/train loop,TDD)。
   **⚠️ 与并行线协调文件归属**——他们在改 `chunk_env_wrapper.py` / `train_chunk_residual.py`
   /(可能)QAgent,直接动会撞车。
2. gating 止住破坏但 success 仍封顶 → 质疑 chunk 粒度(缩短 / 重规划)。
3. goal-conditioned HER(4a)仍 deprioritized(没奖励学不动已不是瓶颈,critic 已醒)。

---

## 4. 运行态 & 踩坑(必读)
- **已杀**:#1(GPU3 staged)、#2(GPU1 staged-降幅)、#3(GPU6 老 potential,无 output_dir,
  重复且污染 best.pt)。**活着**:#4 potential(GPU6,pid 1580385,output_dir=outputs_chunk/potential,
  stdout=`threepiece_potential_balanced.log`,也在塌)。
- **GPU**:1/2/3/7 空;0/4/5 = 单步 paper run(`train_residual_td3`,dexmg-*-final,**非 chunk 对比**)。
- **删了** 坏的 `outputs_chunk/best.pt`(497MB,#1/#2/#3 三方并发写交错;能 torch.load 但 provenance
  乱、是已塌 run 策略、无价值)。干净 ckpt:`outputs_chunk/potential/best.pt`(256MB,#4)。
- ⚠️ **best.pt 路径硬编码** → 并发 chunk run **必须各给 `--output_dir`**,否则抢写 best.pt 成垃圾。
- ⚠️ **日志缓冲坑**:`conda run` 默认捕获+缓冲子进程 stdout → 长跑半盲(日志 0 字节≠卡死)。
  修:`conda run --no-capture-output … python -u …` + `PYTHONUNBUFFERED=1`。
- **log 是否被多写者交错**:每条 run 启动只打一次 `ActionScaler initialized`,`grep -c` >1 即交错。
  本轮 4 个 log 都=1(干净)。查进程真实 stdout 用 `/proc/<pid>/fd/1`(**别猜**)。
- **detached 长跑**:`setsid nohup … &` 时 `echo $!` 给的是启动器 PID(会先退),哨兵要盯
  **真 python PID**(`pgrep -af` 里的 `python -u …`),否则误报 DIED。

---

## 5. 测试 & git
- 单测:`conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -v`(**77 passed**)。
- **全部改动未提交**,分支 `chunk-residual-validation`;working tree 混本轮 + 并行 PBS + 上几轮
  stage 工具链的未提交内容(用户自己管 commit)。
- 项目记忆 `project_dexmg_stage_detector` 已更新(含 PBS / A/B / per-stage 状态)。
