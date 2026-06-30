# OAC 乐观探索接入 resfit — 设计 spec

> 日期: 2026-06-30
> 状态: 已与用户确认设计, 待写实现 plan
> 关联: relative_work/structured_exploration_noise_survey.md(流派 4)、
>       relative_work/OAC_explained.md、参考实现 /mnt/mnt/data/wjm/residual/oac-explore

## 1. 背景与目标

当前残差 RL 的探索是"对称、无方向的高斯白噪声"(`actor.py` 的 `TruncatedNormal(scaled_mu, std)`,
在 `act() -> _act_default() -> dist.sample()` 处采样)。OAC(Ciosek et al., NeurIPS 2019)
指出对称探索有两个病:悲观欠探索、方向无信息。OAC 的修法是**只改采样**:把探索分布的均值,
沿"Q 乐观上界 Q_UB 的梯度方向"偏移一步(协方差不变),朝"可能更好、还没怎么探过"的方向多撒噪声。

目标: 把 OAC 以**开关形式**接进 resfit 的 chunk 残差 RL(默认关、关时与现状逐位等价、零回归),
作为一个可 A/B 的探索改进。**只动探索采样路径,不动 critic/actor 的学习更新。**

## 2. 范围

In:
- `QAgent.act()` 增加 OAC 探索分支(新方法 `_act_oac`)。
- 3 个配置项 + train_chunk_residual 的 CLI 透传。
- 单元测试 + CLI wiring 测试 + CPU 冒烟(不启真训练)。

Out(YAGNI, 明确不做):
- 不动 actor 更新的下界(不做 OAC 的 β_LB 下界泛化)。
- 不做 DICE 之类 off-policy 分布校正。
- 不写 A/B run 脚本(任务与启动时机由用户定;现有 6 个实验在跑)。
- 不专门在 LIBERO / 单步 `train_residual_td3` 路验证(但代码在 QAgent 层, 对它们天然生效)。

## 3. 已确认的设计决策

- **flag 形式**: 显式开关 `--oac_explore`(store_true)+ 两超参 `--oac_beta_ub`、`--oac_delta`。
- **交付范围**: 代码 + flag + 单测/冒烟。
- **偏移在动作空间做**: resfit 的 `TruncatedNormal` 直接在(scaled)动作空间, 没有 SAC 那个
  pre-tanh 层, 所以偏移直接加在残差均值上, 比参考实现(在 pre-tanh 空间偏移)更直接。
- **梯度取在 critic 真正看到的动作上**: `residual_actor=True` 时评估 `clamp(base_action + 残差均值, -1, 1)`,
  否则评估残差均值——镜像 `update_critic`(q_agent.py:359-364), 两种模式都正确。
- **不确定性 σ_Q**: 用 critic ensemble 的均值/标准差(`mu_Q = mean_k Q_k`、`sigma_Q = std_k Q_k`,
  std 用总体口径 unbiased=False)。本仓 chunk 路 `num_q=10`(ensemble, 见 CriticConfig 默认),
  此式对任意头数成立, 且 K=2 时退化为 OAC 原文的 `|Q1 - Q2| / 2`。
  (修正: spec 初稿误写 num_q=2; 实测 chunk 路用 ResidualTD3 配置, 继承 num_q=10。)
- **只在训练 rollout 探索时生效**: eval(`dist.mean`)与 critic target 路径不经过 OAC。
- **默认值保守**: `oac_explore=False`、`oac_beta_ub=4.0`、`oac_delta=0.5`。

## 4. 算法(动作空间 OAC, 已适配本仓)

数据流(探索一步):

```
obs -> _encode -> feat
   -> actor.forward(obs, std) -> TruncatedNormal(scaled_mu, std)   # 取 loc=残差均值, scale=std
   -> 用 critic 在"真实条件动作"上算 Q_UB, 对残差均值求梯度
   -> 均值沿梯度偏移 mu_C -> mu_E
   -> TruncatedNormal(mu_E, std).sample() -> 残差动作(返回给 env wrapper, 接口与现状一致)
```

`_act_oac()` 核心(伪码):

```python
dist  = self.actor.forward(obs, stddev)                  # 复用 actor, 自动含 stage_budget 缩放
mu_T  = dist.loc.detach().clone().requires_grad_(True)   # 残差均值 [B, A]
std_T = dist.scale.detach()                              # [B, A](捕获 stage_budget 对 std 的缩放)
with torch.enable_grad():
    crit_act = (torch.clamp(obs["observation.base_action"] + mu_T, -1.0, 1.0)
                if self.residual_actor else mu_T)        # 镜像 update_critic
    q = self.critic(obs["feat"], self._critic_prop(obs), crit_act).squeeze(-1)  # [K, B]
    mu_Q    = q.mean(0)                                  # [B]  ensemble 均值
    sigma_Q = q.std(0, unbiased=False)                  # [B]  ensemble 标准差(K=2 即 |Q1-Q2|/2)
    q_ub    = (mu_Q + self.oac_beta_ub * sigma_Q).sum()
grad  = torch.autograd.grad(q_ub, mu_T)[0]              # 只对 mu_T 求导, 不污染 critic.params.grad
Sigma = std_T ** 2
denom = torch.sqrt((grad ** 2 * Sigma).sum(-1, keepdim=True)) + 1e-6
mu_C  = math.sqrt(2.0 * self.oac_delta) * (Sigma * grad) / denom
mu_E  = (mu_T + mu_C).detach()
return utils.TruncatedNormal(mu_E, std_T).sample(clip=None)
```

要点:
1. 偏移在动作空间(无 pre-tanh)。
2. 梯度取在 critic 真正看到的 combined+clamp 动作上(clamp 在饱和处梯度为 0, 符合"到边界不再推")。
3. per-env 行独立: `denom` 在动作维上求和、按行(B)归一, B>1 安全。
4. `torch.enable_grad()` 局部包住, 防 rollout 的 no_grad 上下文挡住反传。
5. `autograd.grad(inputs=[mu_T])` 只回 mu_T 的梯度, 不在 critic 参数上累积 `.grad`。
6. act() 期间 agent 处于非训练态(act 断言 `not self.training`), critic 的 dropout/LN 在 eval 态 -> 偏移确定。

`act()` 集成:

```python
obs["feat"] = self._encode(obs, augment=False)
if self.oac_explore and not eval_mode:
    action = self._act_oac(obs, stddev)
else:
    action = self._act_default(obs=obs, eval_mode=eval_mode, stddev=stddev, clip=None, use_target=False)
```

`oac_explore=False` 时分支不进, 完全走原路径。

## 5. 改动面与接口

- `resfit/rl_finetuning/off_policy/rl/q_agent.py`
  - `__init__`: 从 cfg 读 `oac_explore / oac_beta_ub / oac_delta` 存为属性(或直接用 self.cfg.*)。
  - 新增 `_act_oac(self, obs, stddev) -> Tensor`。
  - `act()`: 加 OAC 分支(仅非 eval)。
  - 顶部 import `math`(若未引)。
- `resfit/rl_finetuning/config/rlpd.py`(QAgentConfig 对应 dataclass)
  - 加字段: `oac_explore: bool = False`、`oac_beta_ub: float = 4.0`、`oac_delta: float = 0.5`。
    默认值保证向后兼容(老配置不传 = 关)。
- `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`
  - argparse 加 `--oac_explore`(store_true)、`--oac_beta_ub`(float)、`--oac_delta`(float),
    构造 QAgent cfg 时透传。
- 新增 `resfit/rl_finetuning/chunk_residual/tests/test_oac_explore.py`。

接口不变: `act()` 签名不变, 返回的仍是残差动作; env wrapper / buffer / 学习侧均不感知 OAC。

## 6. 正确性与零回归

- `oac_explore=False` -> `act()` 完全走原 `_act_default`(单测: 同 seed 逐位一致)。
- eval(`dist.mean`)与 critic target(`update_critic` 内的 `_act_default(use_target=True)`)不经过 OAC。
- `residual_actor` True / False 两条路都覆盖。
- `oac_delta=0` -> `mu_C=0` -> 均值不偏移(方向项关掉, 退化为基础分布)。

## 7. 测试策略(全 CPU, 不启真训练)

1. **零回归**: `oac_explore=False`, 固定 RNG, `act()` 输出 == 改前 `_act_default`。
2. **偏移方向**: mock 一个 Q 对某动作维单调的 critic, 断言 `mu_E` 沿 Q 上界梯度方向移动。
   注意: 归一化偏移在 Σ⁻¹-范数下恒为 √(2δ), **与 `oac_beta_ub` 无关**——β_UB 只改变梯度方向
   (σ_Q 项如何加权各动作维), 不改变偏移大小; 偏移大小由 δ 与 std 决定。故**不**写
   "β_UB 越大偏移越大"的断言(会失败)。(终审纠正: spec 初稿此处措辞有误。)
3. **combined 条件**: `_act_oac` 的 q_ub_fn 用 `clamp(base_action + 残差均值)`(residual_actor=True),
   逐行镜像 update_critic; 由代码审查 + residual_actor=True 的 manual act 跑覆盖
   (因 base_action 同时进 actor 输入, 难做隔离的数值断言, 故不单设数值测试)。
4. **batched 独立**: B>1 时各行偏移相互独立, 改一行的 base/feat 不影响别行。
5. **CLI wiring**: `--oac_explore` 等 flag 正确落到 cfg(仿现有 `test_*_cli_wiring.py`)。
6. **形状/数值健壮**: `mu_E`、采样动作形状正确、无 NaN(含 grad 全 0 时 denom 不除零)。

## 8. 风险与调参提醒

- **δ 量级**: 偏移约 √(2δ)·std。std≈0.025, 默认 δ=0.5 -> 偏移≈0.025(约一个 std, 保守)。
  参考 humanoid 用 δ=23.5 是 pre-tanh + SAC 学习 std 的量级, 直接搬过来偏移会大到 ~0.17 远超
  action_scale 0.05, 故本仓默认必须小。后续按任务调。
- **共享 trunk 致 σ 偏小**: 本仓双 critic 头共享空间 trunk(SpatialEmbQEnsemble), |Q1-Q2| 比独立
  critic 小, OAC 的乐观信号偏弱; 是已知局限, 不在本次解决。
- **对高估塌方的脆弱性**: 本仓有 critic 高估->塌方史(见经验 gt_as_base_bug)。乐观采样会把乐观动作
  灌进 buffer; 虽"上界采样/下界学习"解耦不进更新, 仍建议 A/B 时 β_UB 给小, 盯住是否更易塌。
- **探索球本就小**: residual + action_scale + base 锚定把动作压在小球里, OAC 收益上限被球尺寸卡住,
  不会是 humanoid 量级(见 survey 第 10 节)。

## 9. 已核验代码锚点

- `off_policy/rl/actor.py:171-210` —— forward / scaled_mu / TruncatedNormal
- `off_policy/rl/q_agent.py:284-333` —— act / _act_default 采样路径
- `off_policy/rl/q_agent.py:359-364` —— update_critic 的 residual_actor 合动作 + clamp
- `off_policy/common_utils/utils.py:153` —— TruncatedNormal(loc/scale, low=-1/high=1, sample(clip))
- `off_policy/rl/critic.py:250-365` —— Critic / SpatialEmbQEnsemble / forward 返回 q_per_head[K,B,1]
- `chunk_residual/train_chunk_residual.py:87-104` —— add_chunk_transition 存 combined_action
- 参考: /mnt/mnt/data/wjm/residual/oac-explore/optimistic_exploration.py
