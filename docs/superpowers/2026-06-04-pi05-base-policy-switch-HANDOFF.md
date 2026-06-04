# 交接文档：ACT/pi05 基座可切换开关（residual-offpolicy-rl）

- 日期：2026-06-04
- 分支：`chunk-residual-validation`（residual 仓）；kai0 改动在 kai0 仓 `main`
- 目标：让残差 RL 的冻结基座能在 ACT 与 pi05 间用配置开关切换，robomimic 仿真场景
- 关联文档：
  - 设计 spec：`docs/superpowers/specs/2026-06-04-pi05-base-policy-switch-design.md`
  - 实现 plan：`docs/superpowers/plans/2026-06-04-pi05-base-policy-switch.md`

---

## 1. 一句话现状

**开关代码骨架已全部完成、测试绿、review 通过；但 pi05 在 robomimic 上的实际效果尚未验证**——
剩下的微调 + serve + eval + 端到端（Task 8-11）需要 robomimic 数据、GPU 与人工决策，未做。

---

## 2. 已完成（Task 1-7 + cleanup）

全部在分支 `chunk-residual-validation`（kai0 改动在 kai0 `main`）。所有相关测试通过（5 个 residual 侧 + 17 个 kai0 adapter 侧）。

| Task | commit | 内容 |
|---|---|---|
| 1 | `b7d4c0e` | residual 端装 openpi-client；验证 import client+adapter 不触发 flax（numpy 未降级 1.26.4，torch 2.7.1 完好） |
| 2 (kai0) | `cdf2c0d` | `Pi05PolicyAdapter` 视角映射 `image_key_map` 可配置，默认值向后兼容 dsrl_pi05 |
| 3 | `9401f09` | `resfit/lerobot/policies/pi05/load_pi05.py` 工厂：连 `WebsocketClientPolicy` + `from_policy` |
| 4 | `07a17f3` | `BasePolicyConfig` 加 `type` 开关 + pi05 字段；ACT 默认值不变 |
| 5 | `f0dffe7` | `train_residual_td3.py` 加 `build_base_policy` 按 type 分发；ACT 路径行为等价 |
| 6 | `31250dd` | `residual_env_wrapper` 的 `base_policy` 注解放宽为 `BasePolicyProtocol` |
| 7 | `d2aee9a` | 集成 smoke（真 adapter + fake policy 跑 step 循环）+ ACT 回归 |
| cleanup | `42cd209` | 删未用 ACTConfig/ACTPolicy import，注解统一 `BasePolicyProtocol` |

### 验证已完成部分（随时可重跑）
```bash
cd /data2/RL/residual-offpolicy-rl
conda run -n residual python -m pytest tests/ -k "act or base_policy or pi05" -v   # 期望 5 passed
cd /data2/kai0
conda run -n residual python -m pytest resfit_pi05/tests -v                          # 期望 adapter 相关全绿
```

---

## 3. 架构关键决策：websocket 跨进程（必读）

**为何不是同进程**：conda env `residual` 纯 torch 2.7.1、**无 flax**；openpi 的
`src/openpi/training/config.py` 顶层 `import flax.nnx`，在 residual 内 import openpi 必炸。
无现成统一环境（base/kai0_convert 有 flax 但 torch 2.10，与 residual 的 robosuite/torchrl 栈不符）。

**方案**：
```
[GPU 端 · kai0 uv 环境]  scripts/serve_policy.py 加载 pi05 ckpt → WebsocketPolicyServer(:8000)
                                    ↕ 本地 websocket (msgpack_numpy)
[residual 端 · conda residual] WebsocketClientPolicy(host,port)  → Pi05PolicyAdapter.from_policy
                                    → BasePolicyVecEnvWrapper 的 base_policy（step 级）
```
- `Pi05PolicyAdapter.from_policy` 吃任何有 `.infer(obs)->{"actions":...}` 的对象，
  `WebsocketClientPolicy` 正好是 → 开关设计与同进程版几乎一致。
- residual 端只装轻量 `openpi-client`（依赖仅 dm-tree/msgpack/numpy<2/pillow/tree/websockets），不碰 flax/torch。

---

## 4. 关键文件

**residual 仓（/data2/RL/residual-offpolicy-rl）:**
- `resfit/rl_finetuning/config/residual_td3.py` — `BasePolicyConfig`（`type`/`host`/`port`/`prompt`/`action_dim`/`execute_horizon`/`image_key_map`/`kai0_paths`）
- `resfit/lerobot/policies/pi05/load_pi05.py` — `load_pi05_base_policy(cfg, device)`
- `resfit/rl_finetuning/scripts/train_residual_td3.py` — `build_base_policy(cfg, device) -> (policy, actor_name)`
- `resfit/rl_finetuning/wrappers/residual_env_wrapper.py` — `BasePolicyProtocol`
- `resfit/lerobot/dataset/convert_robomimic_to_lerobot.py` — 数据转换（现成，Task 8 用）

**kai0 仓（/data2/kai0）:**
- `resfit_pi05/pi05_policy_adapter.py` — `Pi05PolicyAdapter`（`from_policy` / `from_checkpoint` / `image_key_map`）
- `packages/openpi-client/...websocket_client_policy.py` — `WebsocketClientPolicy`
- `scripts/serve_policy.py` — GPU 端 serve（Task 9b 用）
- `src/openpi/training/config.py` — 微调 config 注册处（Task 9 要新增 robomimic 条目）

---

## 5. 剩余工作（Task 8-11，按 plan 执行）

> 详细步骤见 plan 文档对应 Task。下面是接手要点。

1. **Task 8（数据，residual env）**：robomimic HDF5 → LeRobot。
   - **缺输入**：具体任务名 + demo HDF5 路径（plan 默认占位单臂 Lift，action_dim=7）。
   - 命令：`conda run -n residual python resfit/lerobot/dataset/convert_robomimic_to_lerobot.py --dataset <HDF5> --output_dir ... --repo_id ...`
   - 产出后把"robomimic 图像键→pi05 槽位"映射与 action 维度记入 spec §10，并与 `BasePolicyConfig.image_key_map` 对齐。

2. **Task 9（微调，kai0 uv 环境，非 residual）**：
   - 对照 `LeRobotLiberoDataConfig` 在 `src/openpi/training/config.py` 新增 `Pi05RobomimicDataConfig` + `TrainConfig(name="pi05_robomimic_lift", model=Pi0Config(pi05=True), ...)`。
   - `cd /data2/kai0 && uv run scripts/compute_norm_states_fast.py --config-name pi05_robomimic_lift`
   - `cd /data2/kai0 && uv run scripts/train.py pi05_robomimic_lift --exp_name=pi05_lift_v1`
   - 记下产出的 checkpoint 目录。

3. **Task 9b（GPU serve）**：
   - 先 `uv run scripts/serve_policy.py --help` 确认参数名。
   - `cd /data2/kai0 && uv run scripts/serve_policy.py --policy.config pi05_robomimic_lift --policy.dir <CKPT> --default-prompt "<prompt>" --port 8000`（常驻）。
   - residual 端连通性自检（见 plan Task 9b Step 3）。

4. **Task 10（基座 eval gate，硬门槛）**：
   - 新建 `resfit/rl_finetuning/scripts/eval_pi05_base.py`，用 `load_pi05_base_policy`（连 server）在 robomimic 跑纯基座成功率。
   - 成功率达可用阈值（建议 ≥30%，按任务难度）才进 Task 11；否则排查 domain gap/prompt/视角映射/归一化，回 Task 8-9 迭代。**不要在塌掉的基座上做 RL**（参考历史 Coffee "基座全 0%" 教训）。

5. **Task 11（端到端）**：先确认 server 在跑，配置 `base_policy.type=pi05`/`host`/`port`，短跑冒烟 → 正式训练；并用 `type=act` 跑一遍确认 ACT 回归无副作用。

---

## 6. 已知坑 / 注意事项

- **环境分裂**：微调/serve 用 **kai0 的 uv 环境**（`uv run`，有 flax/jax）；RL 训练/eval/数据转换用 **conda `residual`**。别搞混。
- **numpy<2.0**：openpi-client 要求；本次安装未降级 residual 的 numpy（1.26.4），后续若重装依赖注意别破坏。
- **prompt 一致**：pi05 对 prompt 敏感，微调与推理（serve `--default-prompt` / `BasePolicyConfig.prompt`）必须同一句。
- **action 维度三处对齐**：robomimic action 维度 = `BasePolicyConfig.action_dim` = 残差叠加维度。
- **视角槽位**：单臂只有 agentview + 一个 eye_in_hand，pi05 第三槽位（右手腕）由 openpi transform 补零图 + mask；映射在微调 data config 与 `image_key_map` 两处都要对上。
- **eval_base_policy 双连接**：train 脚本会 build 两次基座 → pi05 时是两个 websocket client 连同一 server，需 server 支持并发连接（一般 OK）。
- **server 生命周期**：Task 10/11 依赖 server 常驻；server 挂则 RL 阻塞，注意健康检查/重连。
- **domain gap 风险**：pi05 为真机/aloha 大图设计，robomimic 仿真小图微调效果未知 → 先用一个简单单臂任务小步验证。
- **分支**：residual 改动在 `chunk-residual-validation`（与 stage-conditioned/residual_flow 工作同分支）；**kai0 的 adapter 改动提交在 kai0 `main`**——若需隔离可挪到独立分支。
- **可换 pi0（drop-in，见 spec §5.4 / plan Task 9·9b）**：接口对 pi0/pi05 透明，换 pi0 只在 GPU serve 端改 `--policy.config`/`--policy.dir`（自训则 `model=Pi0Config(pi05=False)`），residual 侧 `type="pi05"`/host/port 一行不动；纯冒烟可直接 serve 官方 `pi0_aloha_sim`，但 aloha 域 robomimic 成功率低，只能 smoke、不能当基座 gate。
- **未做的 nice-to-have**：没有"真 ACTPolicy 跑完整 build→wrapper→step 循环"的端到端回归测试（现有 ACT 回归是 import/默认值层面）。

---

## 7. 下一步最小动作

给出**具体 robomimic 任务名 + demo HDF5 路径**，即可从 Task 8 继续。其余按本文件 §5 顺序推进，每个 Phase 有 gate。
