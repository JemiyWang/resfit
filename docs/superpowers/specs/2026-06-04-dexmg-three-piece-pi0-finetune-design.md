# dexmg three-piece pi0 微调设计

- 日期: 2026-06-04
- 仓: 代码改动在 /data2/kai0 (openpi, uv 环境); eval 脚本在 /data2/RL/residual-offpolicy-rl (conda residual)
- 关联:
  - 上游开关设计 specs/2026-06-04-pi05-base-policy-switch-design.md (本微调产出的 ckpt 供其 Task 9b/10/11 用)
  - 冒烟脚本 specs/2026-06-04-pi0-base-smoke-eval-design.md

---

## 1. 目标与背景

在 openpi 框架里微调一个 **pi0** 基座, 使其在 dexmg **TwoArmThreePieceAssembly**(双臂)上输出可用 action。
产出的 ckpt 经纯基座 eval 成功率 >= 30% 后, 作为冻结基座供 residual-offpolicy-rl 的残差 RL 使用。

为什么需要这步: residual RL 要求一个"没塌"的基座(Coffee 全 0% 教训)。官方 aloha 基座(pi0_aloha_sim / pi05_base)的
input transform 是 aloha 硬件专属(state 14 维, joint flip, gripper 线性映射), 与 three-piece(state 18, eef-delta)不兼容
-- 上次跨进程冒烟正是在 aloha_policy._decode_state 处按预期崩 (broadcast (14,) vs (18,))。因此必须自训一个 three-piece 适配的基座。

**范围**: 到"基座可用判定"为止 -- 即 data config + dexmg policy 类 + norm stats + 训练 + serve + 纯基座 eval gate。
residual RL 端到端不在本 spec(属上游开关 spec 的 Task 11)。

**训练档位**: 标准档 -- num_train_steps=30_000, gate 阈值成功率 >= 30%。

---

## 2. 核查确认的关键事实(都有 file:line)

### 2.1 action 空间完全对齐(pi0 能当基座的命门, 已确认)

- 数据 action 14 维 = robot0/1 各 [eef_delta_pos 3 + eef_delta_rot 3 + gripper 1]
  (convert_robomimic_to_lerobot.py:378-395)。
- 源 robomimic controller = OSC_POSE, input_type="delta", input_min/max=-1/1
  (two_arm_three_piece_assembly.hdf5 的 env_args controller_configs)。**action 本身已是 delta**(gripper 为绝对命令)。
- residual env action space = gym.spaces.Box(-1, 1, shape=(14,)) (resfit/dexmg/environments/dexmg.py:234-235),
  action_dim 直接取 robosuite env.action_dim = 14。
- 结论: pi0 输出 action[:, :14] 与 env 期望同语义同范围, **无需任何坐标变换**直接喂。

### 2.2 state 18 维语义

- robot0/1 各 [eef_pos 3 + eef_quat 4 + gripper_qpos 2] = 9, 合计 18 (dexmg.py:265-271)。
- 两端格式一致; 微调 mask_state=False(完全可观, 直接喂真值)。

### 2.3 三相机映射(刚好填满 pi0 三槽位)

- 数据三视角 84x84: observation.images.{agentview, robot0_eye_in_hand, robot1_eye_in_hand}。
- pi0 模型侧标准槽位 (pi0_config.py:56-78): base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb。
- 映射: agentview -> base_0_rgb, robot0_eye_in_hand -> left_wrist_0_rgb, robot1_eye_in_hand -> right_wrist_0_rgb。
  三槽位都有真图, image_mask 全 True, 无补零槽位(比单臂省心)。

### 2.4 蓝本与要避开的反例

- 蓝本: teleavatar_policy.py 的 TeleAvatarInputs/Outputs(双臂三相机结构)。
- 要避开(aloha 硬件专属, aloha_policy.py): _joint_flip_mask (104-106)、_gripper_to_angular (117-137)、
  _gripper_from_angular (140-156)、_decode_state (181-187)、_encode_actions (190-202)。新类一律不带。
- **关键正确性点**: 数据 action 已是 delta, 所以 data config **不加** DeltaActions transform
  (这点与 teleavatar 不同 -- 它数据是绝对关节才需 use_delta_joint_actions 转 delta)。直接用 raw 14 维 action。

### 2.5 数据现状

- LeRobot 数据集 repo_id=ankile/dexmg-two-arm-three-piece-assembly, 已缓存
  ~/.cache/huggingface/lerobot/ankile/dexmg-two-arm-three-piece-assembly/, 1002 episodes / 238847 帧 / 456MB, 84x84。
- 源 HDF5: deps/dexmimicgen/datasets/generated/two_arm_three_piece_assembly.hdf5 (4.1G)。
- 数据已是 LeRobot 格式, **无需再做 robomimic->LeRobot 转换**。

---

## 3. 组件设计

### 3.1 dexmg policy 类(新建 /data2/kai0/src/openpi/policies/dexmg_policy.py)

仿 teleavatar_policy 结构, 按 three-piece 维度填, 不带任何 aloha 硬件变换、不带 DeltaActions。

DexmgInputs.__call__:
- state(18 维): 非 PI05 时 pad_to_dim(state, action_dim=32) 补零; mask_state=False(直接喂真值)。
- 三相机 rename map: agentview->base_0_rgb, robot0_eye_in_hand->left_wrist_0_rgb,
  robot1_eye_in_hand->right_wrist_0_rgb; 三个 image_mask 全 True。
  图像同蓝本做 float32->uint8、(C,H,W)->(H,W,C)(若输入已是 HWC/uint8 则跳过对应步骤, 按蓝本判断)。
- actions(14 维, 训练时存在): pad_to_dim(actions, 32); mask_padding(PI0)时生成 action_mask 标记前 14 维有效。
- prompt passthrough。

DexmgOutputs.__call__:
- return {"actions": np.asarray(data["actions"][:, :14])}  # 切回 14 维。

构造参数: action_dim(取 model_config.action_dim)、model_type(默认 PI0)、mask_state(默认 False)。

### 3.2 data config(/data2/kai0/src/openpi/training/config.py 新增 DexmgThreePieceDataConfig)

仿 LerobotTeleAvatarDataConfig / LeRobotLiberoDataConfig:

- repack_transforms (LeRobot 键 -> 通用格式):
  images: {agentview->observation.images.agentview,
           robot0_eye_in_hand->observation.images.robot0_eye_in_hand,
           robot1_eye_in_hand->observation.images.robot1_eye_in_hand}
  state -> observation.state
  actions -> action
  (具体 repack 方向以 teleavatar 现成写法为准, 实现时对照其 RepackTransform 字典方向。)
- data_transforms: inputs=[DexmgInputs(action_dim=model.action_dim, model_type=model.model_type)],
                   outputs=[DexmgOutputs()]。
- 不加 DeltaActions。
- default_prompt = "assemble the three pieces"。
- model_transforms = ModelTransformFactory(default_prompt=...): 内含 ResizeImages(84->224)、
  TokenizePrompt、PadStatesAndActions(action_dim)。
- action_sequence_keys = ("action",); episodes=None(全用)。

### 3.3 TrainConfig 注册(同文件 _CONFIGS)

TrainConfig(
    name="pi0_dexmg_three_piece",
    model=pi0_config.Pi0Config(),                  # pi0: pi05=False(默认)
    data=DexmgThreePieceDataConfig(
        repo_id="ankile/dexmg-two-arm-three-piece-assembly",
        base_config=DataConfig(prompt_from_task=False),
        default_prompt="assemble the three pieces",
    ),
    weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    num_train_steps=30_000,
)

注: 字段名/路径以 config.py 现成 pi0_libero(953-979)为准照抄。

---

## 4. 数据流与流程

训练/推理数据流:
LeRobot 样本 -> repack(键归一) -> DexmgInputs(三相机 rename + state/action pad + mask) ->
  ModelTransform(resize 224 + tokenize prompt + pad) -> pi0 模型。
推理输出: pi0 (32 维 action chunk) -> DexmgOutputs(切前 14 维) -> 14 维 [-1,1] action -> (residual 端) env.step。

执行步骤:
1. norm stats(kai0 uv): uv run scripts/compute_norm_states_fast.py pi0_dexmg_three_piece。
2. 训练(kai0 uv): 先 ~1-2k 步冒烟(起训不崩 + loss 下降) ->
   uv run scripts/train.py pi0_dexmg_three_piece --exp_name=pi0_three_piece_v1 跑满 30k。记下 ckpt 目录。
3. serve(kai0 uv, 常驻, 占 1 卡): uv run scripts/serve_policy.py --policy.config pi0_dexmg_three_piece
   --policy.dir <ckpt> --default-prompt "assemble the three pieces" --port 8000。
4. gate(residual): 用 eval_pi05_base.py 的成功率版连该 server, 跑 N episode 纯基座成功率。

---

## 5. 错误处理与边界

- pi0_base 权重本地缺失: 实现 plan 第一步先查 /data2/kai0/checkpoints 有无 pi0_base/params;
  缺则单独解决(下载慢的老问题, 不阻塞 data config/policy 类的编写与单测)。
- prompt 三处一致: 微调 default_prompt = serve --default-prompt = (residual 端)BasePolicyConfig.prompt
  = "assemble the three pieces"。任一不一致会降基座表现, plan 显式校验。
- 维度断言: DexmgInputs 对 state(18)/action(14) 维度做显式校验, 不匹配早失败。
- gate 不达标(< 30%): 停, 排查 domain gap / prompt / 视角映射 / 归一化, 回到训练迭代; 不在塌掉的基座上做 RL。

---

## 6. 测试策略(TDD, kai0 uv 环境 pytest)

新建 /data2/kai0/src/openpi/policies/dexmg_policy_test.py(或仓内既有测试目录约定):
- test DexmgInputs 三槽位键名: 输出 images 键 == {base_0_rgb, left_wrist_0_rgb, right_wrist_0_rgb}, image_mask 全 True。
- test state/action pad: state、action 都 pad 到 32; action_mask 前 14 维 True、其余 False。
- test mask_state=False: state 非零(真值透传)。
- test DexmgOutputs: 喂 (B,32) actions -> 返回 (B,14)。
- test data config 构造: get_config("pi0_dexmg_three_piece").data.create(assets_dirs, model) 不报错,
  transform 链(repack -> data_transforms -> model_transforms)顺序正确。

冒烟(非单测, 需 GPU): norm stats 能算; 训练能起且 loss 下降; serve 起得来; residual 端连通。

---

## 7. 非目标(YAGNI)

- 不做 robomimic->LeRobot 转换(数据已是 LeRobot 格式)。
- 不微调 pi05(本 spec 固定 pi0; pi05 是 drop-in, 改 Pi0Config(pi05=True) 即可, 留作后续)。
- 不在本 spec 做 residual RL 端到端(属上游开关 spec Task 11)。
- 不改 teleavatar/aloha 现有 policy 类(新建独立 dexmg_policy.py, 零耦合)。

---

## 8. 交付物

- /data2/kai0/src/openpi/policies/dexmg_policy.py (DexmgInputs/DexmgOutputs)
- /data2/kai0/src/openpi/policies/dexmg_policy_test.py
- /data2/kai0/src/openpi/training/config.py 新增 DexmgThreePieceDataConfig + TrainConfig(name="pi0_dexmg_three_piece")
- norm stats assets(运行产出)
- 微调 ckpt(运行产出)
- residual 端 eval_pi05_base.py 的成功率 gate 版(上游开关 spec Task 10, 本 spec 复用)
