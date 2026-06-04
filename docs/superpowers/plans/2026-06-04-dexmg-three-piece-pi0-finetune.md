# dexmg three-piece pi0 微调 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 openpi 框架微调一个 pi0 基座, 使其在 dexmg TwoArmThreePieceAssembly(双臂, action 14 / state 18 / 三视角 84x84)上输出可用 action, 产出的 ckpt 经纯基座 eval 成功率 >=30% 后供 residual RL 当冻结基座。

**Architecture:** 新建独立 `dexmg_policy.py`(DexmgInputs/DexmgOutputs, 仿 teleavatar 但换相机键名、不带 aloha 硬件变换、不带 DeltaActions 因数据已是 delta), 在 config.py 注册 `DexmgThreePieceDataConfig` + `TrainConfig(name="pi0_dexmg_three_piece")`, 然后 norm stats -> 冒烟训练 -> 满训 30k -> serve -> 纯基座成功率 gate。

**Tech Stack:** Python 3.11, openpi(JAX 0.5.3 + PyTorch 2.7.1), uv 环境(kai0), LeRobot 数据, pytest(`*_test.py` 后缀), tyro CLI。

**关键事实(实现时不要再猜):**
- 数据集 repo_id = `ankile/dexmg-two-arm-three-piece-assembly`, 已缓存 `~/.cache/huggingface/lerobot/`, 84x84。
- action 14 = robot0/1 各 [eef_delta_pos3 + eef_delta_rot3 + gripper1], 已是 delta [-1,1]。state 18 = robot0/1 各 [eef_pos3 + eef_quat4 + gripper_qpos2]。
- 三相机 -> pi0 槽位: agentview->base_0_rgb, robot0_eye_in_hand->left_wrist_0_rgb, robot1_eye_in_hand->right_wrist_0_rgb。
- Pi0Config() 默认 action_dim=32, pi05=False; DexmgInputs 把 state/action pad 到 32, DexmgOutputs 切回 14。
- prompt 三处统一: `assemble the three pieces`。
- norm stats: `uv run scripts/compute_norm_states_fast.py --config-name <name>`(CLAUDE.md)。训练: `uv run scripts/train.py <name> --exp_name=<run>`。
- 所有 kai0 命令用 uv 环境; residual 端 eval 用 conda `residual`。

---

## File Structure

**新建(kai0):**
- `/data2/kai0/src/openpi/policies/dexmg_policy.py` — `DexmgInputs` / `DexmgOutputs`(数据<->模型变换, 单一职责)。
- `/data2/kai0/src/openpi/policies/dexmg_policy_test.py` — policy 变换单测。
- `/data2/kai0/src/openpi/training/dexmg_config_test.py` — config 注册单测。

**修改(kai0):**
- `/data2/kai0/src/openpi/training/config.py` — 加 `import ... dexmg_policy`(line 34 后)、`DexmgThreePieceDataConfig` 类、`_CONFIGS` 里 `TrainConfig(name="pi0_dexmg_three_piece")`。

**复用(运行, 不改):**
- `/data2/kai0/scripts/compute_norm_states_fast.py`, `scripts/train.py`, `scripts/serve_policy.py`。
- residual 端 `resfit/rl_finetuning/scripts/eval_pi05_base.py`(Task 7 升级成成功率 gate 版)。

---

## Task 1: DexmgInputs / DexmgOutputs 变换 + 单测(kai0, TDD)

**Files:**
- Create: `/data2/kai0/src/openpi/policies/dexmg_policy.py`
- Test: `/data2/kai0/src/openpi/policies/dexmg_policy_test.py`

- [ ] **Step 1: 写失败测试(全部用例)**

```python
# /data2/kai0/src/openpi/policies/dexmg_policy_test.py
import numpy as np

import openpi.models.model as _model
from openpi.policies.dexmg_policy import DexmgInputs, DexmgOutputs


def _img_chw(c=3, h=84, w=84):
    return np.zeros((c, h, w), dtype=np.float32)


def _full_obs(state, actions=None):
    data = {
        "images": {
            "agentview": _img_chw(),
            "robot0_eye_in_hand": _img_chw(),
            "robot1_eye_in_hand": _img_chw(),
        },
        "state": state,
    }
    if actions is not None:
        data["actions"] = actions
    return data


def test_renames_three_cameras_to_pi0_slots():
    out = DexmgInputs(action_dim=32, model_type=_model.ModelType.PI0)(
        _full_obs(np.zeros(18, dtype=np.float32))
    )
    assert set(out["image"].keys()) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert all(bool(m) for m in out["image_mask"].values())
    # LeRobot stores float32 (C,H,W); transform must emit uint8 (H,W,C)
    assert out["image"]["base_0_rgb"].shape == (84, 84, 3)
    assert out["image"]["base_0_rgb"].dtype == np.uint8


def test_pads_state_and_actions_to_32_and_masks_padding():
    out = DexmgInputs(action_dim=32, model_type=_model.ModelType.PI0)(
        _full_obs(np.ones(18, dtype=np.float32), actions=np.ones((10, 14), dtype=np.float32))
    )
    assert out["state"].shape == (32,)
    assert out["actions"].shape == (10, 32)
    # action_mask marks the real 14 dims valid, the 14->32 padding invalid
    assert out["action_mask"].shape == (10, 32)
    assert bool(out["action_mask"][:, :14].all())
    assert not bool(out["action_mask"][:, 14:].any())


def test_keeps_real_state_when_mask_state_false():
    out = DexmgInputs(action_dim=32, model_type=_model.ModelType.PI0, mask_state=False)(
        _full_obs(np.ones(18, dtype=np.float32))
    )
    # first 18 dims are the real state (=1), the rest is zero-padding
    assert float(out["state"][:18].sum()) == 18.0


def test_unexpected_camera_raises():
    data = _full_obs(np.zeros(18, dtype=np.float32))
    data["images"]["wrong_cam"] = _img_chw()
    import pytest

    with pytest.raises(ValueError):
        DexmgInputs(action_dim=32, model_type=_model.ModelType.PI0)(data)


def test_outputs_slices_first_14():
    out = DexmgOutputs()({"actions": np.zeros((5, 32), dtype=np.float32)})
    assert out["actions"].shape == (5, 14)
```

- [ ] **Step 2: 运行验证失败**

Run: `cd /data2/kai0 && uv run pytest src/openpi/policies/dexmg_policy_test.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'openpi.policies.dexmg_policy'`。

- [ ] **Step 3: 实现 dexmg_policy.py**

```python
# /data2/kai0/src/openpi/policies/dexmg_policy.py
"""Policy transforms for dexmg dual-arm tasks (e.g. TwoArmThreePieceAssembly).

State 18 = robot0[eef_pos3 + eef_quat4 + gripper_qpos2] + robot1[...] .
Action 14 = robot0[eef_delta_pos3 + eef_delta_rot3 + gripper1] + robot1[...] ,
already delta in [-1,1] (robomimic OSC_POSE, input_type=delta), so NO DeltaActions is applied.
Cameras agentview / robot0_eye_in_hand / robot1_eye_in_hand map to pi0 slots
base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb. No aloha hardware-specific transforms.
"""

import dataclasses
from typing import ClassVar

import numpy as np
import torch

import openpi.models.model as _model
import openpi.transforms as transforms

DEXMG_ACTION_DIM = 14


@dataclasses.dataclass(frozen=True)
class DexmgInputs(transforms.DataTransformFn):
    """Map a repacked dexmg sample into the pi0 model input dict."""

    # Model action dimension; used to zero-pad state and actions.
    action_dim: int
    # Determines whether padding mask is produced (pi0/pi0_rtc only).
    model_type: _model.ModelType = _model.ModelType.PI0
    # If True, zero out the state (ablation); default keeps the real state.
    mask_state: bool = False

    rename_map = {
        "agentview": "base_0_rgb",
        "robot0_eye_in_hand": "left_wrist_0_rgb",
        "robot1_eye_in_hand": "right_wrist_0_rgb",
    }
    EXPECTED_CAMERAS: ClassVar[tuple[str, ...]] = tuple(rename_map.keys())

    def __call__(self, data: dict) -> dict:
        # pi0/pi0_rtc are the only models that consume a padding mask here.
        mask_padding = self.model_type in (_model.ModelType.PI0, _model.ModelType.PI0_RTC)

        in_images = data["images"]
        if set(in_images) - set(self.EXPECTED_CAMERAS):
            raise ValueError(f"Expected images in {self.EXPECTED_CAMERAS}, got {tuple(in_images)}")

        if self.model_type in (_model.ModelType.PI05, _model.ModelType.PI05_RTC):
            state = np.asarray(data["state"])
        else:
            state = transforms.pad_to_dim(data["state"], self.action_dim)
        state = state.squeeze()

        images = {}
        image_masks = {}
        for camera in self.EXPECTED_CAMERAS:
            if camera not in in_images:
                raise ValueError(f"Camera {camera} not found in data")
            img = in_images[camera]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            # LeRobot stores float32; pi0 expects uint8.
            if np.issubdtype(img.dtype, np.floating):
                img = (255 * img).astype(np.uint8)
            # (C,H,W) -> (H,W,C)
            if img.shape[0] == 3:
                img = np.transpose(img, (1, 2, 0))
            images[self.rename_map[camera]] = img
            image_masks[self.rename_map[camera]] = np.True_

        masked_state = np.zeros_like(state) if self.mask_state else state
        inputs = {"image": images, "image_mask": image_masks, "state": masked_state}

        if "actions" in data:
            raw_action_dim = np.asarray(data["actions"]).shape[-1]
            if self.model_type in (_model.ModelType.PI05, _model.ModelType.PI05_RTC):
                actions = np.asarray(data["actions"])
            else:
                actions = transforms.pad_to_dim(data["actions"], self.action_dim)
            if mask_padding:
                action_mask = np.ones_like(actions, dtype=bool)
                action_mask[:, raw_action_dim:] = False
                inputs["action_mask"] = action_mask
            inputs["actions"] = actions.squeeze()

        if "prompt" in data:
            inputs["prompt"] = data["prompt"]

        return inputs


@dataclasses.dataclass(frozen=True)
class DexmgOutputs(transforms.DataTransformFn):
    """Slice the model output back to the real 14-dim dexmg action."""

    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][:, :DEXMG_ACTION_DIM])}
```

- [ ] **Step 4: 运行验证通过**

Run: `cd /data2/kai0 && uv run pytest src/openpi/policies/dexmg_policy_test.py -v`
Expected: 5 passed。

- [ ] **Step 5: lint**

Run: `cd /data2/kai0 && uv run ruff check --fix src/openpi/policies/dexmg_policy.py src/openpi/policies/dexmg_policy_test.py && uv run ruff format src/openpi/policies/dexmg_policy.py src/openpi/policies/dexmg_policy_test.py`
Expected: 无报错。

- [ ] **Step 6: Commit(kai0 仓)**

```bash
cd /data2/kai0
git add src/openpi/policies/dexmg_policy.py src/openpi/policies/dexmg_policy_test.py
git commit -m "feat(openpi): dexmg dual-arm policy transforms (DexmgInputs/Outputs)"
```

---

## Task 2: 注册 DexmgThreePieceDataConfig + TrainConfig(kai0)

**Files:**
- Modify: `/data2/kai0/src/openpi/training/config.py`(import 区 line 34 后; 新类放 `LerobotTeleAvatarDataConfig` 后即 line 555 附近; `TrainConfig` 放 `_CONFIGS` 里 `pi0_libero`(953)附近)
- Test: `/data2/kai0/src/openpi/training/dexmg_config_test.py`

- [ ] **Step 1: 写失败测试**

```python
# /data2/kai0/src/openpi/training/dexmg_config_test.py
import openpi.models.pi0_config as pi0_config
from openpi.training import config as _config
from openpi.training.config import DexmgThreePieceDataConfig


def test_pi0_dexmg_three_piece_registered():
    cfg = _config.get_config("pi0_dexmg_three_piece")
    assert isinstance(cfg.data, DexmgThreePieceDataConfig)
    assert cfg.data.repo_id == "ankile/dexmg-two-arm-three-piece-assembly"
    assert cfg.data.default_prompt == "assemble the three pieces"


def test_pi0_dexmg_three_piece_is_pi0_not_pi05():
    cfg = _config.get_config("pi0_dexmg_three_piece")
    assert isinstance(cfg.model, pi0_config.Pi0Config)
    assert cfg.model.pi05 is False
```

- [ ] **Step 2: 运行验证失败**

Run: `cd /data2/kai0 && uv run pytest src/openpi/training/dexmg_config_test.py -v`
Expected: FAIL(`ImportError: cannot import name 'DexmgThreePieceDataConfig'`)。

- [ ] **Step 3: 加 policy import**

在 `/data2/kai0/src/openpi/training/config.py` 第 34 行 `import openpi.policies.teleavatar_policy as teleavatar_policy` 之后新增一行:

```python
import openpi.policies.dexmg_policy as dexmg_policy
```

- [ ] **Step 4: 新增 DexmgThreePieceDataConfig 类**

在 `LerobotTeleAvatarDataConfig` 定义结束(约 line 555)之后插入:

```python
@dataclasses.dataclass(frozen=True)
class DexmgThreePieceDataConfig(DataConfigFactory):
    """dexmg TwoArmThreePieceAssembly: 18-dim state, 14-dim delta actions, 3 cameras.

    Actions are already delta in [-1,1] (robomimic OSC_POSE), so we do NOT apply
    DeltaActions here (unlike the joint-space configs).
    """

    # Injected as the prompt when the data has no "prompt" key.
    default_prompt: str | None = None
    episodes: list[int] | None = None

    repack_transforms: tyro.conf.Suppress[_transforms.Group] = dataclasses.field(
        default=_transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "images": {
                            "agentview": "observation.images.agentview",
                            "robot0_eye_in_hand": "observation.images.robot0_eye_in_hand",
                            "robot1_eye_in_hand": "observation.images.robot1_eye_in_hand",
                        },
                        "state": "observation.state",
                        "actions": "action",
                    }
                )
            ]
        )
    )

    action_sequence_keys: Sequence[str] = ("action",)
    mask_state: bool = False

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        data_transforms = _transforms.Group(
            inputs=[
                dexmg_policy.DexmgInputs(
                    action_dim=model_config.action_dim,
                    model_type=model_config.model_type,
                    mask_state=self.mask_state,
                )
            ],
            outputs=[dexmg_policy.DexmgOutputs()],
        )
        # NOTE: no DeltaActions -- dexmg actions are already delta.
        model_transforms = ModelTransformFactory(default_prompt=self.default_prompt)(model_config)

        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=self.repack_transforms,
            data_transforms=data_transforms,
            model_transforms=model_transforms,
            action_sequence_keys=self.action_sequence_keys,
            episodes=self.episodes,
        )
```

- [ ] **Step 5: 注册 TrainConfig**

在 `_CONFIGS` 列表里 `pi0_libero` 那条(约 line 953)之后插入:

```python
    TrainConfig(
        name="pi0_dexmg_three_piece",
        model=pi0_config.Pi0Config(),
        data=DexmgThreePieceDataConfig(
            repo_id="ankile/dexmg-two-arm-three-piece-assembly",
            base_config=DataConfig(prompt_from_task=False),
            default_prompt="assemble the three pieces",
        ),
        weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
    ),
```

- [ ] **Step 6: 运行验证通过**

Run: `cd /data2/kai0 && uv run pytest src/openpi/training/dexmg_config_test.py -v`
Expected: 2 passed。

- [ ] **Step 7: lint + commit**

```bash
cd /data2/kai0
uv run ruff check --fix src/openpi/training/config.py src/openpi/training/dexmg_config_test.py
uv run ruff format src/openpi/training/config.py src/openpi/training/dexmg_config_test.py
git add src/openpi/training/config.py src/openpi/training/dexmg_config_test.py
git commit -m "feat(openpi): register pi0_dexmg_three_piece train config"
```

---

## Task 3: 前置 — 确认 pi0_base 预训练权重可用(运行)

> weight_loader 指向 `gs://openpi-assets/checkpoints/pi0_base/params`。本机从 gs 下载极慢(~300KB/s), 需提前确认本地是否已有, 缺则单独处理(不阻塞 Task 1/2 的编码与单测)。

- [ ] **Step 1: 查本地是否已有 pi0_base 权重**

Run:
```bash
ls -la /data2/kai0/checkpoints/ 2>/dev/null | grep -i pi0
ls -la ~/.cache/openpi/openpi-assets/checkpoints/ 2>/dev/null | grep -i pi0
```
Expected: 看是否存在 `pi0_base/params`。

- [ ] **Step 2: 若缺失, 下载或改指本地**

- 若本地已有 `pi0_base/params`: 把 Task 2 的 `weight_loader` 改成本地绝对路径(`CheckpointWeightLoader("/abs/path/to/pi0_base/params")`), 重跑 Task 2 Step 6 测试 + 重新 commit。
- 若本地无: 后台启动下载并记录 ETA(不阻塞):
  Run: `cd /data2/kai0 && nohup uv run scripts/download_checkpoints.py > /tmp/dl_pi0_base.log 2>&1 &`
  然后继续 Task 4(norm stats 不依赖预训练权重); 训练(Task 5)前确认下载完成。

---

## Task 4: 计算归一化统计(运行, kai0 uv)

**Files:** 复用 `scripts/compute_norm_states_fast.py`(不改)。

- [ ] **Step 1: 算 norm stats**

Run: `cd /data2/kai0 && uv run scripts/compute_norm_states_fast.py --config-name pi0_dexmg_three_piece`
Expected: 遍历 `ankile/dexmg-two-arm-three-piece-assembly`, 在 assets 目录(asset_id = repo_id)下生成 `norm_stats.json`, 无报错。

- [ ] **Step 2: 校验产物**

Run: `find ~/.cache/openpi -path "*dexmg-two-arm-three-piece-assembly*norm_stats.json" 2>/dev/null; find /data2/kai0/assets -name "norm_stats.json" 2>/dev/null | grep -i dexmg`
Expected: 找到 norm_stats.json。记下其目录, 供 serve 端 `assets` 解析。

---

## Task 5: 冒烟训练(运行, 小步先验证起训不崩)

- [ ] **Step 1: 选空闲卡, 跑 ~2k 步冒烟**

Run:
```bash
cd /data2/kai0 && CUDA_VISIBLE_DEVICES=<free_gpu> WANDB_MODE=offline \
  uv run scripts/train.py pi0_dexmg_three_piece \
  --exp_name=pi0_three_piece_smoke --num_train_steps=2000 --save_interval=2000
```
Expected: 数据 loader 正常加载三相机/state18/action14, 不报维度/键错误; loss 随步数下降; 2000 步落一个 checkpoint。

- [ ] **Step 2: 判定**

- loss 正常下降且落 ckpt -> 进 Task 6 满训。
- 若起训即崩: 按报错定位(常见: repack 键名与数据集不符 / norm stats 缺失 / 显存不足调 batch_size)。修后重跑本步。**不要带病进满训。**

---

## Task 6: 满训 30k(运行)

- [ ] **Step 1: 启动满训**

Run:
```bash
cd /data2/kai0 && CUDA_VISIBLE_DEVICES=<free_gpu> WANDB_MODE=offline \
  uv run scripts/train.py pi0_dexmg_three_piece --exp_name=pi0_three_piece_v1
```
Expected: 跑满 num_train_steps=30_000, 按 save_interval 落 checkpoint。记下最终 checkpoint 目录(供 Task 7 serve, 记为 `<CKPT>`)。

- [ ] **Step 2: 记录**

把 `<CKPT>` 绝对路径与 exp_name 记入 `docs/superpowers/specs/2026-06-04-dexmg-three-piece-pi0-finetune-design.md` 的交付物清单, commit(仅文档):
```bash
cd /data2/RL/residual-offpolicy-rl
git add docs/superpowers/specs/2026-06-04-dexmg-three-piece-pi0-finetune-design.md
git commit -m "docs: record pi0 three-piece finetune checkpoint path"
```

---

## Task 7: serve + 纯基座成功率 gate

> 这是硬门槛: 基座成功率太低不进 residual RL(Coffee 全 0% 教训)。先 serve, 再把 residual 端的冒烟脚本升级成统计成功率的 gate 版。

**Files:**
- 运行: `/data2/kai0/scripts/serve_policy.py`(不改)。
- Modify(residual): `resfit/rl_finetuning/scripts/eval_pi05_base.py`(给 run_smoke 增加成功率统计)。
- Test(residual): `tests/rl_finetuning/test_eval_pi05_base.py`(加成功率统计单测)。

- [ ] **Step 1: GPU 端起 serve(kai0 uv, 常驻)**

先确认参数:
Run: `cd /data2/kai0 && uv run scripts/serve_policy.py --help 2>&1 | head -40`
然后(serve_policy 用 tyro subcommand `policy:checkpoint`):
```bash
cd /data2/kai0 && CUDA_VISIBLE_DEVICES=<free_gpu> uv run scripts/serve_policy.py policy:checkpoint \
  --policy.config pi0_dexmg_three_piece \
  --policy.dir <CKPT> \
  --policy.default_prompt "assemble the three pieces" \
  --port 8000
```
Expected: 打印加载完成并监听 :8000(`--policy.default_prompt` 的确切名以 --help 为准; 若该项不存在则用 serve 顶层 `--default-prompt`)。

- [ ] **Step 2: residual 端连通性自检**

Run:
```bash
conda run -n residual python -c "
import sys; sys.path.insert(0, '/data2/kai0')
from openpi_client.websocket_client_policy import WebsocketClientPolicy
WebsocketClientPolicy(host='127.0.0.1', port=8000); print('connected OK')
"
```
Expected: `connected OK`。

- [ ] **Step 3: 核查 residual env 如何报成功(实现 gate 前必读)**

Run: `grep -n "success\|is_success\|check_success\|terminated" /data2/RL/residual-offpolicy-rl/resfit/rl_finetuning/scripts/train_residual_td3.py | head -20`
据此确认: 一个 episode 的成功信号取自 `info["success"]`、`info` 的某键、还是 `terminated`。记下取法, Step 4 据此实现 `_episode_succeeded`。

- [ ] **Step 4: 写失败测试(成功率统计)**

在 `tests/rl_finetuning/test_eval_pi05_base.py` 追加(`_FakeEnv` 已存在于该文件, 让其 step 在末步返回成功标志, 按 Step 3 的取法对齐键名):

```python
def test_run_smoke_reports_success_rate():
    from resfit.rl_finetuning.scripts.eval_pi05_base import run_smoke

    # _FakeEnv: 末步置成功标志(键名与 _episode_succeeded 读取一致)
    env = _SuccessFakeEnv(success_on_last=True, max_len=3)
    policy = _FakePolicy(action_dim=14)
    report = run_smoke(env, policy, n_episodes=4, max_steps=3, action_dim=14)
    assert report["success_rate"] == 1.0
    assert report["n_success"] == 4
```

(`_SuccessFakeEnv` 仿现有 `_FakeEnv`, 在最后一步的 info/terminated 里按 Step 3 取法置成功。)

- [ ] **Step 5: 运行验证失败**

Run: `cd /data2/RL/residual-offpolicy-rl && conda run -n residual python -m pytest tests/rl_finetuning/test_eval_pi05_base.py::test_run_smoke_reports_success_rate -v`
Expected: FAIL(`report` 无 `success_rate` 键)。

- [ ] **Step 6: 实现成功率统计**

在 `eval_pi05_base.py` 加一个判定函数并在 `run_smoke` 累计(按 Step 3 取法填 `_episode_succeeded` 的真实键名):

```python
def _episode_succeeded(terminated, info) -> bool:
    """按 residual env 的成功信号判定一个 episode 是否成功(键名见 Task 7 Step 3)。"""
    if isinstance(info, dict) and "success" in info:
        return bool(info["success"])
    return bool(terminated)
```

在 `run_smoke` 的 episode 循环里: step 后用最近一次的 `terminated/info` 记录 `succeeded`, 循环末累计 `n_success`; 函数末计算:
```python
    report["n_success"] = n_success
    report["success_rate"] = n_success / n_episodes if n_episodes else 0.0
```
并在 `main` 打印 `success_rate`。

- [ ] **Step 7: 运行验证通过 + 回归**

Run: `cd /data2/RL/residual-offpolicy-rl && conda run -n residual python -m pytest tests/rl_finetuning/test_eval_pi05_base.py -v`
Expected: 全部 passed(新用例 + 原 12 个不破)。

- [ ] **Step 8: 跑真实 gate(需 Step 1 的 server 在跑)**

Run:
```bash
cd /data2/RL/residual-offpolicy-rl && conda run -n residual \
  python -m resfit.rl_finetuning.scripts.eval_pi05_base \
  --host 127.0.0.1 --port 8000 --n_episodes 50 --max_steps 400
```
Expected: 打印 success_rate。

- [ ] **Step 9: Gate 判定 + commit**

- success_rate >= 30% -> 基座可用, 进上游开关 plan 的 Task 11(residual RL 端到端)。
- < 30% -> 停, 排查 domain gap / prompt 是否三处一致 / 视角映射 / norm stats; 回 Task 5-6 迭代。**不在塌掉的基座上做 RL。**

```bash
cd /data2/RL/residual-offpolicy-rl
git add resfit/rl_finetuning/scripts/eval_pi05_base.py tests/rl_finetuning/test_eval_pi05_base.py
git commit -m "feat: success-rate gate for pi0 base eval"
```

---

## Self-Review 备注

**Spec 覆盖:**
- spec §3.1 dexmg policy 类 -> Task 1。
- spec §3.2 data config -> Task 2 Step 4。
- spec §3.3 TrainConfig -> Task 2 Step 5。
- spec §4 流程(norm/train/serve/gate)-> Task 4/5/6/7。
- spec §5 错误处理(pi0_base 缺失 / prompt 一致 / 维度断言 / gate)-> Task 3 / Task 7 Step 9 / Task 1 测试 / Task 7。
- spec §6 测试 -> Task 1(policy 单测)、Task 2(config 注册测试)、Task 7(成功率统计测试)。
- spec §7 YAGNI(无转换/不碰 teleavatar/aloha/独立文件)-> 新建 dexmg_policy.py, 不改现有 policy。

**类型一致:** `DexmgInputs(action_dim, model_type, mask_state)` 在 Task 1 定义、Task 2 Step 4 调用一致; `DexmgOutputs()` 无参一致; DEXMG_ACTION_DIM=14 切片与 spec action 14 一致; config name `pi0_dexmg_three_piece` 在 Task 2/4/5/6/7 全程一致; prompt `assemble the three pieces` 在 Task 2/7 一致。

**必要的实现时核查(非占位, 已显式标注步骤):**
- Task 3: pi0_base 权重本地有无(先 ls)。
- Task 7 Step 1: serve_policy.py 的 prompt 参数确切名(先 --help)。
- Task 7 Step 3: residual env 成功信号的确切键名(先 grep train_residual_td3.py)。
