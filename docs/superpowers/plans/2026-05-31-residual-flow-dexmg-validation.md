# residual_flow 在 dexmg 上的快速验证 — 实现计划

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在不改 ResFiT 原仓库代码的前提下,把残差 RL 抬到 chunk 级,移植 kai0/rlt 的 residual_flow actor 到 PyTorch,在 TwoArmBoxCleanup 上对 chunk-raw vs residual_flow 做干净 A/B,判断 residual_flow"有没有戏"。

**Architecture:** 冻结一切只换 actor。所有新文件放进 `resfit/rl_finetuning/chunk_residual/`,靠"用 `action_dim=480` 构造现成 `QAgent` + 把 residual_flow actor 注入 `agent.actor`"复用 critic/buffer/TD3 更新,不改任何现有文件。三阶段:M0 离线训冻动作自编码器(AE)→ M1 chunk 骨架+现成 raw actor(零新 ML)→ M2 注入 residual_flow 做 A/B。

**Tech Stack:** PyTorch 2.7.1、torchrl(replay buffer + MultiStepTransform)、lerobot(LeRobotDataset、ACTPolicy)、dexmg MuJoCo 仿真、conda env `residual`、wandb。

---

## 约定与共享常量(实现者必读)

- **维度**:`D = 24`(TwoArmBoxCleanup 每步动作维)、`L = 20`(chunk 长度 = ACT 的 `n_action_steps`)、`FLAT = L*D = 480`(展平 chunk 动作维)。state 维 = 38,3 个相机各 `[3,84,84]`。这些是 BoxCleanup 实测值;实现时应从 env / dataset 动态读取(见各 Task),常量仅供理解。
- **归一化两套别混**:ACT 内部动作用 MEAN_STD(`dataset.meta.stats["action"]` 的 mean/std);RL 残差空间用 `ActionScaler`(min-max 映到 [-1,1])。chunk 从 ACT 拿到的是**原始尺度**,必须再过 `action_scaler.scale()` 进入 [-1,1] 才能和残差相加。
- **actor 接口**:`forward(obs: dict, std: float) -> utils.TruncatedNormal`。`obs` 含 key:各相机名、`"observation.state"`、`"observation.base_action"`、`"feat"`(由 `QAgent._encode` 写入)。返回分布,`QAgent._act_default` 取 `.mean`(eval)或 `.sample(clip=stddev_clip)`(探索)。
- **base 相加约定**:`agent.act` 返回**纯残差**;wrapper 在执行时加 base、critic/actor loss 里加 base(`clamp(base+residual, -1, 1)`)。buffer 存 **combined**(`info["scaled_action"]`)。
- **conda / 运行**:所有命令从仓库根目录 `/data2/RL/residual-offpolicy-rl` 跑,`conda run -n residual ...`,eval 需 `MUJOCO_GL=egl PYOPENGL_PLATFORM=egl`。
- **提交策略**:每个 commit **只 `git add` 本 Task 列出的确切新文件路径**,绝不 `git add -A`/`git add .`——仓库 `main` 上有用户复现工作的未提交改动,必须原样不动。若你不想往该仓库提交,可跳过 commit 步骤(它们只是检查点)。新文件全在 `resfit/rl_finetuning/chunk_residual/` 与 `docs/`,不与现有改动冲突。
- **TruncatedNormal 导入**:`from resfit.rl_finetuning.off_policy.common_utils import utils`,用 `utils.TruncatedNormal(loc, scale)`。
- 单元测试(AE/actor)是纯 torch、CPU、无需数据/GPU;集成/冒烟测试需 env+基座+GPU,标注为 manual。

---

## 文件结构

```
resfit/rl_finetuning/chunk_residual/
  __init__.py                     # 空包标记
  action_autoencoder.py           # M0: PyTorch 版 ActionAutoencoder + 损失
  train_action_ae.py              # M0: AE 离线预训练脚本(带 delta_timestamps 数据通路)
  chunk_act_base.py               # M1: 从 ACTPolicy 一次性拿整段 L 步 chunk 的 helper
  chunk_env_wrapper.py            # M1: chunk 级 env wrapper(替代 BasePolicyVecEnvWrapper)
  residual_flow_actor.py          # M2: PyTorch 版 ResidualFlowActor(返回 TruncatedNormal)
  train_chunk_residual.py         # M1/M2: chunk 训练编排(--actor raw|flow 切换)
  tests/
    __init__.py
    test_action_autoencoder.py    # M0 单元测试
    test_chunk_env_wrapper.py     # M1 单元测试(fake env + fake base)
    test_residual_flow_actor.py   # M2 单元测试
```

---

# 阶段 M0 — 动作自编码器(AE)

## Task M0.1: PyTorch 版 ActionAutoencoder

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/__init__.py`
- Create: `resfit/rl_finetuning/chunk_residual/action_autoencoder.py`
- Create: `resfit/rl_finetuning/chunk_residual/tests/__init__.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_action_autoencoder.py`

参考源(JAX,照搬结构):`/data2/kai0/rlt/models_jax/action_autoencoder.py`。关键:JAX Conv 是 channels-last,PyTorch `Conv1d` 是 channels-first,卷积处需 transpose;LayerNorm 始终作用在 channels 维(channels-last 视角的最后一维)。

- [ ] **Step 1: 写失败测试**

`resfit/rl_finetuning/chunk_residual/tests/test_action_autoencoder.py`:
```python
import torch
from resfit.rl_finetuning.chunk_residual.action_autoencoder import (
    ActionAutoencoder,
    action_autoencoder_loss,
)

D, L, LAT = 24, 20, 64
FLAT = L * D


def _make_ae():
    return ActionAutoencoder(action_dim=D, chunk_length=L, latent_dim=LAT,
                             hidden_dim=256, conv_layers=2, conv_kernel=3)


def test_encode_decode_shapes():
    ae = _make_ae()
    x = torch.randn(8, FLAT)
    z = ae.encode(x)
    assert z.shape == (8, LAT)
    recon = ae.decode(z)
    assert recon.shape == (8, FLAT)
    recon2, z2 = ae(x, return_latent=True)
    assert recon2.shape == (8, FLAT) and z2.shape == (8, LAT)


def test_loss_is_scalar_and_decreases_on_overfit():
    torch.manual_seed(0)
    ae = _make_ae()
    x = torch.randn(16, FLAT)
    opt = torch.optim.Adam(ae.parameters(), lr=1e-3)
    loss0, m0 = action_autoencoder_loss(ae, x, velocity_loss_weight=0.1)
    assert loss0.ndim == 0 and "recon_l1" in m0 and "velocity_l1" in m0
    for _ in range(300):
        opt.zero_grad()
        loss, _ = action_autoencoder_loss(ae, x, velocity_loss_weight=0.1)
        loss.backward()
        opt.step()
    loss1, _ = action_autoencoder_loss(ae, x, velocity_loss_weight=0.1)
    assert loss1.item() < 0.25 * loss0.item()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /data2/RL/residual-offpolicy-rl && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_action_autoencoder.py -v`
Expected: FAIL(`ModuleNotFoundError: ... action_autoencoder`)

- [ ] **Step 3: 写实现**

`resfit/rl_finetuning/chunk_residual/__init__.py`:(空文件)

`resfit/rl_finetuning/chunk_residual/tests/__init__.py`:(空文件)

`resfit/rl_finetuning/chunk_residual/action_autoencoder.py`:
```python
"""PyTorch 版动作 chunk 自编码器(从 kai0/rlt 的 JAX 版移植)。

时序卷积 AE,作用在已归一化的展平 chunk [B, chunk_length*action_dim] 上。
LayerNorm 始终作用在 channel 维(channels-last 视角最后一维)。
"""
from __future__ import annotations

import torch
from torch import nn


class _Conv1dBlock(nn.Module):
    """channels-last 输入 [B,T,C] 的 Conv1d + LayerNorm(C) + ReLU。"""

    def __init__(self, channels: int, kernel: int, use_layer_norm: bool):
        super().__init__()
        pad = kernel // 2  # SAME(kernel 为奇数,如 3 → pad=1)
        self.conv = nn.Conv1d(channels, channels, kernel_size=kernel, padding=pad)
        self.norm = nn.LayerNorm(channels) if use_layer_norm else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B,T,C]
        y = self.conv(x.transpose(1, 2)).transpose(1, 2)  # [B,T,C]
        if self.norm is not None:
            y = self.norm(y)
        return torch.relu(y)


class ActionAutoencoder(nn.Module):
    def __init__(self, action_dim: int, chunk_length: int, latent_dim: int = 64,
                 hidden_dim: int = 256, conv_layers: int = 2, conv_kernel: int = 3,
                 use_layer_norm: bool = True):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_length = chunk_length
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.flat_action_dim = chunk_length * action_dim

        self.encoder_input_dense = nn.Linear(action_dim, hidden_dim)
        self.encoder_convs = nn.ModuleList(
            [_Conv1dBlock(hidden_dim, conv_kernel, use_layer_norm) for _ in range(conv_layers)]
        )
        self.encoder_latent_dense = nn.Linear(hidden_dim, latent_dim)
        self.latent_norm = nn.LayerNorm(latent_dim) if use_layer_norm else None

        self.decoder_input_dense = nn.Linear(latent_dim, chunk_length * hidden_dim)
        self.decoder_convs = nn.ModuleList(
            [_Conv1dBlock(hidden_dim, conv_kernel, use_layer_norm) for _ in range(conv_layers)]
        )
        self.decoder_output_dense = nn.Linear(hidden_dim, action_dim)

    def _check(self, x: torch.Tensor):
        if x.ndim != 2 or x.shape[-1] != self.flat_action_dim:
            raise ValueError(f"expected [B,{self.flat_action_dim}], got {tuple(x.shape)}")

    def encode(self, action_flat: torch.Tensor) -> torch.Tensor:
        self._check(action_flat)
        b = action_flat.shape[0]
        x = action_flat.reshape(b, self.chunk_length, self.action_dim)
        x = torch.relu(self.encoder_input_dense(x))   # [B,T,H]
        for blk in self.encoder_convs:
            x = blk(x)
        x = x.mean(dim=1)                              # [B,H] 沿时间平均
        z = self.encoder_latent_dense(x)              # [B,LAT]
        if self.latent_norm is not None:
            z = self.latent_norm(z)
        return z

    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError(f"expected [B,{self.latent_dim}], got {tuple(latent.shape)}")
        b = latent.shape[0]
        x = self.decoder_input_dense(latent)
        x = x.reshape(b, self.chunk_length, self.hidden_dim)
        x = torch.relu(x)
        for blk in self.decoder_convs:
            x = blk(x)
        seq = self.decoder_output_dense(x)            # [B,T,D]
        return seq.reshape(b, self.flat_action_dim)

    def forward(self, action_flat: torch.Tensor, return_latent: bool = False):
        z = self.encode(action_flat)
        recon = self.decode(z)
        if return_latent:
            return recon, z
        return recon


def action_autoencoder_loss(model: ActionAutoencoder, action_flat: torch.Tensor,
                            velocity_loss_weight: float = 0.1):
    """L1 重构 + 速度损失(相邻步差分 L1)。返回 (loss, metrics)。"""
    recon = model(action_flat)
    recon_l1 = (recon - action_flat).abs().mean()

    b = action_flat.shape[0]
    tgt = action_flat.reshape(b, model.chunk_length, model.action_dim)
    rec = recon.reshape(b, model.chunk_length, model.action_dim)
    if model.chunk_length > 1:
        tgt_v = tgt[:, 1:, :] - tgt[:, :-1, :]
        rec_v = rec[:, 1:, :] - rec[:, :-1, :]
        velocity_l1 = (rec_v - tgt_v).abs().mean()
    else:
        velocity_l1 = torch.zeros((), dtype=recon.dtype, device=recon.device)

    loss = recon_l1 + velocity_loss_weight * velocity_l1
    return loss, {"loss": loss.detach(), "recon_l1": recon_l1.detach(),
                  "velocity_l1": velocity_l1.detach()}
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /data2/RL/residual-offpolicy-rl && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_action_autoencoder.py -v`
Expected: PASS(2 passed)

- [ ] **Step 5: 提交**

```bash
cd /data2/RL/residual-offpolicy-rl
git add resfit/rl_finetuning/chunk_residual/__init__.py \
        resfit/rl_finetuning/chunk_residual/action_autoencoder.py \
        resfit/rl_finetuning/chunk_residual/tests/__init__.py \
        resfit/rl_finetuning/chunk_residual/tests/test_action_autoencoder.py
git commit -m "feat(chunk_residual): PyTorch ActionAutoencoder 移植 + 单元测试"
```

---

## Task M0.2: AE 预训练脚本

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/train_action_ae.py`

参考:数据通路仿 `resfit/lerobot/scripts/train_bc_dexmg.py:540-552`(用 `resolve_delta_timestamps` 拿 [B,20,24] chunk);归一化仿 `train_residual_td3.py:246-251`(`ActionScaler.from_dataset_stats(dataset.meta.stats["action"])`)。

- [ ] **Step 1: 写脚本**

`resfit/rl_finetuning/chunk_residual/train_action_ae.py`:
```python
"""离线预训练动作 chunk 自编码器(AE),用 RL 同款 ActionScaler 归一化。

用法(从仓库根目录,conda env residual):
  python -m resfit.rl_finetuning.chunk_residual.train_action_ae \
      --dataset ankile/dexmg-two-arm-box-cleanup \
      --chunk_length 20 --action_scale 0.2 --min_range_per_dim 0.1 \
      --steps 20000 --batch_size 256 \
      --out resfit/rl_finetuning/chunk_residual/ckpt/ae_boxcleanup.pt
冻结后的权重供 M2 的 residual_flow actor 加载。
"""
from __future__ import annotations

import argparse
import os

import torch
from torch.utils.data import DataLoader

from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.factory import resolve_delta_timestamps

from resfit.lerobot.policies.act.configuration_act import ACTConfig
from resfit.rl_finetuning.utils.normalization import ActionScaler
from resfit.rl_finetuning.chunk_residual.action_autoencoder import (
    ActionAutoencoder, action_autoencoder_loss,
)


def build_chunk_dataset(repo_id: str, chunk_length: int):
    """构造带 delta_timestamps 的 LeRobotDataset,使 sample['action'] 为 [chunk_length, D]。"""
    policy_cfg = ACTConfig()
    policy_cfg.chunk_size = chunk_length
    policy_cfg.n_action_steps = chunk_length
    meta = LeRobotDataset(repo_id).meta            # 先拿 meta 解析 delta_timestamps
    delta_timestamps = resolve_delta_timestamps(policy_cfg, meta)
    ds = LeRobotDataset(repo_id, delta_timestamps=delta_timestamps, download_videos=False)
    return ds


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--chunk_length", type=int, default=20)
    p.add_argument("--latent_dim", type=int, default=64)
    p.add_argument("--hidden_dim", type=int, default=256)
    p.add_argument("--conv_layers", type=int, default=2)
    p.add_argument("--action_scale", type=float, default=0.2)
    p.add_argument("--min_range_per_dim", type=float, default=0.1)
    p.add_argument("--steps", type=int, default=20000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--velocity_loss_weight", type=float, default=0.1)
    p.add_argument("--val_fraction", type=float, default=0.05)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", required=True)
    args = p.parse_args()

    ds = build_chunk_dataset(args.dataset, args.chunk_length)
    action_stats = ds.meta.stats["action"]
    action_dim = len(action_stats["min"])
    scaler = ActionScaler.from_dataset_stats(
        action_stats=action_stats, action_scale=args.action_scale,
        min_range_per_dim=args.min_range_per_dim, device=args.device,
    )

    n_val = max(1, int(len(ds) * args.val_fraction))
    n_train = len(ds) - n_val
    g = torch.Generator().manual_seed(0)
    train_ds, val_ds = torch.utils.data.random_split(ds, [n_train, n_val], generator=g)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=4, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    ae = ActionAutoencoder(action_dim=action_dim, chunk_length=args.chunk_length,
                           latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
                           conv_layers=args.conv_layers).to(args.device)
    opt = torch.optim.AdamW(ae.parameters(), lr=args.lr)

    def to_flat(sample):
        a = sample["action"].float().to(args.device)        # [B, L, D]
        a = scaler.scale(a)                                  # [-1,1]
        return a.reshape(a.shape[0], -1)                     # [B, L*D]

    step = 0
    ae.train()
    while step < args.steps:
        for sample in train_loader:
            x = to_flat(sample)
            opt.zero_grad()
            loss, m = action_autoencoder_loss(ae, x, args.velocity_loss_weight)
            loss.backward()
            opt.step()
            step += 1
            if step % 200 == 0:
                print(f"[step {step}] loss={m['loss']:.4f} recon_l1={m['recon_l1']:.4f} "
                      f"vel_l1={m['velocity_l1']:.4f}")
            if step >= args.steps:
                break

    # held-out 评估
    ae.eval()
    with torch.no_grad():
        rec_sum, vel_sum, n = 0.0, 0.0, 0
        for sample in val_loader:
            x = to_flat(sample)
            _, m = action_autoencoder_loss(ae, x, args.velocity_loss_weight)
            rec_sum += float(m["recon_l1"]) * x.shape[0]
            vel_sum += float(m["velocity_l1"]) * x.shape[0]
            n += x.shape[0]
        print(f"[VAL] recon_l1={rec_sum / n:.5f} velocity_l1={vel_sum / n:.5f}")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({
        "state_dict": ae.state_dict(),
        "ae_config": {"action_dim": action_dim, "chunk_length": args.chunk_length,
                      "latent_dim": args.latent_dim, "hidden_dim": args.hidden_dim,
                      "conv_layers": args.conv_layers},
        "action_scaler": {"action_min": action_stats["min"], "action_max": action_stats["max"],
                          "action_scale": args.action_scale,
                          "min_range_per_dim": args.min_range_per_dim},
    }, args.out)
    print(f"saved AE to {args.out}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 语法/导入冒烟检查(不下数据)**

Run: `cd /data2/RL/residual-offpolicy-rl && conda run -n residual python -c "import resfit.rl_finetuning.chunk_residual.train_action_ae as m; print('import ok')"`
Expected: 打印 `import ok`(无 ImportError)

- [ ] **Step 3: 提交**

```bash
cd /data2/RL/residual-offpolicy-rl
git add resfit/rl_finetuning/chunk_residual/train_action_ae.py
git commit -m "feat(chunk_residual): AE 离线预训练脚本"
```

---

## Task M0.3: 实跑 AE 预训练并验收(manual,go/no-go)

**Files:** 无新文件(运行 + 记录)。

- [ ] **Step 1: 跑预训练**

```bash
cd /data2/RL/residual-offpolicy-rl
conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_action_ae \
    --dataset ankile/dexmg-two-arm-box-cleanup \
    --chunk_length 20 --action_scale 0.2 --min_range_per_dim 0.1 \
    --steps 20000 --batch_size 256 \
    --out resfit/rl_finetuning/chunk_residual/ckpt/ae_boxcleanup.pt
```
说明:`--dataset` 用 BoxCleanup 的 repo_id(同 `config/residual_td3.py` 里 `ResidualTD3BoxCleanConfig.offline_data.name`,值为 `ankile/dexmg-two-arm-box-cleanup`)。`action_scale`/`min_range_per_dim` 必须与 RL 端一致(BoxCleanup 用 action_scale=0.2)。

- [ ] **Step 2: 验收 go/no-go**

看末尾 `[VAL] recon_l1=...`。**判据**:`recon_l1` 应明显小于"动作典型幅度"(归一化到 [-1,1] 后,recon_l1 期望 < ~0.05;若 > 0.1 说明 AE 欠拟合,需加 steps/hidden 或查数据)。同时 `velocity_l1` 应同量级偏小。
- 若不达标:增大 `--steps`(如 50000)或 `--hidden_dim`(如 512),重训。
- 达标后 `ckpt/ae_boxcleanup.pt` 即为冻结 AE,供 M2 使用。

---

# 阶段 M1 — chunk 级骨架 + 现成 raw actor

## Task M1.1: 从 ACT 拿整段 chunk 的 helper + chunk env wrapper

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/chunk_act_base.py`
- Create: `resfit/rl_finetuning/chunk_residual/chunk_env_wrapper.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py`

参考:取整段 chunk 仿 `modeling_act.py:164-167,209-210`;wrapper 结构仿 `wrappers/residual_env_wrapper.py`(`_augment_obs`、`step` 的 base+residual+unscale、`info["scaled_action"]`、final_obs 处理)。

- [ ] **Step 1: 写失败测试(用 fake env + fake base,纯 CPU)**

`resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py`:
```python
import gymnasium as gym
import numpy as np
import torch

from resfit.rl_finetuning.chunk_residual.chunk_env_wrapper import ChunkResidualEnvWrapper

D, L = 4, 5  # 小尺寸便于测试


class _FakeVecEnv:
    """单环境、确定性、第 3 步后 terminate 的假向量环境。"""
    def __init__(self):
        self.action_space = gym.spaces.Box(low=-1, high=1, shape=(D,), dtype=np.float32)
        self.observation_space = gym.spaces.Dict({
            "observation.state": gym.spaces.Box(-np.inf, np.inf, (1, 3), dtype=np.float32),
            "observation.images.cam": gym.spaces.Box(0, 1, (1, 3, 4, 4), dtype=np.float32),
        })
        self._t = 0
        self.last_actions = []

    def _obs(self):
        return {
            "observation.state": torch.zeros(1, 3),
            "observation.images.cam": torch.zeros(1, 3, 4, 4),
        }

    def reset(self, **kw):
        self._t = 0
        self.last_actions = []
        return self._obs(), {}

    def step(self, action):
        self.last_actions.append(action.clone())
        self._t += 1
        terminated = torch.tensor([self._t >= 3])
        truncated = torch.tensor([False])
        reward = torch.tensor([1.0 if self._t == 3 else 0.0])
        info = {}
        return self._obs(), reward, terminated, truncated, info


class _FakeBase:
    """返回固定 chunk(原始尺度)的假 ACT。"""
    class _Cfg:
        image_features = {"observation.images.cam": None}
    config = _Cfg()

    def reset(self, env_ids=None):
        pass

    def get_action_chunk(self, raw_obs, chunk_length):
        b = raw_obs["observation.state"].shape[0]
        # 原始尺度 chunk:每步全 0.5
        return torch.full((b, chunk_length, D), 0.5)


class _IdentityScaler:
    def scale(self, a):       # 原样(测试里不验证归一化数值)
        return torch.clamp(a, -1, 1)
    def unscale(self, a):
        return a


class _IdentityStd:
    def standardize(self, s):
        return s


def test_step_executes_chunk_and_accumulates_reward():
    env = _FakeVecEnv()
    w = ChunkResidualEnvWrapper(env, _FakeBase(), _IdentityScaler(), _IdentityStd(),
                                chunk_length=L)
    obs, _ = w.reset()
    assert obs["observation.base_action"].shape == (1, L * D)   # 展平 chunk
    residual = torch.zeros(1, L * D)
    next_obs, reward, terminated, truncated, info = w.step(residual)
    # fake env 第 3 步 terminate → 只执行了 3 步
    assert len(env.last_actions) == 3
    assert terminated.item() is True
    assert reward.item() == 1.0                                 # 累积到成功
    assert info["scaled_action"].shape == (1, L * D)            # combined chunk
    assert next_obs["observation.base_action"].shape == (1, L * D)


def test_residual_is_added_to_base_before_unscale():
    env = _FakeVecEnv()
    scaler = _IdentityScaler()
    w = ChunkResidualEnvWrapper(env, _FakeBase(), scaler, _IdentityStd(), chunk_length=L)
    w.reset()
    residual = torch.full((1, L * D), 0.1)
    w.step(residual)
    # base=0.5(scale 后仍 0.5),+0.1 → 0.6,unscale identity → env 收到 ~0.6
    assert torch.allclose(env.last_actions[0], torch.full((1, D), 0.6), atol=1e-5)
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /data2/RL/residual-offpolicy-rl && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py -v`
Expected: FAIL(`ModuleNotFoundError: ... chunk_env_wrapper`)

- [ ] **Step 3: 写 helper**

`resfit/rl_finetuning/chunk_residual/chunk_act_base.py`:
```python
"""不修改 ACT 的前提下,一次性从 ACTPolicy 拿整段 chunk(原始动作尺度)。

复用 ACTPolicy 的 normalize_inputs / 图像重组 / model / unnormalize_outputs。
"""
from __future__ import annotations

import torch


@torch.no_grad()
def get_action_chunk(base_policy, raw_obs: dict, chunk_length: int) -> torch.Tensor:
    """返回 [B, chunk_length, action_dim] 的原始尺度动作 chunk。"""
    base_policy.eval()
    batch = base_policy.normalize_inputs(raw_obs)          # modeling_act.py:164
    batch = dict(batch)
    batch["observation.images"] = [batch[k] for k in base_policy.config.image_features]  # :167
    chunk = base_policy.model(batch)[0]                    # [B, chunk_size, D] 归一化空间 :209
    chunk = base_policy.unnormalize_outputs({"action": chunk})["action"]  # 原始尺度 :210
    return chunk[:, :chunk_length]
```

- [ ] **Step 4: 写 wrapper**

`resfit/rl_finetuning/chunk_residual/chunk_env_wrapper.py`:
```python
"""chunk 级残差 env wrapper(替代 BasePolicyVecEnvWrapper,不修改原文件)。

约定(与 ResFiT 一致):
- base_action 以归一化 [-1,1] chunk 存于 obs["observation.base_action"](展平 L*D)。
- step 收到的是归一化残差 chunk(展平 L*D);combined = clamp(base+residual,-1,1);
  reshape → unscale → 在底层 env 上开环逐步执行;累积奖励;中途 done 则提前结束。
- info["scaled_action"] = combined(展平),供 buffer 存储(与原 wrapper L145 一致)。
仅支持 num_envs==1(训练端 assert 限制)。
"""
from __future__ import annotations

import torch

from resfit.rl_finetuning.chunk_residual.chunk_act_base import get_action_chunk


class ChunkResidualEnvWrapper:
    def __init__(self, vec_env, base_policy, action_scaler, state_standardizer,
                 chunk_length: int):
        self.vec_env = vec_env
        self.base_policy = base_policy
        self.action_scaler = action_scaler
        self.state_standardizer = state_standardizer
        self.chunk_length = chunk_length
        self.action_dim = vec_env.action_space.shape[-1]
        self.flat_dim = chunk_length * self.action_dim
        # 给注入端/训练端用:暴露 base_policy 取 chunk 的方式(测试可 monkeypatch)
        self._get_chunk = getattr(base_policy, "get_action_chunk", None)

    # ---- 取整段 chunk 并归一化到 [-1,1] 展平 ----
    def _base_chunk_flat(self, raw_obs):
        if self._get_chunk is not None:                      # fake/可替换路径
            chunk_raw = self._get_chunk(raw_obs, self.chunk_length)
        else:
            chunk_raw = get_action_chunk(self.base_policy, raw_obs, self.chunk_length)
        chunk_n = self.action_scaler.scale(chunk_raw)        # [B,L,D] -> [-1,1]
        return chunk_n.reshape(chunk_n.shape[0], -1)         # [B, L*D]

    def _augment(self, raw_obs, base_flat):
        aug = dict(raw_obs)
        aug["observation.base_action"] = base_flat
        aug["observation.state"] = self.state_standardizer.standardize(raw_obs["observation.state"])
        return aug

    def reset(self, **kwargs):
        raw_obs, info = self.vec_env.reset(**kwargs)
        self.base_policy.reset()
        base_flat = self._base_chunk_flat(raw_obs)
        self._last_base_flat = base_flat
        return self._augment(raw_obs, base_flat), info

    def step(self, residual_flat: torch.Tensor):
        combined_flat = torch.clamp(self._last_base_flat + residual_flat, -1.0, 1.0)  # [B,L*D]
        b = combined_flat.shape[0]
        combined_chunk = combined_flat.reshape(b, self.chunk_length, self.action_dim)
        env_chunk = self.action_scaler.unscale(combined_chunk)        # [B,L,D] 原始尺度

        total_reward = torch.zeros(b)
        terminated = torch.zeros(b, dtype=torch.bool)
        truncated = torch.zeros(b, dtype=torch.bool)
        last_info, final_raw_obs = {}, None
        raw_obs = None
        for t in range(self.chunk_length):
            raw_obs, reward, term, trunc, info = self.vec_env.step(env_chunk[:, t])
            total_reward = total_reward + reward.float().cpu()
            terminated = terminated | term.bool().cpu()
            truncated = truncated | trunc.bool().cpu()
            last_info = info
            if bool((term | trunc).any()):
                final_raw_obs = info.get("final_obs", None)
                break

        # 为 chunk 末观测取下一段 base chunk(若 done,底层已 autoreset 到新 episode 起点)
        if bool((terminated | truncated).any()):
            self.base_policy.reset()
        base_flat = self._base_chunk_flat(raw_obs)
        self._last_base_flat = base_flat

        aug_obs = self._augment(raw_obs, base_flat)
        info = dict(last_info)
        info["scaled_action"] = combined_flat
        if final_raw_obs is not None:
            info["final_obs"] = final_raw_obs
        return aug_obs, total_reward, terminated, truncated, info

    def render(self):
        return self.vec_env.render()

    def close(self):
        return self.vec_env.close()

    def __getattr__(self, name):
        return getattr(self.vec_env, name)
```

注:测试里 `_FakeBase` 提供了 `get_action_chunk`,故走 `self._get_chunk` 路径,不触碰真 ACT。真实运行时 `base_policy`(ACTPolicy)无该方法 → 走 `get_action_chunk(...)` helper。

- [ ] **Step 5: 跑测试确认通过**

Run: `cd /data2/RL/residual-offpolicy-rl && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py -v`
Expected: PASS(2 passed)

- [ ] **Step 6: 提交**

```bash
cd /data2/RL/residual-offpolicy-rl
git add resfit/rl_finetuning/chunk_residual/chunk_act_base.py \
        resfit/rl_finetuning/chunk_residual/chunk_env_wrapper.py \
        resfit/rl_finetuning/chunk_residual/tests/test_chunk_env_wrapper.py
git commit -m "feat(chunk_residual): chunk 级 env wrapper + ACT 整段 chunk helper"
```

---

## Task M1.2: chunk 训练编排脚本(--actor raw)

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`

参考(确切接口):`train_residual_td3.py` 的 main 循环(L855 reset / L944-954 act+step / L978-991 add / L1030-1067 update / L998-1010 eval),`_add_transitions_to_buffer`(L147-199),buffer 构造(L400-410),`QAgent(...)`(L365-372),`run_dexmg_evaluation`(`utils/evaluate_dexmg.py`),`create_vectorized_env`(`dexmg.py:624`)。

本 Task 只实现 `--actor raw`(用现成 `Actor`,即 `QAgent(action_dim=480, residual_actor=True)`,零新 ML)。`--actor flow` 的注入在 M2.2 加。

- [ ] **Step 1: 写脚本**

`resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`:
```python
"""chunk 级残差 RL 训练编排(不改原仓库)。

复用:create_vectorized_env、ACTPolicy 基座、QAgent(action_dim=L*D)、torchrl buffer、
run_dexmg_evaluation。只把 env wrapper 换成 ChunkResidualEnvWrapper,RL 决策粒度=chunk。

--actor raw  : 用现成 Actor(M1)
--actor flow : 注入 residual_flow actor(M2,见 _maybe_inject_flow_actor)

用法见文件末 README 注释。x 轴用"环境步"(每 chunk 计 chunk_length 步),与单步基线可比。
"""
from __future__ import annotations

import argparse
import copy
import os

import torch
from tensordict import TensorDict
from torchrl.data import LazyTensorStorage, TensorDictPrioritizedReplayBuffer

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.rl_finetuning.off_policy.rl.q_agent import QAgent
from resfit.rl_finetuning.off_policy.common_utils import utils
from resfit.rl_finetuning.utils.rb_transforms import MultiStepTransform
from resfit.rl_finetuning.utils.evaluate_dexmg import run_dexmg_evaluation
from resfit.rl_finetuning.chunk_residual.chunk_env_wrapper import ChunkResidualEnvWrapper


def to_uint8(obs: dict, image_keys):
    for k in image_keys:
        v = obs[k]
        if v.dtype != torch.uint8:
            obs[k] = (v.clamp(0, 1) * 255).round().to(torch.uint8)


def add_chunk_transition(*, obs, next_obs, combined_action, reward, done, info,
                         image_keys, lowdim_keys, online_rb):
    """单环境;构造与 train_residual_td3._add_transitions_to_buffer(L182-199)同构的 TensorDict。"""
    keys = set(image_keys) | set(lowdim_keys)
    curr = {k: obs[k][0].detach().cpu() for k in keys}
    nxt_src = next_obs
    if done[0] and info.get("final_obs", None) is not None:
        # 用终止观测(需含 base_action;若 final_obs 无 base_action 则补零)
        fo = info["final_obs"]
        nxt = {}
        for k in keys:
            if isinstance(fo, dict) and k in fo:
                nxt[k] = fo[k][0].detach().cpu() if fo[k].ndim > 1 else fo[k].detach().cpu()
            else:
                nxt[k] = next_obs[k][0].detach().cpu()
    else:
        nxt = {k: nxt_src[k][0].detach().cpu() for k in keys}
    to_uint8(curr, image_keys)
    to_uint8(nxt, image_keys)
    td = TensorDict({
        "obs": TensorDict(curr, batch_size=[]),
        "next": TensorDict({"obs": TensorDict(nxt, batch_size=[]),
                            "done": done[0].cpu(), "reward": reward[0].cpu()}, batch_size=[]),
        "action": combined_action[0].detach().cpu(),
        "_priority": torch.tensor(10.0, dtype=torch.float32),
    }, batch_size=[]).unsqueeze(0)
    online_rb.add(td)


def _maybe_inject_flow_actor(args, agent, repr_dim, patch_repr_dim, prop_dim, action_dim):
    """M1: no-op。M2.2 在此把 agent.actor 替换为 residual_flow actor 并重建 actor_opt。"""
    return


def build_base_policy(wandb_id: str, device: str):
    """从 wandb artifact 拉 ACT 基座(复用 ResFiT 现成加载路径)。

    实现提示:train_residual_td3.py 用 base_policy 加载工具从 cfg.base_policy.wandb_id
    拉 run_<id>_best:latest。实现时复用同一加载函数(见 train_residual_td3.py 顶部 import 的
    base policy loader),返回 eval 模式的 ACTPolicy。
    """
    raise NotImplementedError(
        "复用 train_residual_td3.py 里加载 ACT 基座的同一函数(按 wandb_id 拉 _best)。"
        "见 Task M1.2 Step 2 的接线说明。"
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--actor", choices=["raw", "flow"], default="raw")
    p.add_argument("--task", default="TwoArmBoxCleanup")
    p.add_argument("--base_wandb_id", default="dexmg-boxcleanup-bc/d59wny58")
    p.add_argument("--dataset", default="ankile/dexmg-two-arm-box-cleanup")
    p.add_argument("--chunk_length", type=int, default=20)
    p.add_argument("--action_scale", type=float, default=0.2)
    p.add_argument("--min_range_per_dim", type=float, default=0.1)
    p.add_argument("--total_env_steps", type=int, default=500_000)
    p.add_argument("--learning_starts", type=int, default=10_000)
    p.add_argument("--eval_every_env_steps", type=int, default=10_000)
    p.add_argument("--utd", type=int, default=4)
    p.add_argument("--n_step", type=int, default=3)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--buffer_size", type=int, default=200_000)
    p.add_argument("--batch_size", type=int, default=256)
    p.add_argument("--actor_lr", type=float, default=5e-6)
    p.add_argument("--critic_lr", type=float, default=1e-4)
    p.add_argument("--stddev", type=float, default=0.05)
    p.add_argument("--ae_ckpt", default=None)   # M2 flow 用
    p.add_argument("--eval_num_envs", type=int, default=8)
    p.add_argument("--eval_num_episodes", type=int, default=50)
    p.add_argument("--smoke", action="store_true", help="少量步数冒烟(不下真模型时配 --fake)")
    p.add_argument("--device", default="cuda")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    torch.manual_seed(args.seed)

    # --- 归一化器(从 dataset stats 建,与 AE / RL 同款)---
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from resfit.rl_finetuning.utils.normalization import ActionScaler, StateStandardizer
    meta = LeRobotDataset(args.dataset).meta
    action_scaler = ActionScaler.from_dataset_stats(
        meta.stats["action"], action_scale=args.action_scale,
        min_range_per_dim=args.min_range_per_dim, device=args.device)
    state_standardizer = StateStandardizer.from_dataset_stats(
        meta.stats["observation.state"], device=args.device)

    # --- 基座 + env ---
    base_policy = build_base_policy(args.base_wandb_id, args.device)
    vec_env = create_vectorized_env(env_name=args.task, num_envs=1, device=args.device)
    env = ChunkResidualEnvWrapper(vec_env, base_policy, action_scaler, state_standardizer,
                                  chunk_length=args.chunk_length)
    eval_vec = create_vectorized_env(env_name=args.task, num_envs=args.eval_num_envs,
                                     device=args.device)
    eval_env = ChunkResidualEnvWrapper(eval_vec, base_policy, action_scaler, state_standardizer,
                                       chunk_length=args.chunk_length)

    # --- 维度 ---
    image_keys = list(base_policy.config.image_features.keys())
    lowdim_keys = ["observation.state", "observation.base_action"]
    obs0, _ = env.reset()
    img_c, img_h, img_w = obs0[image_keys[0]].shape[1:]
    state_dim = obs0["observation.state"].shape[1]
    action_dim = env.action_dim * args.chunk_length     # = 480

    # --- agent(复用 QAgent,action_dim=480)---
    from resfit.rl_finetuning.config.residual_td3 import ResidualTD3BoxCleanConfig
    cfg = ResidualTD3BoxCleanConfig()
    cfg.agent.actor_lr = args.actor_lr
    cfg.agent.critic_lr = args.critic_lr
    cfg.agent.actor.action_scale = args.action_scale
    agent = QAgent(obs_shape=(img_c, img_h, img_w), prop_shape=(state_dim,),
                   action_dim=action_dim, rl_cameras=image_keys,
                   cfg=cfg.agent, residual_actor=True)

    # repr/patch 维(供 flow actor 构造,复用 QAgent 的算法)
    enc0 = agent.encoders[0]
    repr_dim = int(enc0.repr_dim) * len(image_keys)
    patch_repr_dim = int(enc0.patch_repr_dim)
    _maybe_inject_flow_actor(args, agent, repr_dim, patch_repr_dim, state_dim, action_dim)

    # --- buffer(复用 torchrl 构造,动作 480)---
    online_rb = TensorDictPrioritizedReplayBuffer(
        storage=LazyTensorStorage(max_size=args.buffer_size, device="cpu"),
        alpha=0.0, beta=0.0, eps=1e-6, priority_key="_priority",
        transform=MultiStepTransform(n_steps=args.n_step, gamma=args.gamma),
        pin_memory=True, prefetch=4, batch_size=args.batch_size)

    # --- 训练循环(x 轴=环境步;每 chunk 计入实际执行步数)---
    obs, _ = env.reset()
    env_steps = 0
    next_eval = 0
    best_sr = 0.0
    total = 2 * args.chunk_length if args.smoke else args.total_env_steps
    while env_steps <= total:
        with torch.no_grad(), utils.eval_mode(agent):
            action = agent.act(obs, eval_mode=False, stddev=args.stddev, cpu=False)  # [1,480] 残差
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = terminated | truncated
        add_chunk_transition(obs=obs, next_obs=next_obs, combined_action=info["scaled_action"],
                             reward=reward, done=done, info=info, image_keys=image_keys,
                             lowdim_keys=lowdim_keys, online_rb=online_rb)
        obs = next_obs
        executed = int(info.get("executed_steps", args.chunk_length))
        env_steps += args.chunk_length     # 近似:每 chunk 计 chunk_length 步(成功/早停略有出入)

        if env_steps >= args.learning_starts and len(online_rb) > args.batch_size:
            for i in range(args.utd):
                batch = online_rb.sample()
                update_actor = ((i + 1) % args.utd == 0)
                agent.update(batch, args.stddev, update_actor, bc_batch=None, ref_agent=agent)

        if env_steps >= next_eval:
            with torch.no_grad():
                m = run_dexmg_evaluation(env=eval_env, agent=agent,
                                         num_episodes=args.eval_num_episodes, device=args.device,
                                         global_step=env_steps, save_video=False,
                                         save_q_plots=False, run_name=f"chunk_{args.actor}",
                                         output_dir="outputs_chunk")
            sr = m["eval/success_rate"]
            best_sr = max(best_sr, sr)
            print(f"[env_steps {env_steps}] eval success_rate={sr:.3f} (best {best_sr:.3f})")
            next_eval += args.eval_every_env_steps
        if args.smoke:
            break

    print(f"done. best success_rate={best_sr:.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 接线说明(实现 `build_base_policy`)**

`build_base_policy` 当前抛 NotImplementedError。实现时:打开 `train_residual_td3.py`,找到它**加载 ACT 基座的函数**(按 `cfg.base_policy.wandb_id` 拉 `run_<id>_best:latest` 的那段,函数在文件顶部 import),把同样的调用复制进 `build_base_policy`,返回 `.to(device).eval()` 的 `ACTPolicy`。这是复用、不是改原文件。完成后删除 `raise NotImplementedError`。
- 另外:`run_dexmg_evaluation` 内部对 `eval_env.step(actions)` 期望 actions 是单步动作。**确认它在 chunk wrapper 下的行为**:它调 `agent.act`(返回 480 残差)→ `eval_env.step`(我们的 chunk wrapper,正确执行整段)。但 evaluate_dexmg.py 里 Q 值绘图/统计可能假设单步动作维;若报维度错,给 `run_dexmg_evaluation` 传 `save_q_plots=False`(已传)并在 Step 3 冒烟时定位。若仍不兼容,写一个**新的** chunk 版 eval 函数放本文件(不改 evaluate_dexmg.py)。

- [ ] **Step 3: 语法/导入冒烟**

Run: `cd /data2/RL/residual-offpolicy-rl && conda run -n residual python -c "import resfit.rl_finetuning.chunk_residual.train_chunk_residual as m; print('import ok')"`
Expected: 打印 `import ok`

- [ ] **Step 4: 提交**

```bash
cd /data2/RL/residual-offpolicy-rl
git add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py
git commit -m "feat(chunk_residual): chunk 级训练编排(--actor raw)"
```

---

## Task M1.3: 实跑 M1 chunk-raw 冒烟 + 短训(manual,go/no-go)

**Files:** 无新文件。

- [ ] **Step 1: 冒烟(2 个 chunk,验证不崩、维度通)**

```bash
cd /data2/RL/residual-offpolicy-rl
CUDA_VISIBLE_DEVICES=<空闲卡> MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
    --actor raw --smoke --eval_num_envs 2 --eval_num_episodes 2
```
Expected:能 reset、act 出 [1,480]、wrapper 执行整段、`add_chunk_transition` 不报错、一次 eval 跑通。若 eval 维度报错,按 M1.2 Step 2 处理(传 save_q_plots=False 已做;必要时写 chunk 版 eval)。

- [ ] **Step 2: 短训观察曲线(manual)**

去掉 `--smoke`,跑较短预算(如 `--total_env_steps 100000`),用空闲 GPU,后台 tmux:
```bash
CUDA_VISIBLE_DEVICES=<gpu> MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
    --actor raw --total_env_steps 100000 2>&1 | tee chunk_raw_boxcleanup.log
```

- [ ] **Step 3: 验收 go/no-go**

**判据**:能稳定训不崩,eval success_rate 从基座水平(BoxCleanup base ~0.1)**不显著下降**,最好出现上升趋势。这一步验证 chunk 骨架本身是对的(与 flow 无关)。
- 若崩/不学:先排查 reward 累积、buffer transition 维度、`info["scaled_action"]` 是否为 combined。这些问题与 flow 无关,必须在进 M2 前解决。

---

# 阶段 M2 — residual_flow actor + A/B

## Task M2.1: PyTorch 版 ResidualFlowActor

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/residual_flow_actor.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_residual_flow_actor.py`

参考源(JAX):`/data2/kai0/rlt/models_jax/residual_flow_actor.py`。接口必须对齐现成 `Actor.forward(obs, std) -> utils.TruncatedNormal`(见约定节、`actor.py:150-177`)。

- [ ] **Step 1: 写失败测试**

`resfit/rl_finetuning/chunk_residual/tests/test_residual_flow_actor.py`:
```python
import torch
from resfit.rl_finetuning.chunk_residual.action_autoencoder import ActionAutoencoder
from resfit.rl_finetuning.chunk_residual.residual_flow_actor import ResidualFlowActor

D, L, LAT = 4, 5, 8
FLAT = L * D
REPR, PATCH, PROP = 32, 16, 3
B = 6


def _make_actor():
    ae = ActionAutoencoder(action_dim=D, chunk_length=L, latent_dim=LAT,
                           hidden_dim=32, conv_layers=2)
    return ResidualFlowActor(repr_dim=REPR, patch_repr_dim=PATCH, prop_dim=PROP,
                             action_dim=FLAT, chunk_length=L, action_dim_per_step=D,
                             frozen_ae=ae, feature_dim=16, hidden_dim=32, num_layers=3,
                             latent_delta_scale=0.05, action_delta_clip=0.2)


def _fake_obs():
    return {
        "feat": torch.randn(B, REPR // PATCH, PATCH),     # [B, num_patch, patch_dim]
        "observation.state": torch.randn(B, PROP),
        "observation.base_action": torch.tanh(torch.randn(B, FLAT)),
    }


def test_returns_truncated_normal_with_residual_shape():
    actor = _make_actor().eval()
    dist = actor.forward(_fake_obs(), std=0.0)
    assert dist.mean.shape == (B, FLAT)
    s = dist.sample(clip=0.3)
    assert s.shape == (B, FLAT)


def test_zero_init_residual_is_near_zero():
    actor = _make_actor().eval()
    dist = actor.forward(_fake_obs(), std=0.0)
    # velocity 末层 zero-init → decoded_delta ≈ 0 → 残差均值≈0
    assert dist.mean.abs().max().item() < 1e-4


def test_grad_flows_to_velocity_not_ae():
    actor = _make_actor()
    obs = _fake_obs()
    dist = actor.forward(obs, std=0.0)
    dist.mean.pow(2).sum().backward()
    # AE 参数无梯度(冻结);velocity 末层有梯度(经 decode(z_corr) 回传)
    ae_grad = [p.grad for p in actor.frozen_ae.parameters() if p.grad is not None]
    assert len(ae_grad) == 0
    assert actor.velocity_out.weight.grad is not None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `cd /data2/RL/residual-offpolicy-rl && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_residual_flow_actor.py -v`
Expected: FAIL(`ModuleNotFoundError: ... residual_flow_actor`)

- [ ] **Step 3: 写实现**

`resfit/rl_finetuning/chunk_residual/residual_flow_actor.py`:
```python
"""PyTorch 版 VLA-anchored residual-flow actor(从 kai0/rlt JAX 版移植)。

接口对齐 ResFiT Actor:forward(obs, std) -> utils.TruncatedNormal,返回的均值是"残差"
(decoded_delta);QAgent 在 loss/执行时再加 base_action。
冻结 AE 作为子模块(requires_grad=False),梯度只流到速度网络(经 decode(z_corr) 回传)。
"""
from __future__ import annotations

import torch
from torch import nn

from resfit.rl_finetuning.off_policy.common_utils import utils


def smooth_clip(x: torch.Tensor, limit: float) -> torch.Tensor:
    limit = float(limit)
    if limit <= 0.0:
        return x
    return limit * torch.tanh(x / limit)


class ResidualFlowActor(nn.Module):
    def __init__(self, repr_dim: int, patch_repr_dim: int, prop_dim: int, action_dim: int,
                 chunk_length: int, action_dim_per_step: int, frozen_ae,
                 feature_dim: int = 128, hidden_dim: int = 512, num_layers: int = 3,
                 latent_delta_scale: float = 0.05, action_delta_clip: float = 0.2,
                 velocity_init_scale: float = 0.0, use_layer_norm: bool = True):
        super().__init__()
        self.flat_action_dim = action_dim
        self.chunk_length = chunk_length
        self.action_dim_per_step = action_dim_per_step
        self.latent_dim = frozen_ae.latent_dim
        self.latent_delta_scale = latent_delta_scale
        self.action_delta_clip = action_delta_clip

        # 冻结 AE
        self.frozen_ae = frozen_ae
        for p in self.frozen_ae.parameters():
            p.requires_grad = False

        # 图像特征压缩(仿 Actor 非 spatial_emb 分支:Linear repr_dim->feature_dim)
        comp = [nn.Linear(repr_dim, feature_dim)]
        if use_layer_norm:
            comp.append(nn.LayerNorm(feature_dim))
        comp.append(nn.ReLU())
        self.compress = nn.Sequential(*comp)

        # s_p 维 = feature_dim + prop_dim;velocity 输入 = z_ref + z_rl + LN(s_p) + a_ref
        self.s_p_dim = feature_dim + prop_dim
        self.s_p_norm = nn.LayerNorm(self.s_p_dim) if use_layer_norm else nn.Identity()
        in_dim = self.latent_dim + self.latent_dim + self.s_p_dim + self.flat_action_dim

        layers = []
        d = in_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(d, hidden_dim))
            if use_layer_norm:
                layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.SiLU())
            d = hidden_dim
        self.velocity_mlp = nn.Sequential(*layers)
        self.velocity_out = nn.Linear(hidden_dim, self.latent_dim)
        # zero-init 末层 → 起步残差=0,action=base
        if velocity_init_scale == 0.0:
            nn.init.zeros_(self.velocity_out.weight)
        else:
            nn.init.normal_(self.velocity_out.weight, std=velocity_init_scale)
        nn.init.zeros_(self.velocity_out.bias)

    def trainable_parameters(self):
        """供注入时建优化器:排除冻结 AE。"""
        return [p for n, p in self.named_parameters() if not n.startswith("frozen_ae.")]

    def train(self, mode: bool = True):
        super().train(mode)
        self.frozen_ae.eval()   # AE 始终 eval(仅 LayerNorm,行为不变,但保持语义清晰)
        return self

    def forward(self, obs: dict, std: float):
        a_ref = obs["observation.base_action"]                       # [B, FLAT]
        feat = self.compress(obs["feat"].flatten(1, -1))             # [B, feature_dim]
        s_p = torch.cat([feat, obs["observation.state"]], dim=-1)    # [B, s_p_dim]
        s_p = self.s_p_norm(s_p)

        z_ref = self.frozen_ae.encode(a_ref).detach()                # stopgrad
        a_ref_hat = self.frozen_ae.decode(z_ref).detach()            # stopgrad
        z_rl = torch.zeros_like(z_ref)                               # 确定性

        h = torch.cat([z_ref, z_rl, s_p, a_ref], dim=-1)
        h = self.velocity_mlp(h)
        velocity = self.velocity_out(h)
        latent_delta = torch.tanh(velocity) * self.latent_delta_scale
        z_corr = z_ref + latent_delta
        a_corr_hat = self.frozen_ae.decode(z_corr)                   # 梯度经此回传到 velocity
        decoded_delta = smooth_clip(a_corr_hat - a_ref_hat, self.action_delta_clip)
        return utils.TruncatedNormal(decoded_delta, std)
```

- [ ] **Step 4: 跑测试确认通过**

Run: `cd /data2/RL/residual-offpolicy-rl && conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_residual_flow_actor.py -v`
Expected: PASS(3 passed)

- [ ] **Step 5: 提交**

```bash
cd /data2/RL/residual-offpolicy-rl
git add resfit/rl_finetuning/chunk_residual/residual_flow_actor.py \
        resfit/rl_finetuning/chunk_residual/tests/test_residual_flow_actor.py
git commit -m "feat(chunk_residual): PyTorch residual_flow actor 移植 + 单元测试"
```

---

## Task M2.2: 把 residual_flow 注入训练脚本(--actor flow)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`(本计划新建的文件,非原仓库文件)

- [ ] **Step 1: 实现 `_maybe_inject_flow_actor`**

把 `train_chunk_residual.py` 中的 `_maybe_inject_flow_actor`(M1.2 的 no-op)替换为:
```python
def _maybe_inject_flow_actor(args, agent, repr_dim, patch_repr_dim, prop_dim, action_dim):
    if args.actor != "flow":
        return
    assert args.ae_ckpt is not None, "--actor flow 需要 --ae_ckpt 指向 M0 训好的 AE"
    from resfit.rl_finetuning.chunk_residual.action_autoencoder import ActionAutoencoder
    from resfit.rl_finetuning.chunk_residual.residual_flow_actor import ResidualFlowActor

    ckpt = torch.load(args.ae_ckpt, map_location=args.device)
    ae_cfg = ckpt["ae_config"]
    ae = ActionAutoencoder(**ae_cfg)
    ae.load_state_dict(ckpt["state_dict"])
    ae = ae.to(args.device).eval()

    flow_actor = ResidualFlowActor(
        repr_dim=repr_dim, patch_repr_dim=patch_repr_dim, prop_dim=prop_dim,
        action_dim=action_dim, chunk_length=args.chunk_length,
        action_dim_per_step=ae_cfg["action_dim"], frozen_ae=ae,
        feature_dim=agent.cfg.actor.feature_dim, hidden_dim=512, num_layers=3,
        latent_delta_scale=0.05, action_delta_clip=args.action_scale,  # 对齐 chunk-raw 的幅度
    ).to(args.device)

    agent.actor = flow_actor
    agent.actor_target = copy.deepcopy(flow_actor)
    agent.actor_opt = torch.optim.AdamW(flow_actor.trainable_parameters(), lr=args.actor_lr)
    print("[inject] residual_flow actor 已注入,actor_opt 仅含 velocity 网络参数")
```
要点:`action_delta_clip = args.action_scale`(=0.2)与 chunk-raw 的残差幅度对齐,保证 A/B 只比"残差怎么产生"。`feature_dim` 取自 `agent.cfg.actor.feature_dim`(默认 128),使 compress 维度与现成 actor 一致。

- [ ] **Step 2: 接口兼容冒烟(关键:验证注入后 agent.update 能跑一步)**

新增临时测试或直接用 `--smoke`(需真 env+基座+AE)。最小验证:注入后 `agent.act` 返回 [1,480]、一次 `agent.update` 不报错(actor_target 在 train 模式、AE 在 eval、梯度只进 velocity)。
Run(需 AE ckpt 与基座):
```bash
cd /data2/RL/residual-offpolicy-rl
CUDA_VISIBLE_DEVICES=<gpu> MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
    --actor flow --ae_ckpt resfit/rl_finetuning/chunk_residual/ckpt/ae_boxcleanup.pt \
    --smoke --learning_starts 0 --eval_num_envs 2 --eval_num_episodes 2
```
Expected:注入打印、act/step/update/eval 全跑通不报维度或梯度错。
- 注意:`QAgent.update` 内部对 `actor_target` 断言 training=True(q_agent.py:314)。`copy.deepcopy` 后 actor_target 默认 training=True(QAgent 构造后会 `self.train(True)`;注入发生在构造后,deepcopy 继承当时状态)。若断言失败,在注入后显式 `agent.actor_target.train(True)`。

- [ ] **Step 3: 提交**

```bash
cd /data2/RL/residual-offpolicy-rl
git add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py
git commit -m "feat(chunk_residual): 注入 residual_flow actor(--actor flow)"
```

---

## Task M2.3: A/B 实验 + 验收(manual,最终 go/no-go)

**Files:** 无新文件(运行 + 看曲线)。

- [ ] **Step 1: 跑 chunk-raw 基准(A)**

```bash
cd /data2/RL/residual-offpolicy-rl
CUDA_VISIBLE_DEVICES=<gpuA> MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
    --actor raw --seed 0 --total_env_steps 300000 2>&1 | tee chunk_raw_ab.log
```

- [ ] **Step 2: 跑 residual_flow(B,同种子同配置,只换 actor)**

```bash
cd /data2/RL/residual-offpolicy-rl
CUDA_VISIBLE_DEVICES=<gpuB> MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
conda run -n residual python -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
    --actor flow --ae_ckpt resfit/rl_finetuning/chunk_residual/ckpt/ae_boxcleanup.pt \
    --seed 0 --total_env_steps 300000 2>&1 | tee chunk_flow_ab.log
```
两条除 `--actor`(及 flow 的 `--ae_ckpt`)外完全一致:同 task/基座/种子/buffer/n_step/utd/lr/stddev/action_scale。

- [ ] **Step 3: 验收 go/no-go("有没有戏")**

对比两条 `eval success_rate vs env_steps` 曲线(日志 grep `eval success_rate`,或 wandb 若接入)。
- **硬门槛**:flow 稳定训练,不塌到基座成功率(~0.1)以下(zero-init 应保证起步=基座)。
- **有戏**:flow 追平或超过 raw;"追平但更平滑/更省样本"也算。
- **负结果**:干净 A/B 下 flow 赢不了 raw → "单任务 sim 上没戏"的明确结论(同样是有价值的交付)。
- 记录结论到一个简短结果笔记(可放 `docs/superpowers/` 下)。若有戏,后续可扩到 CanSort / 多种子(超出本计划范围)。

---

## Self-Review(写计划后自查,已执行)

- **Spec 覆盖**:M0=spec §4.4 AE;M1=spec §4.3 wrapper + §4.5 编排(零新 ML);M2=spec §4.2 flow actor + §5 公平 A/B + §7 判据;风险(spec §8)落在各 go/no-go 与 M2.2 Step 2 的断言提示。`offline_fraction` 在 spec §10 为旋钮,本计划取 0(纯 online)以免建 chunk 离线 buffer——已在 Architecture/约定节注明(对 A/B 两边一致,公平)。
- **占位符扫描**:`build_base_policy` 标了 NotImplementedError 并在 M1.2 Step 2 给出确切接线指引(复用 train_residual_td3 的基座加载函数)——这是有意的"复用点",非含糊占位;`_maybe_inject_flow_actor` 在 M1 为 no-op、M2.2 给出完整实现。其余步骤均含完整代码与确切命令。
- **类型一致**:`ResidualFlowActor` 构造参数(repr_dim/patch_repr_dim/prop_dim/action_dim/chunk_length/action_dim_per_step/frozen_ae/feature_dim/hidden_dim/num_layers/latent_delta_scale/action_delta_clip)在测试、实现、注入三处一致;`ActionAutoencoder(**ae_cfg)` 的 ae_cfg 字段与 train_action_ae.py 保存的 `ae_config` 字段一致(action_dim/chunk_length/latent_dim/hidden_dim/conv_layers);`forward(obs, std)` 与 `Actor` 一致,返回 `utils.TruncatedNormal`。
- **已知待实跑验证点**(不阻塞计划,实跑时定位):① `run_dexmg_evaluation` 在 chunk wrapper 下的兼容性(M1.2 Step 2 已给降级方案);② `env_steps` 用 chunk_length 近似计数(早停略有出入,不影响 A/B 公平,两边同样处理);③ 注入后 actor_target.training 断言(M2.2 Step 2 给出修法)。
