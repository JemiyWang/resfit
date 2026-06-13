# 在线 HIQL 子目标注入支持 act_feat Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 让 `train_chunk_residual --subgoal_conditioned` 能在线消费 act_feat 训出的 gc_value/high_actor(复用已加载的 ACT base_policy 每步在线提 530 维特征出子目标 z),从而可跑"等配置 three_piece_aligned_bp_bc01、仅 value 输入换 act_feat"的主训。

**Architecture:** `HiqlSubgoal` 按 gc_value ckpt 的 `state_mode` 分发:eef_piece 走现状;act_feat 持有一个由 base_policy 建的 `ActFeatureExtractor`,`subgoal_online(obs)` 每步提特征→标准化→high_actor 出 z。proprio 全栈统一 dataset-标准化(离线侧同步修正,需重建缓存)。eef_piece 路径逐位不变。

**Tech Stack:** Python, PyTorch, pytest;复用 `ActFeatureExtractor`(act_feature.py)、`load_act_feat_cache`(act_feat_cache.py)、`load_gc_value/load_high_actor`(含 state_mode/act_feat_signature/mean/std)、`StateStandardizer`。

参考 spec:`docs/superpowers/specs/2026-06-13-hiql-subgoal-online-act-feat-design.md`。

测试目录:`resfit/rl_finetuning/chunk_residual/tests/`。命令从仓库根 `/mnt/mnt/data/resfit` 用 `conda run -n residual` 跑。

---

## File Structure
- Modify `resfit/rl_finetuning/chunk_residual/train_hiql_value.py` — act_feat build 的 proprio 改 dataset-标准化(命门 B 离线侧)。
- Modify `resfit/rl_finetuning/chunk_residual/hiql_subgoal.py` — `HiqlSubgoal` 按 state_mode 分发(from_ckpts + subgoal_online + 持有 extractor)。
- Modify `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py` — subgoal init 分发 + `--act_feat_cache` flag + env_state_mode 放宽 + rollout 调用点传 obs。
- Create `resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_act_feat.py` — HiqlSubgoal act_feat stub 单测。
- Create `resfit/rl_finetuning/chunk_residual/verify_act_feat_online_consistency.py` — opt-in offline↔online 特征一致性 smoke。

---

## Task 1: 离线 proprio dataset-标准化(命门 B 离线侧)

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_hiql_value.py`(`read_per_demo_states` act_feat 分支)
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_read_per_demo_states_act_feat.py`

目标:act_feat build 时,把每个 raw_obs 的 18 维 proprio 用 dataset `StateStandardizer` 标准化后再进 extractor(与在线 `obs["observation.state"]` 同款)。注入 `_raw_obs_seqs` 的测试路径若传了 standardizer 就用,否则保持 raw(结构测试)。

- [ ] **Step 1: 写失败测试**(append 到 `tests/test_read_per_demo_states_act_feat.py`)

```python
def test_act_feat_build_standardizes_proprio():
    import numpy as np
    import torch
    from resfit.rl_finetuning.chunk_residual.train_hiql_value import read_per_demo_states

    class _StubExtractorPassthrough:
        def embed_batch(self, raw):
            # 输出 [B, 2 + 18]:前2维占位,后18维 = 传入的 proprio(便于断言它已被标准化)
            b = raw["observation.state"].shape[0]
            emb = torch.zeros(b, 2)
            return torch.cat([emb, torch.as_tensor(raw["observation.state"], dtype=torch.float32)], -1)

    class _StubStd:
        # standardize: (x - 1)/2,便于断言确实被应用
        def standardize(self, x):
            return (torch.as_tensor(x, dtype=torch.float32) - 1.0) / 2.0

    raw = [{"observation.images.agentview": torch.zeros(3, 3, 4, 4),
            "observation.state": torch.ones(3, 18) * 5.0}]   # raw proprio = 5
    seqs, std, stats = read_per_demo_states(
        "ignored.hdf5", "ds", state_mode="act_feat", num_demos=None,
        act_feat_cache=None, act_extractor=_StubExtractorPassthrough(),
        act_image_keys=["observation.images.agentview"], act_ckpt_id="ckptA",
        state_standardizer=_StubStd(), _raw_obs_seqs=raw)
    # build 出的 seqs 已被 act_feat (mean,std) 标准化;但我们验证“喂进 extractor 的 proprio 被 dataset 标准化”:
    # proprio (5) -> dataset-std (5-1)/2 = 2.0;之后整条再过 act_feat 标准化。这里改为验证 raw 特征(关掉 act_feat 标准化不便),
    # 故用 stub extractor 直接回传 proprio,并在 act_feat 标准化前比对:见下方实现保证 _raw_obs_seqs 的 proprio 被就地标准化。
    # 简化断言:再跑一次不传 standardizer(raw),两者 proprio 段应不同。
    seqs_raw, _, _ = read_per_demo_states(
        "ignored.hdf5", "ds", state_mode="act_feat", num_demos=None,
        act_feat_cache=None, act_extractor=_StubExtractorPassthrough(),
        act_image_keys=["observation.images.agentview"], act_ckpt_id="ckptA",
        state_standardizer=None, _raw_obs_seqs=[{
            "observation.images.agentview": torch.zeros(3, 3, 4, 4),
            "observation.state": torch.ones(3, 18) * 5.0}])
    # 两条都各自被 act_feat (mean,std) 标准化,但因输入 proprio 不同(2.0 vs 5.0),标准化后 std 不同 → 不全等
    assert not np.allclose(seqs[0], seqs_raw[0])
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_read_per_demo_states_act_feat.py::test_act_feat_build_standardizes_proprio -v`
Expected: FAIL（`read_per_demo_states` 还没有 `state_standardizer` 参数 → TypeError）。

- [ ] **Step 3: 实现**

3a. `read_per_demo_states` 签名加 `state_standardizer=None` 参数(放在现有 act_feat 参数后):
```python
def read_per_demo_states(hdf5_path, dataset_id, state_mode="eef", num_demos=None,
                         device="cpu", cache_path=None,
                         act_feat_cache=None, act_extractor=None,
                         act_image_keys=None, act_ckpt_id=None, act_proprio_key="observation.state",
                         pooling="mean", state_standardizer=None, _raw_obs_seqs=None):
```

3b. 在 act_feat build 分支(`assert act_extractor is not None` 之后、`raw_feat = [...]` 之前)插入 proprio 标准化:
```python
        raw_seqs = _raw_obs_seqs if _raw_obs_seqs is not None else _build_raw_obs_seqs(
            hdf5_path, act_image_keys, act_proprio_key, num_demos)
        # 命门 B:proprio 全栈 dataset-标准化(与在线 obs.state 同款)。
        std = state_standardizer
        if std is None and _raw_obs_seqs is None:   # 真 build 且未显式传 → 从 dataset stats 建
            std = StateStandardizer.from_dataset_stats(
                LeRobotDatasetMetadata(dataset_id).stats["observation.state"], device="cpu")
        if std is not None:
            for ro in raw_seqs:
                ro[act_proprio_key] = std.standardize(
                    torch.as_tensor(ro[act_proprio_key], dtype=torch.float32))
        raw_feat = [act_extractor.embed_batch(ro).cpu().numpy().astype(np.float32) for ro in raw_seqs]
```
(`StateStandardizer` 和 `LeRobotDatasetMetadata` 已在文件顶部 import;`torch` 也在顶部。)

- [ ] **Step 4: 跑测试 + 回归**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_read_per_demo_states_act_feat.py -v`
Expected: 新测试 + 既有 `test_act_feat_branch_builds_then_caches`(不传 standardizer + 注入 raw → 保持 raw)+ `test_build_raw_obs_seqs_image_preprocessing` 全 PASS。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/train_hiql_value.py resfit/rl_finetuning/chunk_residual/tests/test_read_per_demo_states_act_feat.py
git commit -m "feat(act_feat): offline build standardizes proprio with dataset stats (online 同源, 命门B)"
```

---

## Task 2: `HiqlSubgoal` 按 state_mode 分发

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/hiql_subgoal.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_act_feat.py`(新建)

目标:`HiqlSubgoal` 支持 act_feat:构造时按 state_mode 持有 rel_stats(eef_piece)或 extractor+feat_stats(act_feat);`subgoal_online` 分发。`from_ckpts` 去 eef_piece-only 断言、act_feat 时用 base_policy 建 extractor。先读现有 `hiql_subgoal.py` 全文与现有 `tests/test_hiql_subgoal_wiring.py`。

- [ ] **Step 1: 写失败测试**(新建 `tests/test_hiql_subgoal_act_feat.py`)

```python
import numpy as np
import torch
from torch import nn

from resfit.rl_finetuning.chunk_residual.hiql_subgoal import HiqlSubgoal


class _StubHighActor(nn.Module):
    state_dim = 530
    rep_dim = 10
    def forward(self, s, g):
        b = s.shape[0]
        class D:  # 分布占位,.mean 返回 [b, rep_dim]
            mean = torch.zeros(b, 10)
        return D()


class _StubExtractor:
    def embed_batch(self, obs):
        b = obs["observation.state"].shape[0]
        return torch.cat([torch.zeros(b, 512), torch.as_tensor(obs["observation.state"], dtype=torch.float32)], -1)


def test_subgoal_online_act_feat_dispatch():
    ha = _StubHighActor()
    goal = np.zeros(530, np.float32)
    sg = HiqlSubgoal(gc_value=None, high_actor=ha, goal=goal, device="cpu", renorm_subgoal=False,
                     state_mode="act_feat", extractor=_StubExtractor(),
                     feat_stats=(np.zeros(530, np.float32), np.ones(530, np.float32)))
    obs = {"observation.images.agentview": torch.zeros(3, 3, 4, 4),
           "observation.state": torch.ones(3, 18)}
    z = sg.subgoal_online(obs)            # act_feat:不需要 rel
    assert z.shape == (3, 10)
    assert torch.isfinite(z).all()
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_act_feat.py -v`
Expected: FAIL（`HiqlSubgoal.__init__` 不接受 `state_mode/extractor/feat_stats`）。

- [ ] **Step 3: 实现**(改 `hiql_subgoal.py`)

3a. `__init__` 改为 mode-aware(把原 `goal30/rel_mean/rel_std` 位置参数改成 `goal` + keyword 模式位;**更新 from_ckpts 的调用**):
```python
class HiqlSubgoal:
    def __init__(self, gc_value, high_actor, goal, device="cpu", renorm_subgoal=False,
                 *, state_mode="eef_piece", rel_stats=None, extractor=None, feat_stats=None):
        self.state_mode = state_mode
        self.ha = high_actor.to(device).eval()
        for p in self.ha.parameters():
            p.requires_grad_(False)
        if gc_value is not None:
            self.vf = gc_value.to(device).eval()
            for p in self.vf.parameters():
                p.requires_grad_(False)
            self.rep_dim = gc_value.rep_dim
        else:
            self.vf = None
            self.rep_dim = high_actor.rep_dim
        self.device = device
        self.renorm_subgoal = renorm_subgoal
        self.goal = torch.as_tensor(np.asarray(goal), dtype=torch.float32, device=device).reshape(-1)
        if state_mode == "eef_piece":
            assert rel_stats is not None, "eef_piece 须给 rel_stats"
            self.rel_mean = torch.as_tensor(np.asarray(rel_stats[0]), dtype=torch.float32, device=device)
            self.rel_std = torch.as_tensor(np.asarray(rel_stats[1]), dtype=torch.float32, device=device)
        elif state_mode == "act_feat":
            assert extractor is not None and feat_stats is not None, "act_feat 须给 extractor + feat_stats"
            self.extractor = extractor
            self.feat_mean = torch.as_tensor(np.asarray(feat_stats[0]), dtype=torch.float32, device=device)
            self.feat_std = torch.as_tensor(np.asarray(feat_stats[1]), dtype=torch.float32, device=device)
        else:
            raise ValueError(f"unknown state_mode: {state_mode}")
```
(注:`subgoal_waypoint` 用 `self.vf.phi`,只在 eef_piece 离线用;act_feat 不调它。)

3b. `subgoal_online` 分发(替换原方法):
```python
    @torch.no_grad()
    def subgoal_online(self, obs, rel_raw=None):
        """obs:eef_piece 传 {observation.state:[B,18]std}(+rel_raw);act_feat 传含 images+state 的 obs dict。"""
        if self.state_mode == "eef_piece":
            s = self.build_state30(obs["observation.state"] if isinstance(obs, dict) else obs, rel_raw)
        else:  # act_feat
            feat = self.extractor.embed_batch(obs)           # [B,530] raw
            s = (feat.to(self.device) - self.feat_mean) / self.feat_std
        g = self.goal.unsqueeze(0).expand(s.shape[0], -1)
        z = self.ha(s, g).mean
        if self.renorm_subgoal:
            z = z / (z.norm(dim=-1, keepdim=True) + 1e-8) * (self.rep_dim ** 0.5)
        return z
```
(`build_state30` 保留不动;`goal30`→`self.goal`。)

3c. `from_ckpts` 去 eef_piece-only 断言 + act_feat 分支:
```python
    @classmethod
    def from_ckpts(cls, gc_value_ckpt, high_actor_ckpt, *, goal, device="cpu",
                   renorm_subgoal=False, base_policy=None):
        gc, info = load_gc_value(gc_value_ckpt, map_location=device)
        ha, _ = load_high_actor(high_actor_ckpt, map_location=device)
        sm = info["state_mode"]
        assert sm in ("eef_piece", "act_feat"), f"分层路 gc_value state_mode 须 eef_piece/act_feat,got {sm}"
        assert ha.rep_dim == gc.rep_dim and ha.state_dim == gc.state_dim, "rep/state_dim 不一致"
        if sm == "eef_piece":
            return cls(gc, ha, goal, device=device, renorm_subgoal=renorm_subgoal,
                       state_mode="eef_piece", rel_stats=(info["rel_piece_mean"], info["rel_piece_std"]))
        # act_feat
        assert base_policy is not None, "act_feat 在线子目标须传 base_policy 建特征器"
        from resfit.rl_finetuning.chunk_residual.act_feature import ActFeatureExtractor
        sig = info["act_feat_signature"] or {}
        ext = ActFeatureExtractor(base_policy, image_keys=sig["image_keys"],
                                  proprio_key=sig.get("proprio_key", "observation.state"),
                                  pooling=sig.get("pooling", "mean"))
        return cls(gc, ha, goal, device=device, renorm_subgoal=renorm_subgoal,
                   state_mode="act_feat", extractor=ext, feat_stats=(info["mean"], info["std"]))
```

3d. `representative_goal30` 加泛化别名(文件末尾):`representative_goal = representative_goal30`(对任意维 seqs 取 medoid,逻辑已通用)。

3e. **更新现有 from_ckpts 的下游/调用**:`train_chunk_residual` 现传 `goal30=...`,Task 3 会改为 `goal=...`;本任务只改 `hiql_subgoal.py` 自身,确保 `goal=` 关键字可用。

- [ ] **Step 4: 跑测试 + 回归**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_act_feat.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_wiring.py -v`
Expected: 新 act_feat 测试 PASS;既有 subgoal wiring 测试 —— 若它直接构造 `HiqlSubgoal(...)` 或调 `from_ckpts(goal30=...)`,按新签名(`goal=`、`rel_stats=`)更新这些调用使其通过(eef_piece 行为不变)。报告改了哪些调用。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/hiql_subgoal.py resfit/rl_finetuning/chunk_residual/tests/test_hiql_subgoal_act_feat.py
git commit -m "feat(hiql_subgoal): mode dispatch (eef_piece|act_feat); act_feat reuses base_policy extractor"
```

---

## Task 3: `train_chunk_residual` 在线接线

**Files:**
- Modify: `resfit/rl_finetuning/chunk_residual/train_chunk_residual.py`
- Test: `resfit/rl_finetuning/chunk_residual/tests/test_act_feat_cli_wiring.py`(扩)

先 READ `train_chunk_residual.py` 的:`build_parser`(subgoal flags ~390-400)、subgoal init(634-658)、env_state_mode/eval_state_mode 设定(537-557)、rollout 调用点(`subgoal.subgoal_online(...)`,约 650-695)。

- [ ] **Step 1: 写失败测试**(append 到 `tests/test_act_feat_cli_wiring.py`)

```python
def test_chunk_residual_parser_has_act_feat_cache():
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_parser
    args = build_parser().parse_args(["--task", "TwoArmThreePieceAssembly", "--dataset", "d"])
    assert hasattr(args, "act_feat_cache") and args.act_feat_cache is None
```

- [ ] **Step 2: 跑测试确认失败**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_act_feat_cli_wiring.py -k act_feat_cache -v`
Expected: FAIL（无 `--act_feat_cache`）。

- [ ] **Step 3: 实现**

3a. `build_parser` 加 flag(放在 subgoal 相关 flags 附近):
```python
    p.add_argument("--act_feat_cache", default=None,
                   help="act_feat 子目标:530 序列缓存(算 goal + 同源签名校验);仅 act_feat gc_value 用")
```

3b. subgoal init(634-658)改为按 gc_value ckpt 的 state_mode 分发。先 peek state_mode:
```python
    subgoal = None
    if args.subgoal_conditioned:
        assert args.actor == "raw", "subgoal-conditioning 第一版只支持 --actor raw"
        assert args.gc_value_ckpt and args.high_actor_ckpt, "需 --gc_value_ckpt 与 --high_actor_ckpt"
        from resfit.rl_finetuning.chunk_residual.hiql_subgoal import HiqlSubgoal, representative_goal
        from resfit.rl_finetuning.chunk_residual.hiql_gc_value import load_gc_value
        _, _gc_info = load_gc_value(args.gc_value_ckpt, map_location="cpu")
        _sm = _gc_info["state_mode"]
        if _sm == "act_feat":
            assert args.act_feat_cache, "act_feat 子目标需 --act_feat_cache(算 goal530 + 同源)"
            from resfit.rl_finetuning.chunk_residual.act_feat_cache import load_act_feat_cache
            _seqs, _stats, _cache_sig = load_act_feat_cache(args.act_feat_cache)
            # 同源:cache 签名核心字段 == gc_value act_feat_signature
            _gv_sig = _gc_info.get("act_feat_signature") or {}
            for k in ("act_ckpt_id", "image_keys", "proprio_key", "pooling"):
                assert _cache_sig.get(k) == _gv_sig.get(k), \
                    f"act_feat cache 与 gc_value 签名不符 [{k}]: {_cache_sig.get(k)} vs {_gv_sig.get(k)}"
            goal = representative_goal(_seqs)
            assert goal.shape[0] == _gc_info["mean"].shape[0], "goal 维度须 == gc_value state_dim(530)"
            subgoal = HiqlSubgoal.from_ckpts(args.gc_value_ckpt, args.high_actor_ckpt,
                                             goal=goal, device=args.device,
                                             renorm_subgoal=args.renorm_subgoal, base_policy=base_policy)
        else:  # eef_piece:现状不变
            assert args.subgoal_state30_cache or args.offline_dataset_path, \
                "eef_piece 子目标需 --subgoal_state30_cache 或 --offline_dataset_path"
            from resfit.rl_finetuning.chunk_residual.state30_cache import load_or_build_state30
            _seqs30 = load_or_build_state30(args.offline_dataset_path, args.dataset,
                                            args.offline_num_demos, args.subgoal_state30_cache)
            goal30 = representative_goal(_seqs30)
            assert goal30.shape[0] == 30, f"goal30 须 30 维,got {goal30.shape[0]}"
            subgoal = HiqlSubgoal.from_ckpts(args.gc_value_ckpt, args.high_actor_ckpt,
                                             goal=goal30, device=args.device,
                                             renorm_subgoal=args.renorm_subgoal)
        print(f"[hiql-subgoal] on; mode={_sm} rep_dim={subgoal.rep_dim} renorm={args.renorm_subgoal}")
```

3c. env_state_mode / eval_state_mode:act_feat 不需要 eef_piece 透出 rel。把 537-557 段改为按 gc_value mode 决定。**最稳做法**:先 peek `_sm`(把上面的 `load_gc_value` peek 提到 env 构造之前,或单独 peek 一次),act_feat → `env_state_mode = eval_state_mode = "eef"`;eef_piece → 现状 `"eef_piece"`。同时把现有 `assert env_state_mode == "eef_piece"`(637)放宽为:`act_feat` 允许 `eef`。
```python
    # 在 env 构造前 peek(若上面已 peek,可复用 _sm)
    _subgoal_sm = None
    if args.subgoal_conditioned:
        from resfit.rl_finetuning.chunk_residual.hiql_gc_value import load_gc_value as _lgv
        _subgoal_sm = _lgv(args.gc_value_ckpt, map_location="cpu")[1]["state_mode"]
    if args.subgoal_conditioned and _subgoal_sm == "eef_piece":
        env_state_mode = "eef_piece"; eval_state_mode = "eef_piece"
    else:
        env_state_mode = "eef"; eval_state_mode = "eef"
```
(把现有 `env_state_mode = "eef_piece"` / `eval_state_mode = ...` 两处替换为上述;并删除/放宽 637 行的 `assert env_state_mode == "eef_piece"`。)

3d. rollout 调用点:把 `subgoal.subgoal_online(obs["observation.state"], cur_rel)` 改成传完整 obs:
```python
        obs["observation.subgoal"] = subgoal.subgoal_online(obs, cur_rel).to(
            obs["observation.state"].device)
```
同样 next_obs 处:`subgoal.subgoal_online(next_obs, next_rel)`。eef_piece 分支 `subgoal_online` 内部用 `obs["observation.state"]`(dict),行为不变;act_feat 用 obs 的 images+state。

- [ ] **Step 4: 跑测试 + 回归**

Run: `conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/test_act_feat_cli_wiring.py -v`
Expected: parser 测试 PASS。
回归:`conda run -n residual python -m pytest resfit/rl_finetuning/chunk_residual/tests/ -k "subgoal or hiql or smoke" -q`(确保 eef_piece subgoal wiring 不破)。注:train_chunk_residual 的 import 较重,若该测试文件已能 import 它即可;若 import 失败报 BLOCKED + 具体错。

- [ ] **Step 5: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/train_chunk_residual.py resfit/rl_finetuning/chunk_residual/tests/test_act_feat_cli_wiring.py
git commit -m "feat(train_chunk_residual): online act_feat subgoal dispatch + --act_feat_cache + env_state_mode=eef"
```

---

## Task 4: offline↔online 一致性 smoke(命门 A/B 硬门,opt-in)

**Files:**
- Create: `resfit/rl_finetuning/chunk_residual/verify_act_feat_online_consistency.py`

目标:同一条 demo 初态,offline 从 hdf5(经 Task1 标准化)提的 530 特征 vs online 从 env obs(reset 到同 task)提的 530 特征,断言 allclose;并打印 env obs 图 dtype/shape/range 确认 == CHW float[0,1]。非 TDD,是探针(需 GPU+env)。

- [ ] **Step 1: 写 smoke 脚本**

```python
"""opt-in:验证 act_feat offline(hdf5)↔online(env obs)特征一致 + env 图格式。
跑:CUDA_VISIBLE_DEVICES=0 conda run -n residual python -m \
    resfit.rl_finetuning.chunk_residual.verify_act_feat_online_consistency \
    --base resfit/out/piecce/best --hdf5 resfit/dataset/two_arm_three_piece_assembly.hdf5 \
    --dataset ankile/dexmg-two-arm-three-piece-assembly --task TwoArmThreePieceAssembly
"""
import argparse
from pathlib import Path
import h5py
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", required=True); ap.add_argument("--hdf5", required=True)
    ap.add_argument("--dataset", required=True); ap.add_argument("--task", required=True)
    args = ap.parse_args()
    from resfit.lerobot.utils.load_policy import load_policy
    from resfit.rl_finetuning.chunk_residual.act_feature import ActFeatureExtractor
    from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import STATE18_KEYS, assemble_state18
    from resfit.rl_finetuning.utils.normalization import StateStandardizer
    from lerobot.common.datasets.lerobot_dataset import LeRobotDatasetMetadata

    cand = Path(args.base) / "policy"
    act = load_policy(cand if cand.is_dir() else Path(args.base)); act.eval()
    keys = list(act.config.image_features.keys())
    ext = ActFeatureExtractor(act, image_keys=keys)
    std = StateStandardizer.from_dataset_stats(
        LeRobotDatasetMetadata(args.dataset).stats["observation.state"], device="cpu")

    # offline:demo_0 第0帧(经 Task1 同款预处理:图 HWC->CHW/255;proprio dataset-std)
    with h5py.File(args.hdf5, "r") as f:
        g = f["data"][sorted(f["data"].keys())[0]]
        ro = {}
        for k in keys:
            name = k.replace("observation.images.", "")
            img = torch.as_tensor(g[f"obs/{name}_image"][0:1])
            ro[k] = img.float().div(255.0).permute(0, 3, 1, 2)
        obs_arrays = {kk: g[f"obs/{kk}"][0:1] for kk, _ in STATE18_KEYS}
        ro["observation.state"] = std.standardize(torch.as_tensor(assemble_state18(obs_arrays), dtype=torch.float32))
    feat_off = ext.embed_batch(ro).cpu().numpy()

    # online:env reset(同 task),取第一帧 obs(env 已 CHW float + state 已 std?见打印)
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_base_policy  # 复用 env 构造? 见下
    # 用最小 env 构造:create_vectorized_env + ChunkResidualEnvWrapper(参考 train_chunk_residual env 构造段)
    # 这里只取 reset 后第一帧 aug obs。实现时按 train_chunk_residual 的 env 构造复制最小路径。
    # ---- 打印 env obs 图格式 ----
    # for k in keys: print(k, aug[k].dtype, tuple(aug[k].shape), float(aug[k].min()), float(aug[k].max()))
    # feat_on = ext.embed_batch({**{k: aug[k] for k in keys}, "observation.state": aug["observation.state"]}).cpu().numpy()
    # print("offline vs online allclose:", np.allclose(feat_off, feat_on, atol=1e-3))
    print("offline feat shape:", feat_off.shape, "finite:", bool(np.isfinite(feat_off).all()))
    # NOTE: online env 构造段在实现时补全(参考 train_chunk_residual create_vectorized_env + ChunkResidualEnvWrapper);
    #       初态可能与 demo_0 第0帧不完全一致(随机化),故一致性以"同一 obs 喂两条路特征相同"为准:
    #       更稳:取 env reset 后的 aug obs,既喂 offline 风格(若来自 hdf5 则需对齐),也喂 ext —— 实现时锁定对齐口径。


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 补全 online env 段并跑(opt-in,GPU)**

实现 online env 构造(参考 `train_chunk_residual` 的 `create_vectorized_env` + `ChunkResidualEnvWrapper` 最小路径),reset 取第一帧 `aug` obs:① 打印每路图 dtype/shape/range,确认 CHW float[0,1];② 对**同一帧 obs** 同时走"直接喂 ext"验证确定性;③ 核心一致性判据:**env obs 的图与 proprio 格式 == offline 预处理后格式**(图 CHW float[0,1];state 已 dataset-std)。
Run: `CUDA_VISIBLE_DEVICES=0 conda run -n residual python -m resfit.rl_finetuning.chunk_residual.verify_act_feat_online_consistency --base resfit/out/piecce/best --hdf5 resfit/dataset/two_arm_three_piece_assembly.hdf5 --dataset ankile/dexmg-two-arm-three-piece-assembly --task TwoArmThreePieceAssembly`
Expected: 打印 env 图 = CHW float[0,1]、state 已 std;offline 特征有限。**若 env 图不是 CHW float[0,1] → 在线 extractor 入口补同款变换(改 hiql_subgoal act_feat 分支),再验。**

- [ ] **Step 3: Commit**

```bash
git add resfit/rl_finetuning/chunk_residual/verify_act_feat_online_consistency.py
git commit -m "spike: offline<->online act_feat feature consistency + env image-format check"
```

---

## Self-Review 结论
- **Spec 覆盖**:§3.1 HiqlSubgoal 分发→Task2;§3.2 train_chunk_residual 接线+flag+env_state_mode+rollout→Task3;§3.3 离线 proprio dataset-std→Task1;§3.4 一致性 smoke→Task4;§5 测试→各 Task TDD + Task4 smoke。
- **占位扫描**:Task4 的 online env 构造段标注"实现时按 train_chunk_residual 复制最小路径"——这是 spike 探针、需真 env,允许实现时对齐(已给出判据与命令);其余 code step 均给真实代码。
- **类型一致**:`HiqlSubgoal.__init__` 新签名(`goal`,keyword `state_mode/rel_stats/extractor/feat_stats`)在 Task2 定义、Task3 `from_ckpts(goal=...)` 调用一致;`representative_goal` 在 Task2 定义、Task3 用;`--act_feat_cache` 在 Task3 加、与 train_hiql_* 同名。
- **范围**:单一计划;eef_piece 逐位不变;pi0/LIBERO 范围外。
- **风险**:Task4 是硬门——没过一致性 smoke 不跑 500k。
