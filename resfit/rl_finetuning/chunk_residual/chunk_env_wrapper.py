"""chunk 级残差 env wrapper(替代 BasePolicyVecEnvWrapper,不修改原文件)。

约定(与 ResFiT 一致):
- base_action 以归一化 [-1,1] chunk 存于 obs["observation.base_action"](展平 L*D)。
- step 收到的是归一化残差 chunk(展平 L*D);replan 模式 combined = clamp(base+residual,-1,1),
  queue 模式不 clamp(与原版 BasePolicyVecEnvWrapper 行为一致);
  reshape → unscale → 在底层 env 上开环逐步执行;累积奖励;中途 done 则提前结束。
- info["scaled_action"] = combined(展平),供 buffer 存储(与原 wrapper L145 一致)。
仅支持 num_envs==1(训练端 assert 限制)。
"""
from __future__ import annotations

import numpy as np
import torch

from resfit.rl_finetuning.chunk_residual.chunk_act_base import get_action_chunk
from resfit.rl_finetuning.chunk_residual.hiql_potential import potential_shaping


def staged_bonus(start_stage: int, end_stage: int, bonus: float) -> float:
    """本 chunk 内 stage 推进量的 additive 加分(非负;bonus=0 即关闭)。"""
    return bonus * max(0, int(end_stage) - int(start_stage))


def shaping_reward(start_stage: int, end_stage: int, *,
                   mode: str, bonus: float, gamma: float, done: bool) -> float:
    """三模式 stage 奖励整形(Φ=stage)。

    - none:恒 0(eval 用,指标纯净)。
    - staged:净加 additive,= staged_bonus(逐位不变);不看 gamma/done。
    - potential:PBS(Ng 1999),F = bonus·(γ·Φ(s') − Φ(s)),不改最优策略。
      done 时 Φ(s')=0(终止置零,与 critic 的 Q-target bootstrap 一致)→ F = −bonus·Φ(s)。
    """
    if mode == "none":
        return 0.0
    if mode == "staged":
        return staged_bonus(start_stage, end_stage, bonus)
    if mode == "potential":
        return potential_shaping(int(start_stage), int(end_stage),
                                 bonus=bonus, gamma=gamma, done=done)
    raise ValueError(f"unknown reward_shaping mode: {mode!r}")


def resolve_shaping_mode(reward_shaping, staged_reward: bool) -> str:
    """解析奖励整形模式:canonical --reward_shaping 优先;--staged_reward 为别名。

    reward_shaping: --reward_shaping 的值(未传时为 None);staged_reward: --staged_reward 布尔。
    """
    if reward_shaping is not None:
        return reward_shaping
    return "staged" if staged_reward else "none"


class ChunkResidualEnvWrapper:
    def __init__(self, vec_env, base_policy, action_scaler, state_standardizer,
                 chunk_length: int, stage_reward_bonus: float = 0.0,
                 reward_shaping_mode: str = "none", gamma: float = 0.99,
                 base_action_mode: str = "replan", potential=None,
                 potential_feature_encoder=None):
        self.vec_env = vec_env
        self.base_policy = base_policy
        self.action_scaler = action_scaler
        self.state_standardizer = state_standardizer
        self.chunk_length = chunk_length
        self.stage_reward_bonus = stage_reward_bonus
        self.reward_shaping_mode = reward_shaping_mode
        self.gamma = gamma
        self.base_action_mode = base_action_mode
        self.potential = potential          # None=Φ用stage(现状);HiqlPotential=Φ用V(state)
        self.potential_feature_encoder = potential_feature_encoder
        if potential is not None and getattr(potential, "state_mode", "eef") == "act_feat" \
                and potential_feature_encoder is None:
            raise ValueError("act_feat potential requires potential_feature_encoder")
        self._start_state_std = None         # 本 chunk 起点的标准化 state(potential 模式用)
        self._start_phi_state = None
        self._start_rel_piece = None         # 本 chunk 起点的 raw rel_piece(eef_piece object-aware Φ 用)
        assert chunk_length >= 1, "chunk_length must be >= 1"
        assert base_action_mode in ("replan", "queue"), \
            f"unknown base_action_mode: {base_action_mode!r}"
        if base_action_mode == "queue":
            assert chunk_length == 1, "base_action_mode='queue' 仅支持 chunk_length==1"
        self.action_dim = vec_env.action_space.shape[-1]
        self.num_envs = getattr(vec_env, "num_envs", 1)
        self.flat_dim = chunk_length * self.action_dim
        self._last_base_flat = None
        # 取整段 chunk 的入口(测试可在构造后直接 monkeypatch self._get_chunk)
        self._get_chunk = getattr(base_policy, "get_action_chunk", None)
        # stage 由 worker 经 info["stage_id"] 透出(见 dexmg.py);这里做 episode 内 max-so-far 闩锁
        self._stage = 0          # 闩锁(max-so-far):供 staged/potential reward(防刷分)
        self._stage_now = 0      # 瞬时:供 obs.stage_id(解耦,handoff §2,避免 stage 桶被退化样本污染)
        # 轻量诊断(跨 episode 累计):闩锁桶纯度。瞬时阶段 s < 闩锁 L = 掉件/回退,
        # 该步样本会被标成 stage L 却已退化 → 量化 stage 桶有多脏。不影响训练逻辑。
        self._stage_total_steps = 0
        self._stage_regress_steps = 0
        self._steps_by_latch: dict[int, int] = {}
        self._regress_by_latch: dict[int, int] = {}

    # ---- 取基座动作并归一化到 [-1,1] 展平 ----
    def _base_chunk_flat(self, raw_obs):
        if self.base_action_mode == "queue":                     # cl=1:走 ACT 原生 action queue
            base_action = self.base_policy.select_action(raw_obs)  # [B, D]
            base_n = self.action_scaler.scale(base_action)
            return base_n.reshape(base_n.shape[0], -1)            # [B, D] (=L*D, L=1)
        # replan(默认):每边界重跑模型取前 chunk_length 步
        if self._get_chunk is not None:                          # fake/可替换路径
            chunk_raw = self._get_chunk(raw_obs, self.chunk_length)
        else:
            chunk_raw = get_action_chunk(self.base_policy, raw_obs, self.chunk_length)
        chunk_n = self.action_scaler.scale(chunk_raw)            # [B,L,D] -> [-1,1]
        return chunk_n.reshape(chunk_n.shape[0], -1)             # [B, L*D]

    def _augment(self, raw_obs, base_flat):
        aug = dict(raw_obs)
        aug["observation.base_action"] = base_flat
        aug["observation.state"] = self.state_standardizer.standardize(raw_obs["observation.state"])
        b = base_flat.shape[0]
        aug["observation.stage_id"] = torch.full((b, 1), float(self._stage_now))  # 瞬时(解耦)
        return aug

    def _augment_bootstrap_final_observation(
            self, info: dict, terminated: torch.Tensor,
            truncated: torch.Tensor) -> dict:
        info = dict(info)
        if not bool(info.get("bootstrap_on_truncation", False)):
            return info
        if bool(terminated.any()):
            raise RuntimeError(
                "bootstrap_on_truncation cannot be combined with terminated=True")
        if not bool(truncated.any()):
            raise RuntimeError(
                "bootstrap_on_truncation requires truncated=True")
        if "final_observation" not in info:
            raise RuntimeError(
                "bootstrap_on_truncation requires final_observation")
        if "_wm_final_base_chunk" not in info:
            raise RuntimeError(
                "bootstrap_on_truncation requires _wm_final_base_chunk")

        raw_final_obs = info["final_observation"]
        final_state = raw_final_obs["observation.state"]
        final_base_chunk = torch.as_tensor(
            info["_wm_final_base_chunk"],
            dtype=torch.float32,
            device=final_state.device,
        )
        if final_base_chunk.ndim == 2:
            final_base_chunk = final_base_chunk.unsqueeze(0)
        expected = (self.num_envs, self.chunk_length, self.action_dim)
        if tuple(final_base_chunk.shape) != expected:
            raise RuntimeError(
                f"_wm_final_base_chunk must have shape {expected}, "
                f"got {tuple(final_base_chunk.shape)}")

        final_base_flat = self.action_scaler.scale(final_base_chunk).reshape(
            self.num_envs, self.flat_dim)
        info["final_observation"] = self._augment(
            raw_final_obs, final_base_flat)
        return info

    @staticmethod
    def _extract_rel(info):
        """从(批后)info 取本 env(env 0)的 raw rel_piece;无则 None。

        dexmg 在 eef_piece 模式经 info["rel_piece"] 透出特权 rel_piece(像 stage_id),
        AsyncVectorEnv 批成 (num_envs,12);训练 num_envs==1 取 [0]。eef 模式无此键 → None。
        """
        if not info:
            return None
        rel = info.get("rel_piece")
        if rel is None:
            return None
        rel = np.asarray(rel)
        return rel[0] if rel.ndim == 2 else rel

    def _phi_input(self, raw_obs, *, aug_obs=None, info=None):
        if self.potential is None:
            return None, None
        if getattr(self.potential, "state_mode", "eef") == "act_feat":
            return self.potential_feature_encoder.encode(raw_obs), None
        state_std = aug_obs["observation.state"] if aug_obs is not None \
            else self.state_standardizer.standardize(raw_obs["observation.state"])
        return state_std, self._extract_rel(info)

    def reset(self, **kwargs):
        raw_obs, info = self.vec_env.reset(**kwargs)
        self.base_policy.reset()
        self._stage = 0
        self._stage_now = 0
        base_flat = self._base_chunk_flat(raw_obs)
        self._last_base_flat = base_flat
        aug = self._augment(raw_obs, base_flat)
        self._start_state_std = aug["observation.state"]
        self._start_phi_state, self._start_rel_piece = self._phi_input(
            raw_obs, aug_obs=aug, info=info)
        return aug, info

    def step(self, residual_flat: torch.Tensor):
        """执行一段 chunk(开环逐步),返回 chunk 级 transition。

        普通 done 仍返回 autoreset 后的 augmented observation。带
        bootstrap_on_truncation 标记的 imagination 边界还会在 info 中保留独立的、
        完整 augmented final_observation，供 replay 从生成终点 bootstrap。
        """
        if self._last_base_flat is None:
            raise RuntimeError("Call reset() before step()")
        if self.base_action_mode == "queue":
            combined_flat = self._last_base_flat + residual_flat                       # 等同原版,不 clamp
        else:
            combined_flat = torch.clamp(self._last_base_flat + residual_flat, -1.0, 1.0)  # [B,L*D]
        b = combined_flat.shape[0]
        combined_chunk = combined_flat.reshape(b, self.chunk_length, self.action_dim)
        env_chunk = self.action_scaler.unscale(combined_chunk)        # [B,L,D] 原始尺度

        total_reward = torch.zeros(b)
        terminated = torch.zeros(b, dtype=torch.bool)
        truncated = torch.zeros(b, dtype=torch.bool)
        last_info = {}
        raw_obs = None
        start_stage = self._stage                        # 进入本 chunk 的起点阶段(= max_in_chunk 初值;用于 staged_bonus)
        max_in_chunk = self._stage                       # 本 chunk 内最高阶段(循环中更新)
        for t in range(self.chunk_length):
            raw_obs, reward, term, trunc, info = self.vec_env.step(env_chunk[:, t])
            total_reward = total_reward + reward.float().cpu()
            terminated = terminated | term.bool().cpu()
            truncated = truncated | trunc.bool().cpu()
            last_info = info
            if "stage_id" in info:                       # worker 经 info 透出的特权阶段
                s = int(info["stage_id"][0])             # env 0(训练 num_envs==1)
                self._stage = max(self._stage, s)        # episode 内单调闩锁
                self._stage_now = s                      # 瞬时(可回退),供 stage_id
                max_in_chunk = max(max_in_chunk, s)
                L = self._stage                          # 该步样本将被标的闩锁阶段
                self._stage_total_steps += 1
                self._steps_by_latch[L] = self._steps_by_latch.get(L, 0) + 1
                if s < L:                                # 瞬时退回 → 掉件/回退,桶变脏
                    self._stage_regress_steps += 1
                    self._regress_by_latch[L] = self._regress_by_latch.get(L, 0) + 1
            if bool((term | trunc).any()):
                break

        chunk_done = bool((terminated | truncated).any())
        if self.potential is None:
            total_reward = total_reward + shaping_reward(
                start_stage, max_in_chunk, mode=self.reward_shaping_mode,
                bonus=self.stage_reward_bonus, gamma=self.gamma, done=chunk_done)
        else:
            # object-aware(eef_piece):从 info 取 raw rel_piece 一并喂 Φ;eef 模式 rel=None、phi 忽略。
            end_phi_state, rel_next = self._phi_input(raw_obs, info=last_info)
            phi_start = self.potential.phi(self._start_phi_state, self._start_rel_piece)
            phi_next = self.potential.phi(end_phi_state, rel_next)
            total_reward = total_reward + potential_shaping(
                phi_start, phi_next, bonus=self.stage_reward_bonus,
                gamma=self.gamma, done=chunk_done)

        info = self._augment_bootstrap_final_observation(
            last_info, terminated, truncated)

        if bool((terminated | truncated).any()):
            self.base_policy.reset()
            self._stage = 0                              # autoreset 后新 episode 归零
            self._stage_now = 0
        base_flat = self._base_chunk_flat(raw_obs)
        self._last_base_flat = base_flat

        aug_obs = self._augment(raw_obs, base_flat)      # stage_id = chunk 结束时 max-so-far
        self._start_state_std = aug_obs["observation.state"]   # 本 chunk 末 = 下个 chunk 起点
        # rel_piece 同步携带:done 时 SAME_STEP autoreset 后 info 顶层是 reset_info,
        # rel_piece 与返回的(reset)obs 同步 → 下个 chunk 起点 Φ 一致。
        self._start_phi_state, self._start_rel_piece = self._phi_input(
            raw_obs, aug_obs=aug_obs, info=last_info)
        info["scaled_action"] = combined_flat
        info["max_stage_in_chunk"] = max_in_chunk        # 给 stage-balanced replay
        return aug_obs, total_reward, terminated, truncated, info

    def stage_purity_summary(self) -> str:
        """闩锁桶纯度的可读汇总:总回退率 + 各 stage 桶内退化样本占比。"""
        tot = self._stage_total_steps
        if tot == 0:
            return "no stage steps"
        parts = [f"regress {self._stage_regress_steps}/{tot}={self._stage_regress_steps / tot:.1%}"]
        for L in sorted(self._steps_by_latch):
            n = self._steps_by_latch[L]
            r = self._regress_by_latch.get(L, 0)
            parts.append(f"stage{L}:{r}/{n}={r / n:.0%}")
        return "  ".join(parts)

    def render(self):
        return self.vec_env.render()

    def close(self):
        return self.vec_env.close()

    def __getattr__(self, name):
        return getattr(self.vec_env, name)
