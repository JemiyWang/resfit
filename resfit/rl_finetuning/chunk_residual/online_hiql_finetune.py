"""HIQL 在线联合微调器(Phase 3):冻 φ,在线更新 V 头 + high_actor,online+offline 混采。

设计见 docs/superpowers/specs/2026-06-24-hiql-online-joint-finetune-design.md。
"""
import copy

import numpy as np
import torch

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import build_gc_data, value_update_step
from resfit.rl_finetuning.chunk_residual.hiql_high_actor import high_actor_update_step
from resfit.rl_finetuning.chunk_residual.online_hiql_store import (
    OnlineHiqlStore, sample_value_batch, sample_high_actor_batch)


class OnlineHiqlFinetuner:
    def __init__(self, subgoal, *, offline_seqs, device="cpu",
                 value_lr=1e-5, high_actor_lr=1e-5, way_steps=25,
                 offline_fraction=0.5, batch_size=256, every=1, utd=1,
                 gamma=0.99, expectile=0.7, ema=0.005, future_mode="geometric",
                 value_loss_mode="shared_min", value_mask_mode="done_aware",
                 adv_agg="min", beta=1.0, finetune_value=True, finetune_high_actor=True,
                 max_online_transitions=50_000, min_online_transitions=2_000, seed=0):
        self.subgoal = subgoal
        self.vf = subgoal.vf
        self.ha = subgoal.ha
        self.device = device
        self.way_steps = int(way_steps)
        self.batch_size = int(batch_size)
        self.every = max(int(every), 1)
        self.utd = max(int(utd), 1)
        self.gamma = gamma
        self.expectile = expectile
        self.ema = ema
        self.future_mode = future_mode
        self.value_loss_mode = value_loss_mode
        self.value_mask_mode = value_mask_mode
        self.adv_agg = adv_agg
        self.beta = beta
        self.finetune_value = bool(finetune_value)
        self.finetune_high_actor = bool(finetune_high_actor)
        self.min_online = int(min_online_transitions)
        self.rng = np.random.default_rng(seed)
        self._tick = 0

        # 混采配比:offline_fraction 决定 V/ha 更新 batch 里 offline 占比
        self.offline_bs = int(round(self.batch_size * offline_fraction))
        self.online_bs = self.batch_size - self.offline_bs
        if self.online_bs <= 0:
            raise ValueError("offline_fraction 过大,online_bs<=0(在线微调需要 online 样本)")

        # 冻 φ,解冻 V 头 + high_actor
        for p in self.vf.parameters():
            p.requires_grad_(False)
        if self.finetune_value:
            for p in list(self.vf.v1.parameters()) + list(self.vf.v2.parameters()):
                p.requires_grad_(True)
            self.target = copy.deepcopy(self.vf).to(device).eval()
            for p in self.target.parameters():
                p.requires_grad_(False)
            self.value_opt = torch.optim.Adam(
                list(self.vf.v1.parameters()) + list(self.vf.v2.parameters()), lr=value_lr)
        if self.finetune_high_actor:
            for p in self.ha.parameters():
                p.requires_grad_(True)
            self.ha_opt = torch.optim.Adam(self.ha.parameters(), lr=high_actor_lr)

        # online / offline store
        self.online = OnlineHiqlStore(max_online_transitions)
        self._ep = []
        if self.offline_bs > 0:
            empty = [np.empty(0, dtype=np.int64) for _ in offline_seqs]
            self.offline_data = build_gc_data(list(offline_seqs), empty)
        else:
            self.offline_data = None

    # ---- 采集钩子 ----
    def on_step(self, s):
        self._ep.append(np.asarray(s.detach().to("cpu"), dtype=np.float32).reshape(-1))

    def on_episode_end(self):
        if len(self._ep) >= 2:
            self.online.add_episode(np.stack(self._ep, axis=0))
        self._ep = []

    # ---- 混采:从 online(+offline)各取一份,拼成大 batch ----
    def _mixed_value_batch(self):
        on = sample_value_batch(self.online.data(), self.online_bs, self.rng,
                                future_mode=self.future_mode, gamma=self.gamma)
        if self.offline_data is not None and self.offline_bs > 0:
            off = sample_value_batch(self.offline_data, self.offline_bs, self.rng,
                                     future_mode=self.future_mode, gamma=self.gamma)
            parts = [torch.cat([a, b], dim=0) for a, b in zip(on, off)]
        else:
            parts = list(on)
        return [p.to(self.device) for p in parts]

    def _mixed_high_actor_batch(self):
        on = sample_high_actor_batch(self.online.data(), self.online_bs, self.rng,
                                     way_steps=self.way_steps, future_mode=self.future_mode,
                                     gamma=self.gamma)
        if self.offline_data is not None and self.offline_bs > 0:
            off = sample_high_actor_batch(self.offline_data, self.offline_bs, self.rng,
                                          way_steps=self.way_steps, future_mode=self.future_mode,
                                          gamma=self.gamma)
            parts = [torch.cat([a, b], dim=0) for a, b in zip(on, off)]
        else:
            parts = list(on)
        return [p.to(self.device) for p in parts]

    # ---- 主更新 ----
    def maybe_update(self):
        """每调用一次算一次更新机会;每 self.every 次机会执行一轮(utd 次)value/high_actor 更新。env_steps 无关,避免 chunk_length 缩放导致 every 门控失效。"""
        if not self.online.ready(self.min_online):
            return None
        self._tick += 1
        if self._tick % self.every != 0:
            return None
        v_loss = h_loss = None
        for _ in range(self.utd):
            if self.finetune_value:
                s, s_next, g, success, done = self._mixed_value_batch()
                v_loss = value_update_step(
                    self.vf, self.target, self.value_opt, s, s_next, g, success, done,
                    gamma=self.gamma, expectile=self.expectile, ema=self.ema,
                    value_loss_mode=self.value_loss_mode, value_mask_mode=self.value_mask_mode)
            if self.finetune_high_actor:
                s, sw, g = self._mixed_high_actor_batch()
                h_loss = high_actor_update_step(
                    self.ha, self.vf, self.ha_opt, s, sw, g, beta=self.beta, adv_agg=self.adv_agg)
        metrics = {"finetune/online_trans": float(len(self.online))}
        if v_loss is not None:
            metrics["finetune/value_loss"] = v_loss
        if h_loss is not None:
            metrics["finetune/high_actor_loss"] = h_loss
        return metrics
