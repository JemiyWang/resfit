"""HIQL 分层路运行时 helper(Phase 3):载冻结 gc_value+high_actor,在线/离线算潜子目标 z。

约束:给 actor/critic 的 observation.state 恒 18 维;30 维 eef_piece(18+12 标准化 rel)只用于
算 z;z(10 维)是唯一进策略的 object-aware 信号。online/offline 用同一套 rel mean/std 标准化。
设计见 docs/superpowers/specs/2026-06-08-hiql-hierarchy-residual-design.md。
"""
import numpy as np
import torch

from resfit.rl_finetuning.chunk_residual.hiql_gc_value import load_gc_value
from resfit.rl_finetuning.chunk_residual.hiql_high_actor import load_high_actor


def representative_goal30(seqs):
    """从各 demo 末态取 medoid(离它们均值最近的那条**真实**末态)当固定任务目标 goal30。

    比算术均值更对:均值是离任何真实末态约 1 个单位的"虚构质心"(且把散得最厉害的
    rel_piece/物体部分糊掉),而 high_actor 训练时只见过真实态当 goal,喂均值是训练/推理
    错配。HIQL 的 goal 永远是真实可达态、从不平均(2026-06-09 讨论)。返回 [D] np.float32。
    seqs:list[(T,D)],每条 demo 末帧 s[-1] 为其末态。
    """
    finals = np.stack([np.asarray(s)[-1] for s in seqs], axis=0).astype(np.float64)
    mean = finals.mean(axis=0)
    medoid = finals[int(np.argmin(np.linalg.norm(finals - mean, axis=1)))]
    return medoid.astype(np.float32)


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
        # 注:不用 np.asarray 包装 —— 这些量可能是 ckpt 里 map_location=cuda 加载出的 cuda 张量,
        # np.asarray(cuda_tensor) 会崩;torch.as_tensor 本就吃 numpy/cpu/cuda 张量并搬到 device。
        self.goal = torch.as_tensor(goal, dtype=torch.float32, device=device).reshape(-1)
        if state_mode == "eef_piece":
            assert rel_stats is not None, "eef_piece 须给 rel_stats"
            self.rel_mean = torch.as_tensor(rel_stats[0], dtype=torch.float32, device=device)
            self.rel_std = torch.as_tensor(rel_stats[1], dtype=torch.float32, device=device)
        elif state_mode == "act_feat":
            assert extractor is not None and feat_stats is not None, "act_feat 须给 extractor + feat_stats"
            self.extractor = extractor
            self.feat_mean = torch.as_tensor(feat_stats[0], dtype=torch.float32, device=device)
            self.feat_std = torch.as_tensor(feat_stats[1], dtype=torch.float32, device=device)
        else:
            raise ValueError(f"unknown state_mode: {state_mode}")

    @classmethod
    def from_ckpts(cls, gc_value_ckpt, high_actor_ckpt, *, goal, device="cpu",
                   renorm_subgoal=False, base_policy=None):
        gc, info = load_gc_value(gc_value_ckpt, map_location=device)
        ha, _ = load_high_actor(high_actor_ckpt, map_location=device)
        sm = info["state_mode"]
        assert sm in ("eef_piece", "act_feat"), f"分层路 gc_value state_mode 须 eef_piece/act_feat,got {sm}"
        assert ha.rep_dim == gc.rep_dim, f"rep_dim 不一致: high_actor={ha.rep_dim} gc_value={gc.rep_dim}"
        assert ha.state_dim == gc.state_dim, f"state_dim 不一致: high_actor={ha.state_dim} gc_value={gc.state_dim}"
        if sm == "eef_piece":
            return cls(gc, ha, goal, device=device, renorm_subgoal=renorm_subgoal,
                       state_mode="eef_piece", rel_stats=(info["rel_piece_mean"], info["rel_piece_std"]))
        assert base_policy is not None, "act_feat 在线子目标须传 base_policy 建特征器"
        from resfit.rl_finetuning.chunk_residual.act_feature import ActFeatureExtractor
        sig = info["act_feat_signature"] or {}
        ext = ActFeatureExtractor(base_policy, image_keys=sig["image_keys"],
                                  proprio_key=sig.get("proprio_key", "observation.state"),
                                  pooling=sig.get("pooling", "mean"))
        return cls(gc, ha, goal, device=device, renorm_subgoal=renorm_subgoal,
                   state_mode="act_feat", extractor=ext, feat_stats=(info["mean"], info["std"]))

    def build_state30(self, state_std, rel_raw):
        """18 维已标准化 state(tensor [B,18]) + raw rel_piece([B,12] np/tensor) -> [B,30] tensor。"""
        x = torch.as_tensor(state_std, dtype=torch.float32, device=self.device)
        if x.ndim == 1:
            x = x.unsqueeze(0)
        rel = torch.as_tensor(np.asarray(rel_raw), dtype=torch.float32, device=self.device)
        if rel.ndim == 1:
            rel = rel.unsqueeze(0)
        rel_n = (rel - self.rel_mean) / self.rel_std
        return torch.cat([x, rel_n], dim=-1)

    @torch.no_grad()
    def subgoal_online(self, obs, rel_raw=None):
        """eef_piece: obs 传已 std 的 state([B,18]) 或含 observation.state 的 dict(+rel_raw);
        act_feat: obs 传含 images+observation.state 的 dict。"""
        if self.state_mode == "eef_piece":
            state_std = obs["observation.state"] if isinstance(obs, dict) else obs
            s = self.build_state30(state_std, rel_raw)
        else:  # act_feat
            feat = self.extractor.embed_batch(obs)
            s = (feat.to(self.device) - self.feat_mean) / self.feat_std
        g = self.goal.unsqueeze(0).expand(s.shape[0], -1)
        z = self.ha(s, g).mean
        if self.renorm_subgoal:
            z = z / (z.norm(dim=-1, keepdim=True) + 1e-8) * (self.rep_dim ** 0.5)
        return z

    @torch.no_grad()
    def subgoal_waypoint(self, s30_base, s30_target):
        """离线:z = φ(base=s_t, target=s_{t+k})(真航点)。输入须是已拼好的 30 维 state
        (来自 state30 缓存,18 std + 12 标准化 rel),不是 18 维原始 state。返回 [B, rep_dim]。"""
        b = torch.as_tensor(np.asarray(s30_base), dtype=torch.float32, device=self.device)
        t = torch.as_tensor(np.asarray(s30_target), dtype=torch.float32, device=self.device)
        if b.ndim == 1:
            b = b.unsqueeze(0)
        if t.ndim == 1:
            t = t.unsqueeze(0)
        assert b.shape[-1] == self.vf.state_dim and t.shape[-1] == self.vf.state_dim, \
            f"subgoal_waypoint 须传 {self.vf.state_dim} 维 state(已拼 rel),got {b.shape[-1]}/{t.shape[-1]}"
        return self.vf.phi(b, t)


# dim-agnostic alias
representative_goal = representative_goal30
