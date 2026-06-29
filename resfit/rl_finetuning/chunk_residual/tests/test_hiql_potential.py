import numpy as np
import pytest
import torch

from resfit.rl_finetuning.chunk_residual.hiql_potential import potential_shaping
from resfit.rl_finetuning.chunk_residual.hiql_potential import HiqlPotential
from resfit.rl_finetuning.chunk_residual.hiql_value import ValueMLP, save_value


def test_potential_shaping_not_done():
    # F = bonus*(gamma*phi_next - phi_start)
    f = potential_shaping(1.0, 2.0, bonus=1.0, gamma=0.99, done=False)
    assert abs(f - (0.99 * 2.0 - 1.0)) < 1e-6


def test_potential_shaping_done_zeroes_next():
    # done -> phi_next=0 -> F = -bonus*phi_start
    f = potential_shaping(3.0, 9.9, bonus=2.0, gamma=0.99, done=True)
    assert abs(f - (2.0 * (0.0 - 3.0))) < 1e-6


def test_potential_shaping_accepts_tensor_scalar():
    # online 传 [1] tensor 标量,应与 float 同结果
    f = potential_shaping(torch.tensor([1.0]), torch.tensor([2.0]),
                          bonus=1.0, gamma=0.99, done=False)
    assert abs(float(f) - (0.99 * 2.0 - 1.0)) < 1e-6


def _make_fake_ckpt(tmp_path, vmin, vmax):
    m = ValueMLP(state_dim=3, hidden=8)
    p = str(tmp_path / "value.pt")
    save_value(p, m, v_stats={"min": vmin, "max": vmax, "mean": 0.5 * (vmin + vmax)},
               mean=torch.zeros(3), std=torch.ones(3), dataset_id="dummy")
    return p, m


def test_hiqlpotential_auto_scale(tmp_path):
    p, _ = _make_fake_ckpt(tmp_path, vmin=0.0, vmax=2.0)
    # num_stages=5 -> 动态范围目标 [0,4];auto_scale=(5-1)/(2-0)=2.0;phi_scale=1
    pot = HiqlPotential.from_ckpt(p, num_stages=5, phi_scale=1.0, device="cpu")
    assert abs(pot.scale - 2.0) < 1e-6


def test_hiqlpotential_non_stage_task_scale_is_nonzero(tmp_path):
    p, _ = _make_fake_ckpt(tmp_path, vmin=0.0, vmax=2.0)
    # Tasks without stage detectors pass num_stages=1. HIQL potential should still
    # produce a usable Phi scale instead of silently zeroing all shaping rewards.
    pot = HiqlPotential.from_ckpt(p, num_stages=1, phi_scale=1.0, device="cpu")
    assert abs(pot.scale - 0.5) < 1e-6


def test_hiqlpotential_phi_equals_v_times_scale(tmp_path):
    p, model = _make_fake_ckpt(tmp_path, vmin=0.0, vmax=2.0)
    pot = HiqlPotential.from_ckpt(p, num_stages=5, phi_scale=1.0, device="cpu")
    x = torch.randn(4, 3)
    with torch.no_grad():
        expected = model(x).squeeze(-1) * 2.0
    out = pot.phi(x)
    assert out.shape == (4,)
    assert torch.allclose(out, expected, atol=1e-5)


def test_hiqlpotential_phi_scale_multiplies(tmp_path):
    p, _ = _make_fake_ckpt(tmp_path, vmin=0.0, vmax=2.0)
    pot = HiqlPotential.from_ckpt(p, num_stages=5, phi_scale=0.5, device="cpu")
    assert abs(pot.scale - 1.0) < 1e-6   # 2.0 * 0.5


# --- ③a' object-aware:eef_piece 模式 phi 拼 30 维(state18_std | (rel-mean)/std)---

def _make_objaware_ckpt(tmp_path, vmin=0.0, vmax=2.0):
    """30 维 value ckpt(state_mode=eef_piece)+ rel_piece mean/std,模拟 ③a' 产物。"""
    m = ValueMLP(state_dim=30, hidden=8)
    rel_mean = np.arange(12, dtype=np.float32)          # 非零非一,验证确实用上
    rel_std = np.linspace(0.5, 3.0, 12).astype(np.float32)
    p = str(tmp_path / "value_obj.pt")
    save_value(p, m, v_stats={"min": vmin, "max": vmax, "mean": 0.5 * (vmin + vmax)},
               mean=torch.zeros(18), std=torch.ones(18), dataset_id="dummy",
               state_mode="eef_piece", rel_piece_stats=(rel_mean, rel_std))
    return p, m, rel_mean, rel_std


def test_hiqlpotential_reads_state_mode_and_rel_stats(tmp_path):
    p, _, rel_mean, rel_std = _make_objaware_ckpt(tmp_path)
    pot = HiqlPotential.from_ckpt(p, num_stages=5, phi_scale=1.0, device="cpu")
    assert pot.state_mode == "eef_piece"
    assert pot.model.state_dim == 30
    assert np.allclose(pot.rel_mean.cpu().numpy(), rel_mean)
    assert np.allclose(pot.rel_std.cpu().numpy(), rel_std)


def test_hiqlpotential_phi_eef_piece_builds_30dim_batched(tmp_path):
    """offline 口径:state[T,18] + rel[T,12] -> V(concat[state18, (rel-mean)/std])*scale。"""
    p, model, rel_mean, rel_std = _make_objaware_ckpt(tmp_path)
    pot = HiqlPotential.from_ckpt(p, num_stages=5, phi_scale=1.0, device="cpu")
    state18 = torch.randn(4, 18)
    rel_raw = torch.randn(4, 12)
    rel_n = (rel_raw - torch.as_tensor(rel_mean)) / torch.as_tensor(rel_std)
    x30 = torch.cat([state18, rel_n], dim=-1)
    with torch.no_grad():
        expected = model(x30).squeeze(-1) * pot.scale
    out = pot.phi(state18, rel_raw)
    assert out.shape == (4,)
    assert torch.allclose(out, expected, atol=1e-5)


def test_hiqlpotential_phi_eef_piece_online_single(tmp_path):
    """online 口径:state[1,18] + rel (12,) numpy -> [1] 输出,与 batched 同公式。"""
    p, model, rel_mean, rel_std = _make_objaware_ckpt(tmp_path)
    pot = HiqlPotential.from_ckpt(p, num_stages=5, device="cpu")
    state18 = torch.randn(1, 18)
    rel_raw = np.random.RandomState(0).randn(12).astype(np.float32)
    rel_n = (torch.as_tensor(rel_raw) - torch.as_tensor(rel_mean)) / torch.as_tensor(rel_std)
    x30 = torch.cat([state18, rel_n.unsqueeze(0)], dim=-1)
    with torch.no_grad():
        expected = model(x30).squeeze(-1) * pot.scale
    out = pot.phi(state18, rel_raw)
    assert out.shape == (1,)
    assert torch.allclose(out, expected, atol=1e-5)


def test_hiqlpotential_phi_eef_piece_requires_rel(tmp_path):
    p, _, _, _ = _make_objaware_ckpt(tmp_path)
    pot = HiqlPotential.from_ckpt(p, num_stages=5, device="cpu")
    with pytest.raises((ValueError, TypeError)):
        pot.phi(torch.randn(1, 18))      # eef_piece 模式缺 rel_piece 必须报错


def test_hiqlpotential_phi_eef_mode_ignores_rel(tmp_path):
    """eef 模式(旧默认):phi 无视 rel,与单参旧行为逐位一致。"""
    p, _ = _make_fake_ckpt(tmp_path, vmin=0.0, vmax=2.0)   # state_dim=3, state_mode 默认 eef
    pot = HiqlPotential.from_ckpt(p, num_stages=5, device="cpu")
    assert pot.state_mode == "eef"
    x = torch.randn(4, 3)
    assert torch.allclose(pot.phi(x), pot.phi(x, rel_piece_raw=np.zeros(12)), atol=1e-6)


def _make_actfeat_ckpt(tmp_path, vmin=0.0, vmax=4.0):
    m = ValueMLP(state_dim=4, hidden=8)
    p = str(tmp_path / "value_actfeat.pt")
    mean = torch.tensor([1.0, 2.0, 3.0, 4.0])
    std = torch.tensor([2.0, 2.0, 4.0, 4.0])
    sig = {
        "act_ckpt_id": "base-act",
        "image_keys": ["observation.images.cam"],
        "proprio_key": "observation.state",
        "pooling": "mean",
    }
    save_value(
        p,
        m,
        v_stats={"min": vmin, "max": vmax, "mean": 0.5 * (vmin + vmax)},
        mean=mean,
        std=std,
        dataset_id="dummy",
        state_mode="act_feat",
        act_feat_signature=sig,
        act_weight_sha="sha-act",
    )
    return p, m, mean, std, sig


def test_hiqlpotential_loads_actfeat_metadata_and_standardizes(tmp_path):
    p, model, mean, std, sig = _make_actfeat_ckpt(tmp_path)
    pot = HiqlPotential.from_ckpt(p, num_stages=5, phi_scale=1.0, device="cpu")

    assert pot.state_mode == "act_feat"
    assert pot.model.state_dim == 4
    assert pot.act_feat_signature == sig
    assert pot.act_weight_sha == "sha-act"
    raw = torch.tensor([[3.0, 6.0, 11.0, 20.0]])
    expected_std = (raw - mean) / std
    assert torch.allclose(pot.standardize_features(raw), expected_std)
    assert torch.allclose(
        pot.phi(expected_std),
        model(expected_std).squeeze(-1) * pot.scale,
        atol=1e-5,
    )


class _FakeStateStandardizer:
    def standardize(self, state):
        return torch.as_tensor(state, dtype=torch.float32) + 10.0


class _FakeExtractor:
    def __init__(self):
        self.seen_state = None

    def embed_batch(self, raw_obs):
        self.seen_state = raw_obs["observation.state"].clone()
        image_scalar = raw_obs["observation.images.cam"].float().mean(dim=(1, 2, 3), keepdim=False).unsqueeze(-1)
        return torch.cat([image_scalar, raw_obs["observation.state"].float()], dim=-1)


def test_potential_act_feature_encoder_standardizes_proprio_then_feature(tmp_path):
    from resfit.rl_finetuning.chunk_residual.act_feature import PotentialActFeatureEncoder

    p, _, mean, std, _ = _make_actfeat_ckpt(tmp_path)
    pot = HiqlPotential.from_ckpt(p, num_stages=5, device="cpu")
    extractor = _FakeExtractor()
    encoder = PotentialActFeatureEncoder(
        extractor,
        _FakeStateStandardizer(),
        pot,
        proprio_key="observation.state",
    )
    raw_obs = {
        "observation.state": torch.tensor([[1.0, 2.0, 3.0]]),
        "observation.images.cam": torch.ones(1, 3, 2, 2),
    }

    out = encoder.encode(raw_obs)
    raw_feat = torch.tensor([[1.0, 11.0, 12.0, 13.0]])
    assert torch.allclose(extractor.seen_state, torch.tensor([[11.0, 12.0, 13.0]]))
    assert torch.allclose(out, (raw_feat - mean) / std)


from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import transition_rewards


class _StubPot:
    """phi(state_seq[T,D]) = 每行第一元素,模拟 V(state)。"""
    def phi(self, state_seq, rel_piece_seq=None):
        return state_seq[:, 0]


def test_transition_rewards_hiql_uses_v():
    instant = np.array([0, 1, 2])                 # 3 帧 -> 2 transition,T=3
    state_seq = torch.tensor([[10.], [20.], [30.]])  # phi = [10,20,30]
    r = transition_rewards(instant, bonus=1.0, mode="potential", gamma=0.99,
                           success=True, potential=_StubPot(), state_seq=state_seq)
    exp0 = potential_shaping(10.0, 20.0, bonus=1.0, gamma=0.99, done=False)        # t=0 not done
    exp1 = 1.0 + potential_shaping(20.0, 30.0, bonus=1.0, gamma=0.99, done=True)   # t=1 done(base+1)
    assert abs(r[0] - exp0) < 1e-5
    assert abs(r[1] - exp1) < 1e-5


def test_transition_rewards_none_matches_stage_baseline():
    # potential=None(默认)走现状 latch 路径,逐位等价
    instant = np.array([0, 1, 2])
    from resfit.rl_finetuning.chunk_residual.chunk_env_wrapper import shaping_reward
    r = transition_rewards(instant, bonus=1.0, mode="potential", gamma=0.99, success=True)
    exp0 = shaping_reward(0, 1, mode="potential", bonus=1.0, gamma=0.99, done=False)
    exp1 = 1.0 + shaping_reward(1, 2, mode="potential", bonus=1.0, gamma=0.99, done=True)
    assert abs(r[0] - exp0) < 1e-5 and abs(r[1] - exp1) < 1e-5


class _RelStubPot:
    """phi = state第一列 + rel第一列;rel 非空时必须被透传进来才对得上。"""
    def phi(self, state_seq, rel_piece_seq=None):
        base = np.asarray(state_seq)[:, 0]
        if rel_piece_seq is not None:
            base = base + np.asarray(rel_piece_seq)[:, 0]
        return base


def test_transition_rewards_passes_rel_piece_seq():
    # ③a' object-aware:offline 用 18 维 std state + raw rel_piece,phi 内部拼 30 维。
    instant = np.array([0, 1, 2])
    state_seq = torch.tensor([[10.], [20.], [30.]])
    rel_seq = np.array([[1., 9.], [2., 9.], [3., 9.]])   # 第一列加进 phi → phi=[11,22,33]
    r = transition_rewards(instant, bonus=1.0, mode="potential", gamma=0.99,
                           success=True, potential=_RelStubPot(),
                           state_seq=state_seq, rel_piece_seq=rel_seq)
    exp0 = potential_shaping(11.0, 22.0, bonus=1.0, gamma=0.99, done=False)
    exp1 = 1.0 + potential_shaping(22.0, 33.0, bonus=1.0, gamma=0.99, done=True)
    assert abs(r[0] - exp0) < 1e-5 and abs(r[1] - exp1) < 1e-5


def test_transition_fields_threads_rel_piece_seq():
    # transition_fields 是 build_offline_buffer 的入口,必须把 rel_piece_seq 透到 phi。
    from resfit.rl_finetuning.chunk_residual.offline_hdf5_buffer import transition_fields
    instant = np.array([0, 1, 2])
    state_seq = torch.tensor([[10.], [20.], [30.]])
    rel_seq = np.array([[1., 9.], [2., 9.], [3., 9.]])
    fld = transition_fields(instant, bonus=1.0, mode="potential", gamma=0.99,
                            success=True, potential=_RelStubPot(),
                            state_seq=state_seq, rel_piece_seq=rel_seq)
    exp0 = potential_shaping(11.0, 22.0, bonus=1.0, gamma=0.99, done=False)
    assert abs(fld["reward"][0] - exp0) < 1e-5


class _ActFeatStubPot:
    state_mode = "act_feat"

    def phi(self, state_seq, rel_piece_seq=None):
        assert rel_piece_seq is None
        return torch.as_tensor(state_seq, dtype=torch.float32)[:, 1]


def test_transition_rewards_actfeat_potential_uses_act_feat_seq():
    instant = np.array([0, 0, 0])
    low_state = torch.tensor([[100.0, 100.0], [100.0, 100.0], [100.0, 100.0]])
    act_feat = torch.tensor([[1.0, 10.0], [1.0, 20.0], [1.0, 30.0]])
    r = transition_rewards(
        instant,
        bonus=1.0,
        mode="potential",
        gamma=0.99,
        success=True,
        potential=_ActFeatStubPot(),
        state_seq=low_state,
        act_feat_seq=act_feat,
    )
    exp0 = potential_shaping(10.0, 20.0, bonus=1.0, gamma=0.99, done=False)
    exp1 = 1.0 + potential_shaping(20.0, 30.0, bonus=1.0, gamma=0.99, done=True)
    assert abs(r[0] - exp0) < 1e-5
    assert abs(r[1] - exp1) < 1e-5


def test_transition_rewards_actfeat_requires_act_feat_seq():
    instant = np.array([0, 0, 0])
    with pytest.raises(AssertionError, match="act_feat potential"):
        transition_rewards(
            instant,
            bonus=1.0,
            mode="potential",
            gamma=0.99,
            success=True,
            potential=_ActFeatStubPot(),
            state_seq=torch.zeros(3, 2),
        )


from types import SimpleNamespace

from resfit.rl_finetuning.chunk_residual.train_chunk_residual import (
    _offline_buffer_signature,
    _load_act_feat_cache_for_training,
    _needs_act_feat_cache,
    _prepare_libero_offline,
    _reject_libero_actfeat_potential_offline,
    _validate_actfeat_potential_cache,
    build_parser,
)


def test_parser_potential_source_defaults():
    args = build_parser().parse_args(["--task", "TwoArmThreePieceAssembly"])
    assert args.potential_source == "stage"
    assert args.hiql_value_ckpt is None
    assert args.phi_scale == 1.0


def test_parser_potential_source_hiql():
    args = build_parser().parse_args(
        ["--task", "TwoArmThreePieceAssembly", "--potential_source", "hiql",
         "--hiql_value_ckpt", "v.pt", "--phi_scale", "0.5"])
    assert args.potential_source == "hiql"
    assert args.hiql_value_ckpt == "v.pt"
    assert args.phi_scale == 0.5


def test_needs_act_feat_cache_for_potential_only():
    pot = SimpleNamespace(state_mode="act_feat")
    assert _needs_act_feat_cache(pot, None) is True
    assert _needs_act_feat_cache(None, "act_feat") is True
    assert _needs_act_feat_cache(SimpleNamespace(state_mode="eef"), None) is False


def test_env_state_mode_actfeat_potential_keeps_env_eef():
    from resfit.rl_finetuning.chunk_residual.train_chunk_residual import _env_state_mode_for_training

    assert _env_state_mode_for_training("act_feat", None) == "eef"
    assert _env_state_mode_for_training("act_feat", "eef_piece") == "eef_piece"
    assert _env_state_mode_for_training("eef_piece", None) == "eef_piece"


def test_prepare_libero_offline_rejects_actfeat_potential_before_build():
    args = SimpleNamespace(
        libero_stats_json="/tmp/libero/meta/stats.json",
        libero_suite="libero_spatial",
        libero_task_id=0,
        offline_num_demos=1,
    )
    _reject_libero_actfeat_potential_offline(None)
    _reject_libero_actfeat_potential_offline(SimpleNamespace(state_mode="eef"))
    with pytest.raises(NotImplementedError, match="LIBERO offline buffer.*act_feat HIQL potential"):
        _prepare_libero_offline(
            args, ["observation.images.agentview"], 84, SimpleNamespace(state_mode="act_feat"))


def test_load_act_feat_cache_for_training_loads_potential_only():
    pot = SimpleNamespace(
        state_mode="act_feat",
        model=SimpleNamespace(state_dim=4),
        act_feat_signature={
            "act_ckpt_id": "base-act",
            "image_keys": ["observation.images.cam"],
            "proprio_key": "observation.state",
            "pooling": "mean",
        },
        act_weight_sha="sha-act",
        feature_mean=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        feature_std=torch.tensor([2.0, 2.0, 4.0, 4.0]),
    )
    seqs = [np.ones((3, 4), np.float32)]
    stats = (np.array([1.0, 2.0, 3.0, 4.0], np.float32),
             np.array([2.0, 2.0, 4.0, 4.0], np.float32))
    args = SimpleNamespace(act_feat_cache="cache.npz")

    got = _load_act_feat_cache_for_training(
        args,
        pot,
        None,
        load_fn=lambda path: (seqs, stats, dict(pot.act_feat_signature), "sha-act"),
    )

    assert got == (seqs, stats, pot.act_feat_signature, "sha-act")


def test_load_act_feat_cache_for_training_requires_cache_path_for_actfeat_potential():
    pot = SimpleNamespace(
        state_mode="act_feat",
        model=SimpleNamespace(state_dim=4),
        act_feat_signature=None,
        act_weight_sha=None,
        feature_mean=torch.zeros(4),
        feature_std=torch.ones(4),
    )
    args = SimpleNamespace(act_feat_cache=None)

    with pytest.raises(AssertionError, match="act_feat potential"):
        _load_act_feat_cache_for_training(args, pot, None, load_fn=lambda path: None)


def test_validate_actfeat_potential_cache_rejects_signature_mismatch():
    pot = SimpleNamespace(
        model=SimpleNamespace(state_dim=4),
        act_feat_signature={
            "act_ckpt_id": "base-act",
            "image_keys": ["observation.images.cam"],
            "proprio_key": "observation.state",
            "pooling": "mean",
        },
        act_weight_sha="sha-act",
        feature_mean=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        feature_std=torch.tensor([2.0, 2.0, 4.0, 4.0]),
    )
    args = SimpleNamespace(act_feat_cache="cache.npz")

    with pytest.raises(AssertionError, match="签名不符"):
        _validate_actfeat_potential_cache(
            args,
            pot,
            {
                "act_ckpt_id": "wrong-act",
                "image_keys": ["observation.images.cam"],
                "proprio_key": "observation.state",
                "pooling": "mean",
            },
            "sha-act",
            [np.ones((3, 4), np.float32)],
            cache_stats=(np.array([1.0, 2.0, 3.0, 4.0], np.float32),
                         np.array([2.0, 2.0, 4.0, 4.0], np.float32)),
        )


def test_validate_actfeat_potential_cache_rejects_stats_mismatch():
    pot = SimpleNamespace(
        model=SimpleNamespace(state_dim=4),
        act_feat_signature={
            "act_ckpt_id": "base-act",
            "image_keys": ["observation.images.cam"],
            "proprio_key": "observation.state",
            "pooling": "mean",
        },
        act_weight_sha="sha-act",
        feature_mean=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        feature_std=torch.tensor([2.0, 2.0, 4.0, 4.0]),
    )
    args = SimpleNamespace(act_feat_cache="cache.npz")

    with pytest.raises(AssertionError, match="mean"):
        _validate_actfeat_potential_cache(
            args,
            pot,
            dict(pot.act_feat_signature),
            "sha-act",
            [np.ones((3, 4), np.float32)],
            cache_stats=(np.array([0.0, 2.0, 3.0, 4.0], np.float32),
                         np.array([2.0, 2.0, 4.0, 4.0], np.float32)),
        )


class _CacheArgs:
    dataset = "ankile/dummy"
    offline_dataset_path = None
    offline_num_demos = 2
    action_scale = 0.2
    min_range_per_dim = 0.1
    stage_reward_bonus = 1.0
    gamma = 0.99
    n_step = 3
    task = "TwoArmBoxCleanup"
    offline_stage_cache = None
    potential_source = "hiql"
    hiql_value_ckpt = "/tmp/value.pt"
    phi_scale = 1.0
    subgoal_conditioned = False
    offline_base_mode = "gt"
    data_source = "hdf5"
    lerobot_root = None
    act_feat_cache = "/tmp/act_feat_cache.npz"


def test_offline_buffer_signature_actfeat_potential_includes_cache_and_stats_hash():
    potential = SimpleNamespace(
        model=SimpleNamespace(state_dim=4),
        scale=1.25,
        state_mode="act_feat",
        feature_mean=torch.tensor([1.0, 2.0, 3.0, 4.0]),
        feature_std=torch.tensor([2.0, 2.0, 4.0, 4.0]),
        act_feat_signature={
            "act_ckpt_id": "base-act",
            "image_keys": ["observation.images.cam"],
            "proprio_key": "observation.state",
            "pooling": "mean",
        },
        act_weight_sha="sha-act",
    )

    out = _offline_buffer_signature(
        _CacheArgs(),
        ["observation.images.cam"],
        12,
        "potential",
        potential=potential,
        act_feat_cache_sig={"cache_id": "cache-sig"},
        act_feat_cache_sha="cache-sha",
    )

    assert out["value_state_mode"] == "act_feat"
    assert out["value_state_dim"] == 4
    assert out["value_state_stats_sha"]
    assert out["act_feat_cache"] == "/tmp/act_feat_cache.npz"
    assert out["act_feat_cache_signature"] == {"cache_id": "cache-sig"}
    assert out["act_feat_cache_weight_sha"] == "cache-sha"
    assert out["hiql_value_act_feat_signature"] == potential.act_feat_signature
    assert out["hiql_value_act_weight_sha"] == "sha-act"
