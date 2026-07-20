"""Task 14:train_hiql_value.py 支持 pi0_feat(kai0 ψ)单状态 V。

block 实机任务没有可用的 ACT encoder,V 只能吃 kai0 的 prefix_feat(ψ)——而 ψ 正是
基座推理的副产物,与 wm_bridge 的 Kai0HiqlScorer 同源。

pi0_feat 是纯缓存路(训练时不连 serve),故可用合成缓存把整条路端到端测掉。
"""
import numpy as np
import pytest
import torch

from resfit.rl_finetuning.chunk_residual.hiql_value import (
    ValueMLP, load_value, save_value,
)
from resfit.rl_finetuning.chunk_residual.pi0_feat_cache import save_pi0_feat_cache
from resfit.rl_finetuning.chunk_residual.train_hiql_value import (
    assert_pi0_caches_samesource, build_parser, validate_terminal_reward_cfg,
)

CORE_SIG = {
    "serve_ckpt_id": "kai0-block-v1",
    "image_keys": ["observation.images.top_head", "observation.images.hand_left",
                   "observation.images.hand_right"],
    "proprio_key": "observation.state",
    "pooling": "mean",
    "prompt": "build block",
}


def _make_cache(path, raw_seqs, *, sig_over=None, num_demos=None):
    """造一个合成 pi0_feat 缓存:seqs 用自己的 stats 标准化(与真实 build 同款)。"""
    allf = np.concatenate(raw_seqs, axis=0)
    mean = allf.mean(axis=0).astype(np.float32)
    std = np.maximum(allf.std(axis=0), 1e-6).astype(np.float32)
    seqs_std = [((s - mean) / std).astype(np.float32) for s in raw_seqs]
    sig = dict(CORE_SIG, num_demos=num_demos)
    if sig_over:
        sig.update(sig_over)
    save_pi0_feat_cache(str(path), seqs_std, (mean, std), signature=sig)
    return sig


# ---- save/load_value 的 pi0_feat_signature 往返(wm_bridge 同源校验的锚) ----

def test_save_load_value_roundtrips_pi0_feat_signature():
    import tempfile, os
    m = ValueMLP(4, 8)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "v.pt")
        save_value(p, m, v_stats={"min": 0.0, "max": 1.0, "mean": 0.5},
                   mean=torch.zeros(4), std=torch.ones(4), dataset_id="block",
                   state_mode="pi0_feat", pi0_feat_signature=CORE_SIG)
        _, info = load_value(p)
        assert info["state_mode"] == "pi0_feat"
        assert info["pi0_feat_signature"]["serve_ckpt_id"] == "kai0-block-v1"
        assert info["pi0_feat_signature"]["prompt"] == "build block"


def test_load_value_pi0_signature_absent_is_none():
    """既有 value.pt(无 pi0 签名)读回应为 None,不炸。"""
    import tempfile, os
    m = ValueMLP(3, 8)
    with tempfile.TemporaryDirectory() as d:
        p = os.path.join(d, "v.pt")
        save_value(p, m, v_stats={"min": 0.0, "max": 1.0, "mean": 0.5},
                   mean=torch.zeros(3), std=torch.ones(3), dataset_id="x")
        _, info = load_value(p)
        assert info["pi0_feat_signature"] is None


# ---- CLI ----

def test_state_mode_accepts_pi0_feat():
    args = build_parser().parse_args(["--dataset", "y", "--hdf5", "x",
                                      "--state_mode", "pi0_feat"])
    assert args.state_mode == "pi0_feat"


def test_pi0_feat_args_are_exposed():
    args = build_parser().parse_args([
        "--dataset", "y", "--hdf5", "x", "--state_mode", "pi0_feat",
        "--pi0_feat_cache", "c.npz", "--pi0_serve_ckpt_id", "kai0-block-v1",
        "--pi0_image_keys", "a,b,c", "--pi0_proprio_key", "observation.state",
        "--pi0_prompt", "build block", "--pi0_pooling", "mean"])
    assert args.pi0_serve_ckpt_id == "kai0-block-v1"
    assert args.pi0_image_keys == ["a", "b", "c"]
    assert args.pi0_pooling == "mean"


def test_success_signed_allows_pi0_feat():
    """block 的 V 走 pi0_feat + success_signed,必须放行。"""
    args = build_parser().parse_args([
        "--dataset", "y", "--state_mode", "pi0_feat",
        "--terminal_reward_mode", "success_signed",
        "--success_dataset", "s.npz", "--failure_dataset", "f.npz"])
    validate_terminal_reward_cfg(args)   # 不抛


def test_success_signed_still_rejects_eef():
    args = build_parser().parse_args([
        "--dataset", "y", "--hdf5", "x", "--state_mode", "eef",
        "--terminal_reward_mode", "success_signed",
        "--success_dataset", "a", "--failure_dataset", "b"])
    with pytest.raises(SystemExit):
        validate_terminal_reward_cfg(args)


# ---- 跨缓存同源断言:成功集/失败集的 ψ 必须来自同一个 kai0 ----

def test_samesource_passes_when_core_fields_match(tmp_path):
    s = _make_cache(tmp_path / "s.npz", [np.ones((5, 3), np.float32)])
    f = _make_cache(tmp_path / "f.npz", [np.zeros((5, 3), np.float32)])
    assert_pi0_caches_samesource([s, f])          # 不抛


def test_samesource_raises_on_different_serve_ckpt(tmp_path):
    """成功集与失败集的 ψ 若来自不同 kai0,特征空间不同 → 必须当场报错。"""
    s = _make_cache(tmp_path / "s.npz", [np.ones((5, 3), np.float32)])
    f = _make_cache(tmp_path / "f.npz", [np.zeros((5, 3), np.float32)],
                    sig_over={"serve_ckpt_id": "kai0-OTHER"})
    with pytest.raises(ValueError, match="serve_ckpt_id"):
        assert_pi0_caches_samesource([s, f])


def test_samesource_raises_on_different_prompt(tmp_path):
    s = _make_cache(tmp_path / "s.npz", [np.ones((5, 3), np.float32)])
    f = _make_cache(tmp_path / "f.npz", [np.zeros((5, 3), np.float32)],
                    sig_over={"prompt": "something else"})
    with pytest.raises(ValueError, match="prompt"):
        assert_pi0_caches_samesource([s, f])


# ---- 端到端:pi0_feat + success_signed 跑通 main() 并产出正确 value.pt ----

def _run_main(argv, monkeypatch):
    import sys
    from resfit.rl_finetuning.chunk_residual import train_hiql_value as T
    monkeypatch.setattr(sys, "argv", ["train_hiql_value"] + argv)
    T.main()


def test_pi0_feat_success_signed_end_to_end(tmp_path, monkeypatch):
    """成功集/失败集两个缓存 → 联合标准化 → ±1 终端奖励 → value.pt。

    这是 block 实机 V 的真实训练路径,只是把 kai0 产出的缓存换成合成缓存。
    """
    rng = np.random.default_rng(0)
    # 两集特征分布不同(模拟各自 build 出的缓存 stats 不一致)
    succ = [rng.normal(4.0, 1.5, (12, 3)).astype(np.float32) for _ in range(3)]
    fail = [rng.normal(-2.0, 0.7, (10, 3)).astype(np.float32) for _ in range(2)]
    _make_cache(tmp_path / "s.npz", succ)
    _make_cache(tmp_path / "f.npz", fail)
    out = tmp_path / "block_value.pt"

    _run_main([
        "--dataset", "block", "--output", str(out),
        "--state_mode", "pi0_feat",
        "--pi0_serve_ckpt_id", CORE_SIG["serve_ckpt_id"],
        "--pi0_image_keys", ",".join(CORE_SIG["image_keys"]),
        "--pi0_proprio_key", CORE_SIG["proprio_key"],
        "--pi0_pooling", CORE_SIG["pooling"], "--pi0_prompt", CORE_SIG["prompt"],
        "--terminal_reward_mode", "success_signed",
        "--success_dataset", str(tmp_path / "s.npz"),
        "--failure_dataset", str(tmp_path / "f.npz"),
        "--steps", "300",
    ], monkeypatch)

    model, info = load_value(str(out))
    assert info["state_mode"] == "pi0_feat"
    # ψ 同源锚必须被写进去(wm_bridge 启动时靠它校验)
    assert info["pi0_feat_signature"]["serve_ckpt_id"] == CORE_SIG["serve_ckpt_id"]
    assert info["state_dim"] == 3
    # 存的是联合 stats:用它标准化两集原始特征,应逐维 mean≈0/std≈1
    mean = np.asarray(info["mean"], np.float32)
    std = np.asarray(info["std"], np.float32)
    allraw = np.concatenate(succ + fail, axis=0)
    z = (allraw - mean) / std
    np.testing.assert_allclose(z.mean(axis=0), 0.0, atol=1e-4)
    np.testing.assert_allclose(z.std(axis=0), 1.0, atol=1e-4)


def test_pi0_feat_success_signed_learns_failure_is_worse(tmp_path, monkeypatch):
    """±1 终端奖励生效:失败轨迹末态的 V 应低于成功轨迹末态。"""
    succ = [np.linspace(0, 1, 30, dtype=np.float32).reshape(10, 3)]
    fail = [np.linspace(0, -1, 30, dtype=np.float32).reshape(10, 3)]
    _make_cache(tmp_path / "s.npz", succ)
    _make_cache(tmp_path / "f.npz", fail)
    out = tmp_path / "v.pt"

    _run_main([
        "--dataset", "block", "--output", str(out), "--state_mode", "pi0_feat",
        "--pi0_serve_ckpt_id", CORE_SIG["serve_ckpt_id"],
        "--pi0_image_keys", ",".join(CORE_SIG["image_keys"]),
        "--pi0_proprio_key", CORE_SIG["proprio_key"],
        "--pi0_pooling", CORE_SIG["pooling"], "--pi0_prompt", CORE_SIG["prompt"],
        "--terminal_reward_mode", "success_signed",
        "--success_dataset", str(tmp_path / "s.npz"),
        "--failure_dataset", str(tmp_path / "f.npz"),
        "--steps", "3000",
    ], monkeypatch)

    model, info = load_value(str(out))
    mean = np.asarray(info["mean"], np.float32)
    std = np.asarray(info["std"], np.float32)
    with torch.no_grad():
        v_s = model(torch.from_numpy((succ[0][-1] - mean) / std)).item()
        v_f = model(torch.from_numpy((fail[0][-1] - mean) / std)).item()
    assert v_s > v_f


def test_pi0_feat_rejects_mismatched_cache_signature(tmp_path, monkeypatch):
    """CLI 指向了错误的缓存(serve_ckpt_id 不符)→ 当场报错,不静默训。"""
    _make_cache(tmp_path / "s.npz", [np.ones((6, 3), np.float32)])
    _make_cache(tmp_path / "f.npz", [np.zeros((6, 3), np.float32)])
    with pytest.raises(ValueError, match="serve_ckpt_id"):
        _run_main([
            "--dataset", "block", "--output", str(tmp_path / "v.pt"),
            "--state_mode", "pi0_feat",
            "--pi0_serve_ckpt_id", "WRONG-kai0",          # 与缓存不符
            "--pi0_image_keys", ",".join(CORE_SIG["image_keys"]),
            "--pi0_proprio_key", CORE_SIG["proprio_key"],
            "--pi0_pooling", CORE_SIG["pooling"], "--pi0_prompt", CORE_SIG["prompt"],
            "--terminal_reward_mode", "success_signed",
            "--success_dataset", str(tmp_path / "s.npz"),
            "--failure_dataset", str(tmp_path / "f.npz"),
            "--steps", "10",
        ], monkeypatch)
