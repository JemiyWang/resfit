import numpy as np
import jax.numpy as jnp
import pytest
import serve_with_feat as _sf
from feature_policy import FeaturePolicy, pool_prefix, wrap_with_feature


def test_pool_last_picks_last_valid_token():
    out = jnp.array([[[1., 1], [2, 2], [9, 9]], [[3, 3], [4, 4], [5, 5]]])
    mask = jnp.array([[1, 1, 0], [1, 1, 1]])
    got = pool_prefix(out, mask, "last")
    assert jnp.allclose(got, jnp.array([[2., 2], [5, 5]]))   # 末有效 token,非 [:, -1]


def test_pool_mean_ignores_padding():
    out = jnp.array([[[1., 1], [3, 3], [9, 9]]])
    mask = jnp.array([[1, 1, 0]])
    got = pool_prefix(out, mask, "mean")
    assert jnp.allclose(got, jnp.array([[2., 2]]))           # (1+3)/2,padding 不计


def test_pool_unknown_raises():
    with pytest.raises(ValueError):
        pool_prefix(jnp.zeros((1, 1, 2)), jnp.ones((1, 1)), "bogus")


class _StubInner:
    metadata = {"ckpt": "stub"}

    def __init__(self):
        self.infer_calls = 0
        self.transform_calls = 0
        self._model = object()

    def _input_transform(self, x):
        self.transform_calls += 1
        return x

    def infer(self, obs, **kw):
        self.infer_calls += 1
        return {"actions": np.zeros((2, 7), np.float32), "state": np.asarray(obs["state"])}


def test_feature_policy_adds_prefix_feat_and_keeps_actions():
    inner = _StubInner()
    fp = FeaturePolicy(inner, pooling="last",
                       prefix_feat_fn=lambda _inner, _obs, _pool: np.arange(5, dtype=np.float32))
    out = fp.infer({"state": np.ones(3, np.float32)})
    assert inner.infer_calls == 1
    assert np.allclose(out["actions"], 0.0)               # 原 actions 透传
    assert np.allclose(out["prefix_feat"], np.arange(5))  # 新字段
    assert out["prefix_feat"].dtype == np.float32
    assert inner.transform_calls == 0


def test_feature_policy_metadata_passthrough():
    inner = _StubInner()
    fp = FeaturePolicy(inner, prefix_feat_fn=lambda *a: np.zeros(2, np.float32))
    assert fp.metadata == {"ckpt": "stub"}


def test_metadata_overrides_are_merged_without_mutating_inner():
    inner = _StubInner()
    fp = wrap_with_feature(
        inner,
        pooling="mean",
        prefix_feat_fn=lambda *args: np.zeros(2, np.float32),
        metadata_overrides={
            "serve_ckpt_id": "pi05_paper_awbc_19999",
            "asset_id": "pick_paper_all_merged",
        },
    )
    assert fp.metadata == {
        "ckpt": "stub",
        "serve_ckpt_id": "pi05_paper_awbc_19999",
        "asset_id": "pick_paper_all_merged",
    }
    assert inner.metadata == {"ckpt": "stub"}


def test_wrap_with_feature_returns_feature_policy():
    inner = _StubInner()
    fp = wrap_with_feature(inner, pooling="mean", prefix_feat_fn=lambda *a: np.zeros(2, np.float32))
    assert isinstance(fp, FeaturePolicy) and fp.pooling == "mean"


class _FakeArgs:
    def __init__(self, port, pooling):
        self.host = "0.0.0.0"
        self.port = port
        self.pooling = pooling
        self.default_prompt = None


def test_serve_with_feat_wraps_and_serves(monkeypatch):
    captured = {}
    monkeypatch.setattr(_sf, "_create_policy", lambda args: _StubInner())
    monkeypatch.setattr(_sf, "_serve",
                        lambda policy, host, port: captured.update(policy=policy, port=port))
    _sf.run(_FakeArgs(port=8000, pooling="last"))
    assert isinstance(captured["policy"], FeaturePolicy)
    assert captured["policy"].pooling == "last" and captured["port"] == 8000
