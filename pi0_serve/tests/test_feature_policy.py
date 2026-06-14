import jax.numpy as jnp
import pytest
from feature_policy import pool_prefix


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
