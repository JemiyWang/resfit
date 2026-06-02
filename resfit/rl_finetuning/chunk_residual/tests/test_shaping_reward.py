"""shaping_reward 纯函数:三模式 none|staged|potential 的奖励整形。

重点钉死:
- none 恒 0;
- staged 分支与旧 staged_bonus 逐位相同(行为不变);
- potential(PBS, Ng 1999)的 done 置零符号、Δ=0/回退取值、整轨迹 telescoping。
"""
import pytest

from resfit.rl_finetuning.chunk_residual.chunk_env_wrapper import (
    resolve_shaping_mode, shaping_reward, staged_bonus,
)


# ---------- none ----------
def test_none_is_zero_regardless():
    assert shaping_reward(0, 3, mode="none", bonus=1.0, gamma=0.99, done=False) == 0.0
    assert shaping_reward(0, 3, mode="none", bonus=1.0, gamma=0.99, done=True) == 0.0


# ---------- staged 逐位等于旧公式 ----------
def test_staged_matches_staged_bonus_positive():
    assert shaping_reward(0, 2, mode="staged", bonus=1.0, gamma=0.9, done=False) \
        == staged_bonus(0, 2, 1.0) == 2.0


def test_staged_matches_staged_bonus_zero_and_negative():
    assert shaping_reward(2, 2, mode="staged", bonus=1.0, gamma=0.9, done=False) == 0.0
    assert shaping_reward(3, 1, mode="staged", bonus=1.0, gamma=0.9, done=False) == 0.0


def test_staged_ignores_gamma_and_done():
    # staged 不看 gamma/done → 与旧 staged_bonus 完全一致
    assert shaping_reward(0, 2, mode="staged", bonus=1.0, gamma=0.5, done=True) == 2.0


# ---------- potential(PBS) ----------
def test_potential_step_not_done():
    # F = bonus·(γ·Φ' − Φ);Φ=stage
    assert shaping_reward(0, 1, mode="potential", bonus=1.0, gamma=0.9, done=False) \
        == pytest.approx(0.9)          # 1·(0.9·1 − 0)
    assert shaping_reward(1, 2, mode="potential", bonus=1.0, gamma=0.9, done=False) \
        == pytest.approx(0.8)          # 1·(0.9·2 − 1)


def test_potential_delta_zero_is_mildly_negative():
    # 停在同一 stage:γ 折现使其略负(PBS 的温和性,非 0)
    assert shaping_reward(2, 2, mode="potential", bonus=1.0, gamma=0.9, done=False) \
        == pytest.approx(-0.2)         # 1·(0.9·2 − 2)


def test_potential_terminal_zeroes_phi_next():
    # ⭐ 最易错点:done 时 Φ(s')=0 → F = bonus·(γ·0 − start) = −bonus·start
    assert shaping_reward(2, 3, mode="potential", bonus=1.0, gamma=0.9, done=True) \
        == pytest.approx(-2.0)         # 注意:end=3 被忽略,用 start=2
    # start=0 收尾 → 0
    assert shaping_reward(0, 2, mode="potential", bonus=1.0, gamma=0.9, done=True) \
        == pytest.approx(0.0)


def test_potential_bonus_zero_disabled():
    assert shaping_reward(0, 3, mode="potential", bonus=0.0, gamma=0.9, done=False) == 0.0


def test_potential_telescopes_over_trajectory():
    # 整轨迹 0→1→2→(done@2):shaping 总和应 telescoping 成与策略无关常数
    # = bonus·(γ·Φ_T − Φ_0) 展开 = bonus·(3γ − 3);bonus=1,γ=0.9 → −0.3
    g, b = 0.9, 1.0
    total = (shaping_reward(0, 1, mode="potential", bonus=b, gamma=g, done=False)
             + shaping_reward(1, 2, mode="potential", bonus=b, gamma=g, done=False)
             + shaping_reward(2, 2, mode="potential", bonus=b, gamma=g, done=True))
    assert total == pytest.approx(b * (3 * g - 3))   # −0.3


# ---------- 非法模式 ----------
def test_unknown_mode_raises():
    with pytest.raises(ValueError):
        shaping_reward(0, 1, mode="bogus", bonus=1.0, gamma=0.9, done=False)


# ---------- flag 解析:canonical --reward_shaping 优先,--staged_reward 为别名 ----------
def test_resolve_default_none():
    assert resolve_shaping_mode(None, False) == "none"


def test_resolve_staged_alias():
    # 仅旧 --staged_reward(当前在跑的命令)→ staged
    assert resolve_shaping_mode(None, True) == "staged"


def test_resolve_canonical_wins_over_alias():
    assert resolve_shaping_mode("potential", True) == "potential"
    assert resolve_shaping_mode("none", True) == "none"


def test_resolve_canonical_explicit():
    assert resolve_shaping_mode("staged", False) == "staged"
    assert resolve_shaping_mode("potential", False) == "potential"
