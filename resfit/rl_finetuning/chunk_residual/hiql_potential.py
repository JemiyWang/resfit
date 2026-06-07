"""把 ③a 的冻结 value 当 PBS 势函数 Φ(模块 ③b)。

设计见 docs/superpowers/specs/2026-06-07-hiql-potential-design.md。
potential_shaping 是通用 PBS 公式(Φ 可为 int stage 或 V(state)*scale);
HiqlPotential 加载冻结 value 并把标准化 state 映射到 Φ。
"""
import torch

from resfit.rl_finetuning.chunk_residual.hiql_value import load_value


def potential_shaping(phi_start, phi_next, *, bonus, gamma, done):
    """通用 PBS 整形:F = bonus*(gamma*phi_next - phi_start),done 时 phi_next=0。

    phi_start/phi_next 可为 float 或单元素张量(online b=1);返回 float。
    """
    pn = 0.0 if done else float(phi_next)
    return bonus * (gamma * pn - float(phi_start))
