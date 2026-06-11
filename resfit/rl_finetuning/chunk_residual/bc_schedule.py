"""demo-BC 系数的训练步数调度(模块②a 增强)。

当前 BC 系数在 train_chunk_residual.py 建 agent 时一次性写死(cfg.agent.bc_loss_coef)。
本模块提供线性衰减:训练早期强约束(锚专家防塌方)、后期放手(残差自由探索)。
纯函数、无重依赖,便于单测;由训练循环每步调用后写回 agent.cfg.bc_loss_coef。
"""


def linear_bc_coef(env_steps, *, c0, c_final, total_steps):
    """BC 系数线性衰减。

    env_steps 从 0 到 total_steps 时,返回值从 c0 线性变到 c_final;区间外 clip
    (env_steps<0 -> c0,env_steps>=total_steps -> c_final)。total_steps<=0 时
    退化为 c_final(防除零)。

    参数:
      env_steps:   当前已采样环境步数。
      c0:          起始系数(= --demo_bc_coef)。
      c_final:     终值系数(= --bc_coef_final,floor)。
      total_steps: 衰减区间长度(= --total_env_steps)。
    """
    if total_steps <= 0:
        return c_final
    progress = min(max(env_steps / total_steps, 0.0), 1.0)
    return c0 + (c_final - c0) * progress
