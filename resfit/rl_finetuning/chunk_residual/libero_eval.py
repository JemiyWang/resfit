"""LIBERO 在线评测:env-无关的成功率/回报 rollout。

为什么不复用 run_dexmg_evaluation:那个模块顶层 `from resfit.dexmg...dexmg import
VectorizedEnvWrapper`(robosuite-1.5 专属),libero 环境(robosuite-1.4.1)import 不动它;
且它带 dexmg 专属的 Q 轨迹绘图 / 视频 / subgoal 注入。这里只保留与环境无关的核心 rollout
(reset → agent.act(eval_mode) → env.step → 按 done 统计),指标口径与 run_dexmg_evaluation
完全一致(success = reward==1.0,return = 单 episode reward 之和),返回相同的 eval/* 键,
供 train_chunk_residual 的 eval block 与 wandb_logging.build_eval_log_dict 直接消费。
"""
from __future__ import annotations

import numpy as np
import torch


def _inject_subgoal(subgoal, obs, base_policy):
    """LIBERO eval 子目标注入:返回该 obs 的 z(调用方写进 obs['observation.subgoal'])。
    对称 evaluate_dexmg._eval_inject_subgoal,但 LIBERO 无 rel_piece,当前只支持 pi0_feat。"""
    if subgoal.state_mode == "pi0_feat":
        return subgoal.subgoal_online(obs, prefix_feat=base_policy.last_prefix_feat())
    raise NotImplementedError(
        f"LIBERO eval 子目标注入目前只支持 state_mode=pi0_feat,得到 {subgoal.state_mode!r}")


def run_libero_evaluation(*, env, agent, num_episodes: int, device: str = "cpu",
                          subgoal=None, base_policy=None) -> dict:
    """在 eval_env(ChunkResidualEnvWrapper 包 libero 向量 env)上跑 num_episodes 个 episode,
    返回 {eval/success_rate, eval/mean_return, eval/mean_successful_episode_length}。

    与 run_dexmg_evaluation 的对齐点:
    - agent.eval() 进入评测,结束 agent.train(True) 还原(否则后续 update 用错模式)。
    - 确定性策略:agent.act(eval_mode=True, stddev=0.0, cpu=False)。
    - 向量化:env.num_envs 个 env 同时跑,按 env_idx 收集,凑满 num_episodes 即停。
    - autoreset:env 是 AutoresetMode.SAME_STEP,done 后该 env 在同一 step 自动重置,
      next_obs 已是新 episode 首帧,故 done 后只清 per-env buffer、obs=next_obs 续跑。
    - success = (该步 reward == 1.0);return = 该 episode 各步 reward 之和。
    device 仅为调用方签名对齐保留(动作设备由 agent 决定),此处不强制搬运。
    """
    if subgoal is not None and base_policy is None:
        raise ValueError("run_libero_evaluation: subgoal!=None 时必须传 base_policy(读 last_prefix_feat)")
    agent.eval()
    num_envs = env.num_envs if hasattr(env, "num_envs") else 1

    ep_rewards: list[list[float]] = [[] for _ in range(num_envs)]
    successes: list[bool] = []
    returns: list[float] = []
    episode_lengths: list[int] = []

    done_episodes = 0
    obs, _ = env.reset()
    if subgoal is not None:
        obs["observation.subgoal"] = _inject_subgoal(subgoal, obs, base_policy)

    dots = ["."] * num_episodes
    print(f"[libero-eval] {num_episodes} episodes: {''.join(dots)}", end="", flush=True)

    try:
        while done_episodes < num_episodes:
            with torch.no_grad():
                actions = agent.act(obs, eval_mode=True, stddev=0.0, cpu=False)

            next_obs, reward, terminated, truncated, _info = env.step(actions)
            if subgoal is not None:
                next_obs["observation.subgoal"] = _inject_subgoal(subgoal, next_obs, base_policy)
            done_flags = terminated | truncated

            for env_idx in range(num_envs):
                ep_rewards[env_idx].append(float(reward[env_idx].item()))

                if bool(done_flags[env_idx]):
                    is_success = bool(reward[env_idx].item() == 1.0)   # 对齐 run_dexmg_evaluation
                    successes.append(is_success)
                    returns.append(float(sum(ep_rewards[env_idx])))
                    episode_lengths.append(len(ep_rewards[env_idx]))
                    ep_rewards[env_idx].clear()   # autoreset:为新 episode 清空

                    dots[done_episodes] = "✓" if is_success else "✗"
                    print(f"\r[libero-eval] {num_episodes} episodes: {''.join(dots)}",
                          end="", flush=True)

                    done_episodes += 1
                    if done_episodes == num_episodes:
                        break

            obs = next_obs
        print()  # 收尾换行
    finally:
        agent.train(True)   # 无论是否异常都还原训练模式

    success_rate = float(np.mean(successes)) if successes else 0.0
    mean_return = float(np.mean(returns)) if returns else 0.0
    succ_lengths = [L for L, s in zip(episode_lengths, successes) if s]
    mean_succ_len = float(np.mean(succ_lengths)) if succ_lengths else 0.0

    return {
        "eval/success_rate": success_rate,
        "eval/mean_return": mean_return,
        "eval/mean_successful_episode_length": mean_succ_len,
    }
