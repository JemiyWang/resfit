"""根因探针:真环境里 _check_grasp(piece_1) 到底触不触发?robot.gripper 是啥结构?

用 sync 模式(env 同进程,可直接 introspect robosuite env)。驱动基座 rollout,
逐步检查 grasp / first / second 三个谓词,看抓取阶段 grasp 是否曾为 True。
不改任何生产代码,纯只读探查。

用法:
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python -m resfit.rl_finetuning.chunk_residual.verify_grasp_debug \
    --base_dir bc_run_2026-05-31_14-59-16_dexmg-two-arm-three-piece-assembly_act/policy_step_199999
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import torch

from resfit.dexmg.environments.dexmg import create_vectorized_env
from resfit.rl_finetuning.chunk_residual.chunk_act_base import get_action_chunk
from resfit.rl_finetuning.chunk_residual.train_chunk_residual import build_base_policy


def _unwrap_robosuite(vec_env):
    e = vec_env.vec_env.envs[0]            # SyncVectorEnv → RobosuiteGymWrapper
    while not hasattr(e, "_check_grasp"):
        e = e.env
    return e


def _grasp_any(rob):
    """复刻 stage_detectors._grasped,但把异常/结果都暴露出来。"""
    try:
        hits = []
        for ri, robot in enumerate(rob.robots):
            for gi, gripper in enumerate(robot.gripper):
                g = rob._check_grasp(gripper=gripper,
                                     object_geoms=[x for x in rob.piece_1.contact_geoms])
                hits.append((ri, gi, type(gripper).__name__, bool(g)))
        return hits, None
    except Exception as ex:
        return None, f"{type(ex).__name__}: {ex}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", default="TwoArmThreePieceAssembly")
    p.add_argument("--base_dir", required=True)
    p.add_argument("--chunk_length", type=int, default=20)
    p.add_argument("--max_steps", type=int, default=800)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    base_policy = build_base_policy(args.base_dir, args.device)
    vec_env = create_vectorized_env(env_name=args.task, num_envs=1, device=args.device, debug=True)
    rob = _unwrap_robosuite(vec_env)
    print(f"[robosuite env] {type(rob).__name__}; robots={len(rob.robots)}")
    g0 = rob.robots[0].gripper
    print(f"[robot.gripper] type={type(g0).__name__}; 迭代元素类型={[type(x).__name__ for x in g0]}")
    print(f"[piece_1.contact_geoms] n={len(rob.piece_1.contact_geoms)} 例={list(rob.piece_1.contact_geoms)[:3]}")

    raw_obs, _ = vec_env.reset()
    base_policy.reset()
    grasp_true = first_true = second_true = 0
    err_once = None
    steps = 0
    while steps < args.max_steps:
        chunk = get_action_chunk(base_policy, raw_obs, args.chunk_length)
        for t in range(args.chunk_length):
            raw_obs, reward, term, trunc, info = vec_env.step(chunk[:, t])
            steps += 1
            hits, err = _grasp_any(rob)
            if err and err_once is None:
                err_once = err
            if hits and any(h[3] for h in hits):
                grasp_true += 1
            if rob._check_first_piece_is_assembled():
                first_true += 1
            if rob._check_second_piece_is_assembled():
                second_true += 1
            if bool((term | trunc).any()):
                base_policy.reset()
                break
            if steps >= args.max_steps:
                break

    print(f"\n{'#'*60}")
    print(f"[共 {steps} 步] _check_grasp(piece_1) 曾为 True 的步数 = {grasp_true}")
    print(f"            _check_first_piece_is_assembled True 步数 = {first_true}")
    print(f"            _check_second_piece_is_assembled True 步数 = {second_true}")
    if err_once:
        print(f"[grasp 调用异常] {err_once}")
    if grasp_true == 0 and not err_once:
        print("→ 根因:调用不报错,但 piece_1 抓取从不被检出(对象/几何/夹爪配对问题),非我代码 bug")
    elif err_once:
        print("→ 根因:_check_grasp 调用姿势报错,被 _grasped 的 try/except 吞成 False")
    vec_env.close()


if __name__ == "__main__":
    main()
