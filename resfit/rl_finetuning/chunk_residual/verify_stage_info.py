"""验证特权 stage 标签经 RobosuiteGymWrapper.step 的 info 跨 spawn 边界透出。

驱动真实工厂 create_vectorized_env(默认 async/spawn，即训练路径):
  - 有检测器的任务(TwoArmThreePieceAssembly): step 的 info 必须含 "stage_id"，
    零动作下抓不到东西 → 应恒为 0(int，逐 env 一个)。
  - 无检测器的任务(TwoArmBoxCleanup): info 不应出现 "stage_id"(优雅退化)。

注:reset 的 info 为空(RobosuiteGymWrapper.reset 返回 {})，stage 只在 step 出。
注:零动作无法触发 stage>0(需真实抓取);此处只验证“通道通 + 起点值正确 + 负控”，
   stage 递增留待真实 rollout(基座 BC 训好后)确认。

用法(仓库根目录, conda env residual):
  CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  python -m resfit.rl_finetuning.chunk_residual.verify_stage_info
退出码: 0=正控+负控均符合预期; 1=否。
"""
from __future__ import annotations

import traceback

import numpy as np

from resfit.dexmg.environments.dexmg import create_vectorized_env


def probe(task: str, n_steps: int = 4):
    print(f"\n{'='*68}\n[probe] task={task}  (async/spawn, 训练默认路径)\n{'='*68}")
    vec_env = None
    seen = []          # 每步 info 是否含 stage_id 及其值
    reset_keys = step_keys = None
    try:
        vec_env = create_vectorized_env(env_name=task, num_envs=1, device="cpu")
        _, rinfo = vec_env.reset()
        reset_keys = list(rinfo.keys())
        act = np.zeros((1, vec_env.action_space.shape[-1]), dtype=np.float32)
        for t in range(n_steps):
            _, _, _, _, info = vec_env.step(act)
            if t == 0:
                step_keys = list(info.keys())
            if "stage_id" in info:
                v = info["stage_id"]
                seen.append((True, np.asarray(v).tolist()))
            else:
                seen.append((False, None))
        print(f"[reset] info keys: {reset_keys}")
        print(f"[step]  info keys: {step_keys}")
        print(f"[step]  stage_id 逐步: {seen}")
    except Exception:
        print("[ERROR] 构建/驱动失败:")
        traceback.print_exc()
    finally:
        if vec_env is not None:
            try:
                vec_env.close()
            except Exception:
                pass
    return seen


def main():
    # 正控: 有检测器 → stage_id 出现且恒 0
    tp = probe("TwoArmThreePieceAssembly")
    pos_ok = len(tp) > 0 and all(present and vals == [0] for present, vals in tp)

    # 负控: 无检测器 → stage_id 不应出现
    bc = probe("TwoArmBoxCleanup")
    neg_ok = len(bc) > 0 and all(not present for present, _ in bc)

    print(f"\n{'#'*68}")
    print(f"[正控] ThreePiece stage_id 出现且恒0: {pos_ok}")
    print(f"[负控] BoxCleanup 无 stage_id 键:     {neg_ok}")
    raise SystemExit(0 if (pos_ok and neg_ok) else 1)


if __name__ == "__main__":
    main()
