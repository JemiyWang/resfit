"""真 env 验证 lifttray_stage:从 hdf5 起不渲染 env,逐帧 set_state 跑检测器,打印每条 demo 的
瞬时 stage 分布与终段闩锁。确认 set_state 后 check_contact("pot_base",obj) 正确、分布呈 0→1→2→3。
用法: MUJOCO_GL=egl PYTHONPATH=/mnt/mnt/data/resfit \
      /mnt/mnt/data/envs/residual/bin/python -m resfit.rl_finetuning.chunk_residual.verify_stage_lifttray [n]
"""
import sys

import h5py
import numpy as np

from resfit.rl_finetuning.chunk_residual.offline_stage_replay import (
    make_replay_env,
    replay_instant_stages,
)
from resfit.rl_finetuning.chunk_residual.stage_detectors import (
    NUM_STAGES,
    get_stage_detector,
)

DATA = "resfit/dataset/two_arm_lift_tray.hdf5"


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    env, env_name = make_replay_env(DATA)
    det = get_stage_detector(env_name)
    assert det is not None, f"无 {env_name} 检测器"
    goal = NUM_STAGES[env_name] - 1
    ok = 0
    with h5py.File(DATA, "r") as f:
        keys = sorted(f["data"].keys(), key=lambda k: int(k.split("_")[1]))[:n]
        for k in keys:
            d = f["data"][k]
            stages = replay_instant_stages(
                env, d["states"][:], model_file=d.attrs["model_file"], detector=det)
            latched = int(np.maximum.accumulate(stages)[-1])
            uniq, cnt = np.unique(stages, return_counts=True)
            passed = latched == goal and set(range(goal + 1)) <= set(uniq.tolist())
            ok += passed
            print(f"{k}: instant={dict(zip(uniq.tolist(), cnt.tolist()))} "
                  f"final_latched={latched}/{goal} {'OK' if passed else 'CHECK'}")
    print(f"\n{ok}/{n} demo 达终段且历经 0..{goal}")


if __name__ == "__main__":
    main()
