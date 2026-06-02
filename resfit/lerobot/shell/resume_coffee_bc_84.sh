#!/usr/bin/env bash
# 一次性恢复脚本:Coffee BC eval 分辨率 bug 修复后,续训重评。
#
# 背景:launch_act_twoarmcoffee.sh 之前误设 --eval_camera_size 224,但策略是 84x84 训练的,
# 导致 eval 全程喂 224 → 分布外 → 30+ 次 eval 全 0%,从没出过 best,Coffee RL 因拉不到
# run_gbiv6udg_best 而被阻塞。launch 脚本已改回 --eval_camera_size 84(录像仍用 render 224)。
#
# 本脚本:用 --resume_run_id 续上同一个 wandb run(gbiv6udg),从其 latest artifact(step 190000)
# 恢复 policy+optimizer+step,继续训到 200000。续训后第一次 eval 在 step 195000,这次用 84 评测,
# 只要成功率 >0 就会触发 "New best" → 存 best/ → 推 run_gbiv6udg_best:latest,RL 即可解除阻塞。
#
# 用法: bash resfit/rl_finetuning/../lerobot/shell/resume_coffee_bc_84.sh   (从仓库根目录跑)

set -euo pipefail
cd "$(dirname "$0")/../../.."   # -> 仓库根目录

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# 给刚被 kill 的旧进程一点时间彻底释放 wandb run 连接,避免 resume="must" 撞 "run is running"
sleep 15

conda run -n residual --no-capture-output \
python -m resfit.lerobot.scripts.train_bc_dexmg \
    --dataset ankile/dexmg-two-arm-coffee \
    --policy act \
    --steps 200000 \
    --batch_size 256 \
    --wandb_project dexmg-coffee-bc \
    --eval_env TwoArmCoffee \
    --rollout_freq 5000 \
    --eval_video_key observation.images.frontview \
    --eval_render_size 224 \
    --eval_camera_size 84 \
    --eval_num_envs 16 \
    --eval_num_episodes 50 \
    --num_workers 8 \
    --wandb_enable \
    --resume_run_id gbiv6udg
