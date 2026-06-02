#!/usr/bin/env bash

# Launch ACT BC training for Square task from Robomimic
# This matches the example command provided by the user

# 指定 GPU:改下面的数字即可(命令行前缀 CUDA_VISIBLE_DEVICES=x 可临时覆盖)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"

# Headless 离屏渲染:eval rollout 必须用 EGL,否则到第一次评测会崩
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# 自带 conda 环境:依赖装在 residual env 里,无论当前 env 是哪个都能跑对
conda run -n residual --no-capture-output \
python -m resfit.lerobot.scripts.train_bc_dexmg \
    --dataset ankile/robomimic-mh-square-image \
    --policy act \
    --batch_size 256 \
    --wandb_project robomimic-square-bc \
    --wandb_enable \
    --eval_env Square \
    --rollout_freq 5000 \
    --steps 50000 \
    --eval_video_key observation.images.agentview \
    --eval_num_envs 16 \
    --eval_num_episodes 50 \
    --num_workers 8 \
    --log_freq 100 \
    --save_freq 1000
