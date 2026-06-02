#!/usr/bin/env bash

# Launch ACT BC training for TwoArmCoffee task from DexMimicGen
# This script uses the two-arm coffee dataset at ankile/dexmg-two-arm-coffee

# 指定 GPU:改下面的数字即可(命令行前缀 CUDA_VISIBLE_DEVICES=x 可临时覆盖)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-4}"

# Headless 离屏渲染:eval rollout 必须用 EGL,否则到第一次评测会崩
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# 自带 conda 环境:依赖装在 residual env 里,无论当前 env 是哪个都能跑对
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
    --wandb_enable
