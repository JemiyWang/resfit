#!/usr/bin/env bash

# Launch ACT BC training for Can task from Robomimic

# Headless 离屏渲染:eval rollout 必须用 EGL,否则到第一次评测会崩
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# 自带 conda 环境:依赖装在 residual env 里,无论当前激活的是哪个 env 都能跑对
# --no-capture-output 让日志实时输出(否则 conda run 会缓冲,tee 不到东西)
conda run -n residual --no-capture-output \
python -m resfit.lerobot.scripts.train_bc_dexmg \
    --dataset ankile/robomimic-mh-can-image \
    --policy act \
    --batch_size 256 \
    --wandb_project robomimic-can-bc \
    --wandb_enable \
    --eval_env Can \
    --rollout_freq 5000 \
    --steps 50000 \
    --eval_video_key observation.images.agentview \
    --eval_num_envs 16 \
    --eval_num_episodes 50 \
    --num_workers 8 \
    --log_freq 100 \
    --save_freq 1000
