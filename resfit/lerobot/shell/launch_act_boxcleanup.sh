#!/usr/bin/env bash

# Launch ACT BC training for TwoArmBoxCleanup task from DexMimicGen
# 注意:官方没提供这个任务的 BC 启动脚本,本脚本是照着 cansorting/coffee 补的。
# 关键值已对照 resfit/rl_finetuning/config/residual_td3.py 的 ResidualTD3BoxCleanConfig 核对:
#   task / eval_env = TwoArmBoxCleanup
#   dataset         = ankile/dexmg-two-arm-box-cleanup (1000 demos)
#   相机 agentview 在该环境的 rl_camera 列表里,可用作 eval 视频键
# --steps 论文未规定(BC 步数论文没给),这里取 200000(双臂硬任务,偏保守),可按需调。

# 指定 GPU:改下面的数字即可(命令行前缀 CUDA_VISIBLE_DEVICES=x 可临时覆盖)
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

# Headless 离屏渲染:eval rollout 必须用 EGL,否则到第一次评测会崩
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# 自带 conda 环境:依赖装在 residual env 里,无论当前 env 是哪个都能跑对
conda run -n residual --no-capture-output \
python -m resfit.lerobot.scripts.train_bc_dexmg \
    --dataset ankile/dexmg-two-arm-box-cleanup \
    --policy act \
    --batch_size 256 \
    --wandb_project dexmg-boxcleanup-bc \
    --wandb_enable \
    --eval_env TwoArmBoxCleanup \
    --rollout_freq 5000 \
    --steps 200000 \
    --eval_video_key observation.images.agentview \
    --eval_num_envs 16 \
    --eval_num_episodes 50 \
    --num_workers 8 \
    --log_freq 100 \
    --save_freq 10000
