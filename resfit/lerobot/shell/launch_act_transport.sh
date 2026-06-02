#!/usr/bin/env bash

# Launch ACT BC training for Transport (robosuite TwoArmTransport, opposed 双臂).
# 数据集: ankile/dexmg-two-arm-transport(dexmimicgen 版,1029 集)
# 照 launch_act_boxcleanup.sh 完整版补齐(原裸版缺 EGL/GPU/conda,会在第一次 eval 崩)。
#   - 该数据集含 5 个相机,但 Transport env 只产出 shouldercamera0/1,故用 --policy_cameras
#     过滤到这俩,否则策略输入维度和 eval 环境对不上、第一次评测就崩
#   - eval_video_key=shouldercamera0:Transport 的相机集是 shouldercamera0/1(没有 agentview)
#   - steps=200000:长 horizon(800)硬任务
#   - 关键值对照 residual_td3.py 的 ResidualTD3TwoArmTransportConfig(dataset / eval_env / 相机)

# 指定 GPU:改下面数字或用 run_bc.sh <task> <gpu> 覆盖
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-5}"

# Headless 离屏渲染:eval rollout 必须用 EGL,否则到第一次评测会崩
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# resfit 没装成包,代码靠 cwd 导入;切到仓库根目录,这样从任何路径 bash 本脚本都能跑
cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1

# 自带 conda 环境:依赖装在 residual env 里,无论当前 env 是哪个都能跑对
conda run -n residual --no-capture-output \
python -m resfit.lerobot.scripts.train_bc_dexmg \
    --dataset ankile/dexmg-two-arm-transport \
    --policy act \
    --batch_size 256 \
    --wandb_project dexmg-transport-bc \
    --wandb_enable \
    --eval_env Transport \
    --policy_cameras shouldercamera0 shouldercamera1 \
    --rollout_freq 5000 \
    --steps 200000 \
    --eval_video_key observation.images.shouldercamera0 \
    --eval_num_envs 16 \
    --eval_num_episodes 50 \
    --num_workers 8 \
    --log_freq 100 \
    --save_freq 10000
