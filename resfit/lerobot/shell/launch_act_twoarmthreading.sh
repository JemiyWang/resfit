#!/usr/bin/env bash

# Launch ACT BC training for TwoArmThreading (DexMimicGen, Panda 双臂).
# 数据集: ankile/dexmg-two-arm-threading
# 照 launch_act_boxcleanup.sh 完整版补齐(原裸版缺 EGL/GPU/conda,会在第一次 eval 崩)。
#   - eval_video_key=agentview:该环境相机集为 agentview + robot0/1_eye_in_hand;原写 frontview
#     是错的(此环境没有 frontview 相机),已改为 agentview
#   - steps=200000:双臂硬任务,原 50000 偏少
#   - 关键值对照 residual_td3.py 的 ResidualTD3TwoArmThreadingConfig(dataset / eval_env / 相机)

# 指定 GPU:改下面数字或用 run_bc.sh <task> <gpu> 覆盖
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-3}"

# Headless 离屏渲染:eval rollout 必须用 EGL,否则到第一次评测会崩
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl

# resfit 没装成包,代码靠 cwd 导入;切到仓库根目录,这样从任何路径 bash 本脚本都能跑
cd "$(dirname "${BASH_SOURCE[0]}")/../../.." || exit 1

# 自带 conda 环境:依赖装在 residual env 里,无论当前 env 是哪个都能跑对
conda run -n residual --no-capture-output \
python -m resfit.lerobot.scripts.train_bc_dexmg \
    --dataset ankile/dexmg-two-arm-threading \
    --policy act \
    --batch_size 256 \
    --wandb_project dexmg-twoarmthreading-bc \
    --wandb_enable \
    --eval_env TwoArmThreading \
    --rollout_freq 5000 \
    --steps 200000 \
    --eval_video_key observation.images.agentview \
    --eval_num_envs 16 \
    --eval_num_episodes 50 \
    --num_workers 8 \
    --log_freq 100 \
    --save_freq 10000
