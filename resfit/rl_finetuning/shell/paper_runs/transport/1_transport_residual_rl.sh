#!/bin/bash
# Residual-TD3 RL for TwoArmTransport (robosuite opposed 双臂, dexmg-two-arm-transport 数据).
# 启动: bash resfit/rl_finetuning/shell/run_rl.sh transport <gpu>
# 前置:dexmg-transport-bc 已产出 _best,且 residual_td3.py 里
#       ResidualTD3TwoArmTransportConfig.base_policy.wandb_id 已填真实 run_id。
# 注意:transport 是长 horizon(800)任务,gamma 用 0.998(与 RLPD transport 一致);
#       相机 shouldercamera0/1 已在 config 里设好;opposed 构型由 dexmg.py 自动处理。

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_two_arm_transport_config \
    algo.prefetch_batches=4 \
    algo.n_step=5 \
    algo.gamma=0.998 \
    algo.learning_starts=10_000 \
    algo.critic_warmup_steps=10_000 \
    algo.num_updates_per_iteration=4 \
    algo.stddev_max=0.025 \
    algo.stddev_min=0.025 \
    algo.buffer_size=300_000 \
    agent.actor.action_scale=0.2 \
    agent.actor_lr=1e-6 \
    wandb.project=dexmg-transport-final \
    wandb.name=residual-rl \
    wandb.notes=paper_runs/transport/1_transport_residual_rl \
    wandb.group=residual-rl \
    debug=false
