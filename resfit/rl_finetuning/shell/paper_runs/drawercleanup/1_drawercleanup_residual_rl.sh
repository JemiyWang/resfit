#!/bin/bash
# Residual-TD3 RL for TwoArmDrawerCleanup.
# 启动方式(自动套 GPU + EGL + conda + tmux):
#   bash resfit/rl_finetuning/shell/run_rl.sh drawercleanup <gpu>
# 前置:dexmg-drawercleanup-bc 已产出 _best,且 residual_td3.py 里
#       ResidualTD3DrawerCleanupConfig.base_policy.wandb_id 已填真实 run_id。
# 双臂硬任务超参与 coffee/boxcleanup 一致(n_step=5 / buffer=300k / action_scale=0.2)。

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_drawer_cleanup_config \
    algo.prefetch_batches=4 \
    algo.n_step=5 \
    algo.gamma=0.995 \
    algo.learning_starts=10_000 \
    algo.critic_warmup_steps=10_000 \
    algo.num_updates_per_iteration=4 \
    algo.stddev_max=0.025 \
    algo.stddev_min=0.025 \
    algo.buffer_size=300_000 \
    agent.actor.action_scale=0.2 \
    agent.actor_lr=1e-6 \
    wandb.project=dexmg-drawercleanup-final \
    wandb.name=residual-rl \
    wandb.notes=paper_runs/drawercleanup/1_drawercleanup_residual_rl \
    wandb.group=residual-rl \
    debug=false
