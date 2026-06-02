#!/bin/bash

python -m resfit.rl_finetuning.scripts.train_residual_td3 \
    --config-name=residual_td3_can_config \
    algo.prefetch_batches=4 \
    wandb.project=robomimic-can-final \
    wandb.name=residual-rl \
    wandb.notes=paper_runs/can/1_can_residual_rl \
    wandb.group=residual-rl \
    debug=false
