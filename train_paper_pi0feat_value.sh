#!/usr/bin/env bash
set -euo pipefail

cd /mnt/mnt/data/resfit
source resfit/lerobot/shell/torchcodec_env.sh 2>/dev/null || true
export PYTHONPATH="/mnt/mnt/data/resfit${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p /mnt/mnt/data/resfit/outputs_chunk/paper_pi0feat_logs

/mnt/mnt/data/envs/residual/bin/python -m \
  resfit.rl_finetuning.chunk_residual.train_hiql_value \
  --dataset paper_success \
  --output /mnt/mnt/data/resfit/outputs_chunk/paper_value_pi0feat.pt \
  --state_mode pi0_feat \
  --terminal_reward_mode success_signed \
  --success_dataset outputs_chunk/paper_pi0feat_shards/success_shard0.npz \
  --success_dataset outputs_chunk/paper_pi0feat_shards/success_shard1.npz \
  --success_dataset outputs_chunk/paper_pi0feat_shards/success_shard2.npz \
  --success_dataset outputs_chunk/paper_pi0feat_shards/success_shard3.npz \
  --failure_dataset outputs_chunk/paper_pi0feat_shards/fail_shard0.npz \
  --failure_dataset outputs_chunk/paper_pi0feat_shards/fail_shard1.npz \
  --failure_dataset outputs_chunk/paper_pi0feat_shards/fail_shard2.npz \
  --failure_dataset outputs_chunk/paper_pi0feat_shards/fail_shard3.npz \
  --pi0_serve_ckpt_id pi05_paper_awbc_19999 \
  --pi0_image_keys top_head,hand_left,hand_right \
  --pi0_proprio_key observation.state \
  --pi0_pooling mean \
  --pi0_prompt "put the paper roll on the holder" \
  --gamma 0.99 --expectile 0.7 --ema 0.005 \
  --lr 0.0003 --batch_size 256 --steps 50000 \
  --value_hidden 256 --seed 0 \
  2>&1 | tee /mnt/mnt/data/resfit/outputs_chunk/paper_pi0feat_logs/train_value.log
