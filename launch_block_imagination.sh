#!/usr/bin/env bash
# 用法: bash launch_block_imagination.sh <训练GPU> <SEED>
# 例:   bash launch_block_imagination.sh 5 0        # 单 seed / 单训练卡
#
# 前置(3 个 serve 必须都在):
#   GPU0  D-serve(WM,9000)
#   GPU1  kai0 base serve(8001, serve_block_awbc, pooling=mean)
#   GPU6  优势估计器 serve(8002, adv_serve, value_block)
# 训练用另一张空卡(2/3/4/5/7 任一)。B=1,单 seed 只占 1 张训练卡(共 4 卡)。
#
# 每 50k 存持久 checkpoint(imagination_step_*.pt)+ rollout 10 集写 imagined_adv_eval.jsonl
# (优势估计器逐帧进度值 ∈ [-1,1],≈1=成功;imagined proxy,非实机成功率)。
set -euo pipefail
GPU=${1:?need 训练 GPU id}
SEED=${2:?need seed}
cd /mnt/mnt/data/resfit
source resfit/lerobot/shell/torchcodec_env.sh 2>/dev/null || true
BLK=/mnt/mnt/data/domains_rise/block
OUT=/mnt/mnt/data/resfit/outputs_imagination/block_shore_mixed50_seed${SEED}
CUDA_VISIBLE_DEVICES=${GPU} PYTHONPATH=/mnt/mnt/data/resfit MUJOCO_GL=egl \
HF_LEROBOT_HOME=${BLK} HF_HUB_OFFLINE=1 \
/mnt/mnt/data/envs/residual/bin/python -m resfit.rl_finetuning.wm_bridge.launch_imagination \
  --value_ckpt /mnt/mnt/data/resfit/outputs_chunk/block_value_pi0feat.pt \
  --wm_host 127.0.0.1 --wm_port 9000 \
  --init_state_dataset ${BLK}/block_success --init_state_dataset ${BLK}/block_fail \
  --pi0_serve_ckpt_id pi05_block_awbc_49999 --num_denois_steps 10 \
  --offline_chunk_dataset ${BLK}/block_success \
  --offline_chunk_cache_root /mnt/mnt/data/resfit/cache/block_mixed_replay \
  --offline_fraction 0.5 --batch_size 256 \
  --base_policy_type pi05 --base_action_mode replan --chunk_length 50 \
  --actor raw --action_scale 0.2 --min_range_per_dim 0.1 \
  --demo_bc_coef 0.1 --bc_coef_final 0.01 \
  --critic_warmup_steps 10000 --learning_starts 10000 \
  --no_stage_balanced \
  --n_step 1 --gamma 0.995 --imagination_gamma 0.995 \
  --reward_shaping none --potential_source stage \
  --pi0_host 127.0.0.1 --pi0_port 8001 --pi0_prompt "build block" --pi0_action_dim 16 \
  --dataset block_success \
  --adv_host 127.0.0.1 --adv_port 8002 --n_eval_episodes 10 \
  --total_env_steps 500000 --eval_every_env_steps 50000 --eval_num_envs 1 \
  --seed ${SEED} --wandb_mode online --wandb_name block_shore_mixed50_seed${SEED} \
  --output_dir ${OUT} 2>&1 | tee ${OUT}.log
