#!/usr/bin/env bash
# 用法: bash launch_cup_imagination.sh <训练GPU> <SEED>
# 例:   bash launch_cup_imagination.sh 5 0        # 单 seed / 单训练卡
#
# 前置(3 个 serve 必须都在):
#   GPU0  D-serve(WM,9000)
#   GPU1  kai0 base serve(8001, pi05_pick_cup_awbc_49999, pooling=mean)
#   GPU6  优势估计器 serve(8002, adv_serve, RISE value_cup)
# 训练用另一张空卡(2/3/4/5/7 任一)。B=1,单 seed 只占 1 张训练卡(共 4 卡)。
#
# 每 50k 存持久 checkpoint(imagination_step_*.pt)+ rollout 10 集写 imagined_adv_eval.jsonl
# (优势估计器逐帧进度值 ∈ [-1,1],≈1=成功;imagined proxy,非实机成功率)。
set -euo pipefail
GPU=${1:?need 训练 GPU id}
SEED=${2:?need seed}
cd /mnt/mnt/data/resfit
source resfit/lerobot/shell/torchcodec_env.sh 2>/dev/null || true
CUP=/mnt/mnt/data/domains_rise/cup
OUT=/mnt/mnt/data/resfit/outputs_imagination/cup_shore_mixed50_seed${SEED}
mkdir -p "$(dirname "${OUT}")"
CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=/mnt/mnt/data/resfit MUJOCO_GL=egl \
HF_LEROBOT_HOME="${CUP}" HF_HUB_OFFLINE=1 \
/mnt/mnt/data/envs/residual/bin/python -m resfit.rl_finetuning.wm_bridge.launch_imagination \
  --value_ckpt /mnt/mnt/data/resfit/outputs_chunk/cup_value_pi0feat.pt \
  --task_profile cup \
  --wm_host 127.0.0.1 --wm_port 9000 \
  --init_state_dataset "${CUP}/cup_success" --init_state_dataset "${CUP}/cup_fail" \
  --pi0_serve_ckpt_id pi05_pick_cup_awbc_49999 --num_denois_steps 10 \
  --offline_chunk_dataset "${CUP}/cup_success" \
  --offline_chunk_cache_root /mnt/mnt/data/resfit/cache/cup_mixed_replay \
  --offline_fraction 0.5 --batch_size 256 \
  --base_policy_type pi05 --base_action_mode replan --chunk_length 50 \
  --actor raw --action_scale 0.2 --min_range_per_dim 0.1 \
  --demo_bc_coef 0.1 \
  --utd 4 --actor_lr 0.000001 --critic_lr 0.0001 \
  --critic_warmup_steps 10000 --learning_starts 10000 \
  --no_stage_balanced \
  --n_step 1 --gamma 0.995 --imagination_gamma 0.995 \
  --reward_shaping none --potential_source stage \
  --pi0_host 127.0.0.1 --pi0_port 8001 --pi0_prompt "pick cup" --pi0_action_dim 16 \
  --dataset cup_success \
  --adv_host 127.0.0.1 --adv_port 8002 --adv_prompt "pick cup" --n_eval_episodes 10 \
  --total_env_steps 500000 --eval_every_env_steps 50000 --eval_num_envs 1 \
  --seed "${SEED}" --wandb_mode online --wandb_name "cup_shore_mixed50_seed${SEED}" \
  --output_dir "${OUT}" 2>&1 | tee "${OUT}.log"
