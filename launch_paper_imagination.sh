#!/usr/bin/env bash
# Usage: bash launch_paper_imagination.sh <training GPU> <seed>
set -euo pipefail

GPU=${1:?need training GPU id}
SEED=${2:?need seed}
PAPER=/mnt/mnt/data/domains_rise/paper
OUT_ROOT=/mnt/mnt/data/resfit/outputs_imagination
LOG_ROOT=/mnt/mnt/data/resfit/logs
RUN_NAME=paper_shore_mixed50_seed${SEED}
WANDB_MODE=online
EXTRA_ARGS=()
if [[ "${SMOKE:-0}" == "1" ]]; then
  RUN_NAME=${RUN_NAME}_smoke
  WANDB_MODE=disabled
  EXTRA_ARGS+=(--smoke)
fi
OUT=${OUT_ROOT}/${RUN_NAME}

cd /mnt/mnt/data/resfit
source resfit/lerobot/shell/torchcodec_env.sh 2>/dev/null || true
mkdir -p "${OUT_ROOT}" "${LOG_ROOT}"

CUDA_VISIBLE_DEVICES="${GPU}" PYTHONPATH=/mnt/mnt/data/resfit MUJOCO_GL=egl \
HF_LEROBOT_HOME="${PAPER}" HF_HUB_OFFLINE=1 \
/mnt/mnt/data/envs/residual/bin/python -m resfit.rl_finetuning.wm_bridge.launch_imagination \
  --value_ckpt /mnt/mnt/data/resfit/outputs_chunk/paper_value_pi0feat.pt \
  --task_profile paper \
  --source_wandb_run eegyfzmz \
  --wm_host 127.0.0.1 --wm_port 9000 \
  --init_state_dataset "${PAPER}/paper_success" \
  --init_state_dataset "${PAPER}/paper_fail" \
  --pi0_serve_ckpt_id pi05_paper_awbc_19999 \
  --pi0_serve_ckpt_dir /mnt/mnt/data/data2/kai0/checkpoints/paper/19999 \
  --pi0_asset_id pick_paper_all_merged --pi0_pooling mean \
  --num_denois_steps 10 \
  --offline_chunk_dataset "${PAPER}/paper_success" \
  --offline_chunk_cache_root /mnt/mnt/data/resfit/cache/paper_mixed_replay \
  --offline_fraction 0.5 --batch_size 256 \
  --base_policy_type pi05 --base_action_mode replan --chunk_length 50 \
  --actor raw --action_scale 0.2 --min_range_per_dim 0.1 \
  --demo_bc_coef 0.1 \
  --utd 4 --actor_lr 0.000001 --critic_lr 0.0001 \
  --critic_warmup_steps 10000 --learning_starts 10000 \
  --no_stage_balanced \
  --n_step 1 --gamma 0.995 --imagination_gamma 0.995 \
  --reward_shaping none --potential_source stage \
  --task pick_paper_roll \
  --pi0_host 127.0.0.1 --pi0_port 8001 \
  --pi0_prompt "put the paper roll on the holder" \
  --pi0_action_dim 16 --pi0_execute_horizon 10 \
  --dataset paper_success \
  --adv_host 127.0.0.1 --adv_port 8002 \
  --adv_prompt "put the paper roll on the holder" \
  --adv_ckpt /mnt/mnt/data/resfit/RISE_Hi/policy_and_value/policy_offline_and_value/checkpoints/value_paper/value_paper/20000/model.safetensors \
  --adv_config value_paper --n_eval_episodes 10 \
  --total_env_steps 500000 --eval_every_env_steps 50000 \
  --eval_num_envs 1 --eval_num_episodes 50 \
  --seed "${SEED}" \
  --wandb_project dexmg-chunk-residual \
  --wandb_entity 674575221-beijing-institute-of-technology \
  --wandb_mode "${WANDB_MODE}" \
  --wandb_name "paper_shore_mixed50_seed${SEED}" \
  --output_dir "${OUT}" \
  "${EXTRA_ARGS[@]}" \
  2>&1 | tee "${LOG_ROOT}/${RUN_NAME}.log"
