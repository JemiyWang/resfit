#!/bin/bash
# threading staged_joint 多 seed 补跑。逐字对齐已跑完的 seed-0 run
# (wandb run-20260704_111336-ulguni4f, name threading_actfeat_bp_bc01_hiqlv512_sg15_staged_joint)。
# 仅两处改动:① 加 --seed <SEED>;② wandb_name / output_dir 加 _seed<SEED> 后缀(不覆盖 seed-0 的 best.pt)。
# 其余全部一字不差 → offcache(26G,seed 不进签名)/gc_value/high_actor/act_feat/stage_cache/base 全复用,秒级加载、零重建、零磁盘新增。
# 用法: setsid bash run_threading_staged_joint_seed.sh <gpu> <seed> > threading_staged_joint_seed<seed>.log 2>&1 < /dev/null &
set -euo pipefail
GPU=${1:?need gpu}
SEED=${2:?need seed}
export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1
export PYTHONPATH=/mnt/mnt/data/resfit
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
cd /mnt/mnt/data/resfit
PY=/mnt/mnt/data/envs/residual/bin/python
echo "[threading_staged_joint] START GPU=$GPU seed=$SEED  $(date '+%F %T')"
exec $PY -u -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreading \
  --base_wandb_id resfit/out/threading/best \
  --dataset ankile/dexmg-two-arm-threading \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw \
  --stage_balanced --reward_shaping staged --stage_reward_bonus 1.0 \
  --offline_dataset_path resfit/dataset/two_arm_threading.hdf5 \
  --offline_fraction 0.5 --offline_stage_cache outputs_chunk/two_arm_threading_stages.npz \
  --offline_base_mode base_policy --demo_bc_coef 0.1 \
  --offline_buffer_cache outputs_chunk/threading_actfeat_bp_hiqlv512_sg15_staged_offcache \
  --subgoal_conditioned \
  --gc_value_ckpt outputs_chunk/two_arm_threading_gc_value_actfeat_hiqlv512.pt \
  --high_actor_ckpt outputs_chunk/two_arm_threading_high_actor_actfeat_hiqlv512_sg15.pt \
  --act_feat_cache outputs_chunk/two_arm_threading_act_feat.npz \
  --subgoal_way_steps 15 \
  --online_finetune_value --online_finetune_high_actor \
  --seed "$SEED" \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual \
  --wandb_name "threading_actfeat_bp_bc01_hiqlv512_sg15_staged_joint_seed${SEED}" \
  --output_dir "outputs_chunk/threading_actfeat_bp_bc01_hiqlv512_sg15_staged_joint_seed${SEED}"
