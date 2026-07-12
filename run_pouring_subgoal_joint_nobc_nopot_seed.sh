#!/usr/bin/env bash
# pouring subgoal_joint_nobc_nopot seed 变体:逐字对齐 run_pouring_subgoal_joint_nobc_nopot.sh
#   (即 pouring_subgoal_joint_nobc_nopot.log 那版:base seed=0,7-7 停在 270k/500k best=0.92),
#   唯一变量 = --seed。
#   offcache 只读复用 pouring_actfeat_hdf5_bp_hiqlv512_sg15_as005_subgoal_joint_nobc_nopot_offcache
#   (seed 不进 _offline_buffer_signature;已存在,多个 seed 并发只读同一份,省首建 ~80min)。
# 用法: setsid bash run_pouring_subgoal_joint_nobc_nopot_seed.sh <gpu> <seed> > pouring_subgoal_joint_nobc_nopot_seed<seed>.log 2>&1 < /dev/null &
set -u
GPU=${1:?需要 gpu}; SEED=${2:?需要 seed}
export PATH="/root/miniconda3/bin:$PATH"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1
export PYTHONPATH=/mnt/mnt/data/resfit
cd /mnt/mnt/data/resfit || exit 3
RUN="conda run -n residual --no-capture-output python -u"

OFFCACHE=outputs_chunk/pouring_actfeat_hdf5_bp_hiqlv512_sg15_as005_subgoal_joint_nobc_nopot_offcache   # 只读复用(签名不含 seed)
BASE="/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2"
NAME=pouring_actfeat_hdf5_bp_hiqlv512_sg15_as005_subgoal_joint_nobc_nopot_seed${SEED}
gate(){ [ -s "$1" ] || { echo "[subgoal_nobc_nopot_seed$SEED] FATAL 复用产物缺失: $1"; exit 4; }; }
gate "$OFFCACHE/buffer_meta.json"
gate "outputs_chunk/pouring_gc_value_actfeat_hdf5_hiqlv512.pt"
gate "outputs_chunk/pouring_high_actor_actfeat_hdf5_hiqlv512.pt"
gate "outputs_chunk/pouring_act_feat_hdf5.npz"
gate "$BASE/policy/model.safetensors"
echo "[subgoal_nobc_nopot_seed$SEED] START pouring subgoal+joint (no bc, no potential)  GPU=$CUDA_VISIBLE_DEVICES seed=$SEED  $(date '+%F %T')  (offcache 复用)"

$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmPouring --base_wandb_id "$BASE" \
  --dataset ankile/dexmg-two-arm-pouring \
  --offline_dataset_path resfit/dataset/two_arm_pouring.hdf5 \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw \
  --reward_shaping none \
  --offline_fraction 0.5 --offline_base_mode base_policy \
  --offline_buffer_cache "$OFFCACHE" \
  --subgoal_conditioned \
  --gc_value_ckpt outputs_chunk/pouring_gc_value_actfeat_hdf5_hiqlv512.pt \
  --high_actor_ckpt outputs_chunk/pouring_high_actor_actfeat_hdf5_hiqlv512.pt \
  --act_feat_cache outputs_chunk/pouring_act_feat_hdf5.npz \
  --subgoal_way_steps 15 --online_finetune_value --online_finetune_high_actor \
  --seed "$SEED" \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual \
  --wandb_name "$NAME" \
  --output_dir "outputs_chunk/$NAME"
echo "[subgoal_nobc_nopot_seed$SEED] ALL DONE $(date '+%F %T')  out=outputs_chunk/$NAME"
