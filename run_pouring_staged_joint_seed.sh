#!/usr/bin/env bash
# pouring staged_joint seed 变体:逐字对齐 run_pouring_staged_joint.sh
#   (即 pouring_staged_joint.log 那版:base seed=0,2026-07-08 14:35 因主机 OOM 被杀,
#    200k/500k best=0.98),唯一变量 = --seed。
#   offcache 只读复用 pouring_actfeat_hdf5_bp_hiqlv512_sg15_as005_staged_joint_offcache
#   (seed 不进 _offline_buffer_signature;多个 seed 并发只读同一份,省首建时间)。
# 用法: setsid bash run_pouring_staged_joint_seed.sh <gpu> <seed> > pouring_staged_joint_seed<seed>.log 2>&1 < /dev/null &
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

OFFCACHE=outputs_chunk/pouring_actfeat_hdf5_bp_hiqlv512_sg15_as005_staged_joint_offcache   # 只读复用(签名不含 seed)
STAGES=outputs_chunk/two_arm_pouring_stages.npz
BASE="/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2"
NAME=pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_staged_joint_seed${SEED}
gate(){ [ -s "$1" ] || { echo "[pouring_staged_seed$SEED] FATAL 复用产物缺失: $1"; exit 4; }; }
gate "$OFFCACHE/buffer_meta.json"
gate "$STAGES"
gate "outputs_chunk/pouring_gc_value_actfeat_hdf5_hiqlv512.pt"
gate "outputs_chunk/pouring_high_actor_actfeat_hdf5_hiqlv512.pt"
gate "outputs_chunk/pouring_act_feat_hdf5.npz"
gate "$BASE/policy/model.safetensors"
echo "[pouring_staged_seed$SEED] START pouring staged+joint  GPU=$CUDA_VISIBLE_DEVICES seed=$SEED  $(date '+%F %T')  (offcache 复用)"

$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmPouring --base_wandb_id "$BASE" \
  --dataset ankile/dexmg-two-arm-pouring \
  --offline_dataset_path resfit/dataset/two_arm_pouring.hdf5 \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw --stage_balanced \
  --reward_shaping staged --stage_reward_bonus 1.0 \
  --offline_stage_cache "$STAGES" \
  --offline_fraction 0.5 --offline_base_mode base_policy --demo_bc_coef 0.1 \
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
echo "[pouring_staged_seed$SEED] ALL DONE $(date '+%F %T')  out=outputs_chunk/$NAME"
