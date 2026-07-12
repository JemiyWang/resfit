#!/usr/bin/env bash
# threading staged + 联合训练(online joint finetune)。
#   参照 threading_actfeat_bp_bc01_hiqlv512_sg15_pothiql,只改两处:
#     ① reward: potential(hiql) → staged(stage 整数净加 bonus;不需要 value.pt)
#     ② 打开联合训练: --online_finetune_value --online_finetune_high_actor
#   其余完全照搬 pothiql:actor raw / chunk1 / queue / base_n10 / action_scale0.05 /
#     actor_lr1e-6 / stage_balanced / offline_fraction0.5 / base_policy锚 + bc0.1 /
#     subgoal sg15(gc_value hiqlv512 + high_actor sg15)/ 500k。
#   前置产物(act_feat / stages / gc_value / high_actor)全复用,无需重训。
# 用法: setsid bash run_threading_staged_joint.sh <gpu> > threading_staged_joint.log 2>&1 < /dev/null &
set -u
GPU=${1:-0}
export PATH="/root/miniconda3/bin:$PATH"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1
export PYTHONPATH=/mnt/mnt/data/resfit
cd /mnt/mnt/data/resfit || exit 3
RUN="conda run -n residual --no-capture-output python -u"

HDF5=resfit/dataset/two_arm_threading.hdf5
DS=ankile/dexmg-two-arm-threading
STAGES=outputs_chunk/two_arm_threading_stages.npz                          # 复用(3段)
CACHE=outputs_chunk/two_arm_threading_act_feat.npz                         # 复用
BASE=resfit/out/threading/best
GCV=outputs_chunk/two_arm_threading_gc_value_actfeat_hiqlv512.pt           # 复用(joint 在线微调,初值取它)
HIA=outputs_chunk/two_arm_threading_high_actor_actfeat_hiqlv512_sg15.pt    # 复用(joint 在线微调,初值取它)
OFFCACHE=outputs_chunk/threading_actfeat_bp_hiqlv512_sg15_staged_offcache  # 新(staged 签名,重建)
OUT=outputs_chunk/threading_actfeat_bp_bc01_hiqlv512_sg15_staged_joint     # 新
WAY=15
gate(){ [ -s "$1" ] || { echo "[staged_joint] FATAL 复用产物缺失: $1"; exit 4; }; }

echo "[staged_joint] gate 复用产物(act_feat/stages/gc_value/high_actor/base)..."
gate "$CACHE"; gate "$STAGES"; gate "$GCV"; gate "$HIA"; gate "$BASE/model.safetensors"
echo "[staged_joint] START threading staged+joint  GPU=$CUDA_VISIBLE_DEVICES  $(date '+%F %T')"

$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreading --base_wandb_id "$BASE" --dataset "$DS" \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw --stage_balanced \
  --reward_shaping staged --stage_reward_bonus 1.0 \
  --offline_dataset_path "$HDF5" \
  --offline_fraction 0.5 --offline_stage_cache "$STAGES" \
  --offline_base_mode base_policy --demo_bc_coef 0.1 \
  --offline_buffer_cache "$OFFCACHE" \
  --subgoal_conditioned --gc_value_ckpt "$GCV" --high_actor_ckpt "$HIA" --act_feat_cache "$CACHE" \
  --subgoal_way_steps "$WAY" \
  --online_finetune_value --online_finetune_high_actor \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual --wandb_name threading_actfeat_bp_bc01_hiqlv512_sg15_staged_joint \
  --output_dir "$OUT"
echo "[staged_joint] ALL DONE threading staged+joint $(date '+%F %T')  out=$OUT"
