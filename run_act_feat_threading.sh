#!/usr/bin/env bash
# threading act_feat 端到端:gc_value→high_actor→500k main(照搬 threading_aligned_bp_bc01 命令,仅 v=act_feat)。
# 用法: bash run_act_feat_threading.sh <gpu>。代码已 smoke 验过整条。
set -u
GPU=${1:?usage: run_act_feat_threading.sh <gpu>}
export PATH="/root/miniconda3/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=/mnt/mnt/data/resfit
cd /mnt/mnt/data/resfit || exit 3
RUN="conda run -n residual --no-capture-output python -u"
HDF5=resfit/dataset/two_arm_threading.hdf5
DS=ankile/dexmg-two-arm-threading
STAGES=outputs_chunk/two_arm_threading_stages.npz
CACHE=outputs_chunk/two_arm_threading_act_feat.npz
BASE=resfit/out/threading/best
GCV=outputs_chunk/two_arm_threading_gc_value_actfeat.pt
HIA=outputs_chunk/two_arm_threading_high_actor_actfeat.pt
gate(){ [ -s "$1" ] || { echo "[thr] FATAL 产物缺失: $1 (停)"; exit 4; }; }

echo "[thr] START $(date '+%F %T')  GPU=$CUDA_VISIBLE_DEVICES"

echo "[thr] 1) act_feat gc_value(建缓存)... $(date '+%T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_hiql_gc_value \
  --state_mode act_feat --act_base_ckpt "$BASE" \
  --hdf5 "$HDF5" --dataset "$DS" --stage_cache "$STAGES" \
  --act_feat_cache "$CACHE" --output "$GCV"
gate "$GCV"; echo "[thr] 1) gc_value DONE $(date '+%T')"

echo "[thr] 2) act_feat high_actor(命中缓存)... $(date '+%T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
  --state_mode act_feat --act_base_ckpt "$BASE" \
  --hdf5 "$HDF5" --dataset "$DS" --stage_cache "$STAGES" \
  --act_feat_cache "$CACHE" --gc_value_ckpt "$GCV" --output "$HIA"
gate "$HIA"; echo "[thr] 2) high_actor DONE $(date '+%T')"

echo "[thr] 3) 500k 主训(EGL)... $(date '+%T')"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreading --base_wandb_id "$BASE" --dataset "$DS" \
  --offline_dataset_path "$HDF5" --offline_stage_cache "$STAGES" \
  --subgoal_conditioned --gc_value_ckpt "$GCV" --high_actor_ckpt "$HIA" \
  --act_feat_cache "$CACHE" \
  --reward_shaping none --demo_bc_coef 0.1 --action_scale 0.05 \
  --offline_fraction 0.5 --base_n_action_steps 10 \
  --offline_buffer_cache outputs_chunk/threading_actfeat_offcache \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual --wandb_name threading_actfeat_bp_bc01 \
  --output_dir outputs_chunk/threading_actfeat_bp_bc01
echo "[thr] EXIT rc=$? $(date '+%F %T')"
