#!/usr/bin/env bash
# threading sg15:参数对齐正在跑的 three_piece_actfeat_bp_bc01_hiqlv512_sg15。
#   subgoal_way_steps=15,demo-BC 固定 0.1(不衰减)。
# 复用共享 hiqlv512 gc_value(run_threading_gcvalue_hiqlv512.sh 先训好)+ act_feat + stage_cache。
# 重做:high_actor(way_steps 15)+ offcache(主训启动时自动建)。产物用 _hiqlv512_sg15 后缀。
# 用法: setsid bash run_threading_hiqlv512_sg15.sh <gpu> > threading_hiqlv512_sg15.log 2>&1 < /dev/null &
set -u
GPU=${1:-3}
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
STAGES=outputs_chunk/two_arm_threading_stages.npz                          # 复用
CACHE=outputs_chunk/two_arm_threading_act_feat.npz                         # 复用
BASE=resfit/out/threading/best
GCV=outputs_chunk/two_arm_threading_gc_value_actfeat_hiqlv512.pt           # 复用(共享,先训好)
HIA=outputs_chunk/two_arm_threading_high_actor_actfeat_hiqlv512_sg15.pt    # 新(way_steps 15)
OFFCACHE=outputs_chunk/threading_actfeat_bp_hiqlv512_sg15_offcache         # 新(会重建)
OUT=outputs_chunk/threading_actfeat_bp_bc01_hiqlv512_sg15                  # 新
WAY=15
gate(){ [ -s "$1" ] || { echo "[sg15] FATAL 复用产物缺失: $1"; exit 4; }; }

echo "[sg15] gate 复用产物(gc_value/act_feat/stage)..."
gate "$CACHE"; gate "$STAGES"; gate "$GCV"
echo "[sg15] START threading hiqlv512 sg${WAY}  GPU=$CUDA_VISIBLE_DEVICES  $(date '+%F %T')"

echo "[sg15] high_actor 重训(way_steps ${WAY})... $(date '+%T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
  --state_mode act_feat --act_base_ckpt "$BASE" \
  --hdf5 "$HDF5" --dataset "$DS" --stage_cache "$STAGES" \
  --act_feat_cache "$CACHE" --gc_value_ckpt "$GCV" --output "$HIA" \
  --way_steps "$WAY"
gate "$HIA"; echo "[sg15] high_actor DONE $(date '+%T')"

echo "[sg15] 主训 500k(对齐 three_piece sg15:stage_balanced,subgoal_way_steps ${WAY},bc 固定 0.1)... $(date '+%T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreading --base_wandb_id "$BASE" --dataset "$DS" \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw --stage_balanced --reward_shaping none \
  --offline_dataset_path "$HDF5" \
  --offline_fraction 0.5 --offline_stage_cache "$STAGES" \
  --offline_base_mode base_policy --demo_bc_coef 0.1 \
  --offline_buffer_cache "$OFFCACHE" \
  --subgoal_conditioned --gc_value_ckpt "$GCV" --high_actor_ckpt "$HIA" --act_feat_cache "$CACHE" \
  --subgoal_way_steps "$WAY" \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual --wandb_name threading_actfeat_bp_bc01_hiqlv512_sg15 \
  --output_dir "$OUT"
echo "[sg15] ALL DONE threading hiqlv512 sg${WAY} $(date '+%F %T')  out=$OUT"
