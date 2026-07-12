#!/usr/bin/env bash
# three_piece 残差重跑:value 换 HIQL-aligned(concat rep / hidden=512 / layers=3 / LN /
#   hiql loss+mask / geometric future,对齐 boxcleanup 的 value 配方),subgoal_way_steps 25→15。
# 在原 three_piece_actfeat_bp_bc01(hdf5 + stage_balanced + actfeat_bp_offcache)配置基础上改;
# 所有产物加新后缀 _hiqlv512_sg15,绝不覆盖原 three_piece_* 产物与 best.pt。
# 三步串行:gc_value(复用 act_feat 缓存) -> high_actor(命中缓存) -> 主训 500k。
# 用法: setsid bash run_three_piece_hiqlv512_sg15.sh <gpu> > three_piece_hiqlv512_sg15.log 2>&1 < /dev/null &
set -u
GPU=${1:-4}
export PATH="/root/miniconda3/bin:$PATH"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1
export PYTHONPATH=/mnt/mnt/data/resfit
cd /mnt/mnt/data/resfit || exit 3
RUN="conda run -n residual --no-capture-output python -u"

HDF5=resfit/dataset/two_arm_three_piece_assembly.hdf5
DS=ankile/dexmg-two-arm-three-piece-assembly
STAGES=outputs_chunk/three_piece_stages.npz
CACHE=outputs_chunk/three_piece_act_feat.npz                              # 复用现有特征缓存
BASE=resfit/out/piecce/best
GCV=outputs_chunk/three_piece_gc_value_actfeat_hiqlv512.pt               # 新
HIA=outputs_chunk/three_piece_high_actor_actfeat_hiqlv512.pt            # 新
OFFCACHE=outputs_chunk/three_piece_actfeat_bp_hiqlv512_sg15_offcache    # 新(会重建)
OUT=outputs_chunk/three_piece_actfeat_bp_bc01_hiqlv512_sg15             # 新
gate(){ [ -s "$1" ] || { echo "[driver] FATAL 产物缺失: $1 (上一步失败,停)"; exit 4; }; }

echo "[driver] START three_piece hiqlv512 sg15  GPU=$CUDA_VISIBLE_DEVICES  $(date '+%F %T')"
echo "[driver] GCV=$GCV"
echo "[driver] OUT=$OUT"

if [ -s "$GCV" ]; then
  echo "[driver] 1) gc_value 已存在,跳过重训(与 way_steps 无关,保留): $GCV  $(date '+%T')"
else
echo "[driver] 1) gc_value HIQL-aligned(concat/512/3/LN/hiql,复用 act_feat 缓存)... $(date '+%T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_hiql_gc_value \
  --state_mode act_feat --act_base_ckpt "$BASE" \
  --hdf5 "$HDF5" --dataset "$DS" --stage_cache "$STAGES" \
  --act_feat_cache "$CACHE" --output "$GCV" \
  --rep_dim 10 \
  --value_rep_mode concat \
  --value_hidden 512 \
  --value_layers 3 \
  --use_layer_norm 1 \
  --value_loss_mode hiql \
  --value_mask_mode hiql \
  --goal_future_mode geometric
fi
gate "$CACHE"; gate "$GCV"; echo "[driver] 1) gc_value READY $(date '+%T')"

echo "[driver] 2) high_actor(命中缓存)... $(date '+%T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
  --state_mode act_feat --act_base_ckpt "$BASE" \
  --hdf5 "$HDF5" --dataset "$DS" --stage_cache "$STAGES" \
  --act_feat_cache "$CACHE" --gc_value_ckpt "$GCV" --output "$HIA" \
  --way_steps 15
gate "$HIA"; echo "[driver] 2) high_actor DONE $(date '+%T')"

echo "[driver] 3) 主训 500k(base_policy,hdf5+stage_balanced,subgoal_way_steps 15)... $(date '+%T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreePieceAssembly --base_wandb_id "$BASE" \
  --dataset "$DS" \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw --stage_balanced --reward_shaping none \
  --offline_dataset_path "$HDF5" \
  --offline_fraction 0.5 --offline_stage_cache "$STAGES" \
  --offline_base_mode base_policy --demo_bc_coef 0.1 \
  --offline_buffer_cache "$OFFCACHE" \
  --subgoal_conditioned --gc_value_ckpt "$GCV" --high_actor_ckpt "$HIA" --act_feat_cache "$CACHE" \
  --subgoal_way_steps 15 \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual --wandb_name three_piece_actfeat_bp_bc01_hiqlv512_sg15 \
  --output_dir "$OUT"
echo "[driver] ALL DONE three_piece hiqlv512 sg15 $(date '+%F %T')  out=$OUT"
