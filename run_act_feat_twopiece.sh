#!/usr/bin/env bash
# act_feat 离线两件套(等配置 three_piece_aligned 的 gc_value/high_actor,仅 state_mode=act_feat)。
# 首跑 gc_value 会把每条 demo 整段帧过冻结 ACT 建缓存(慢);high_actor 命中缓存秒读。
# 训练只读 hdf5 + 过 ACT + 训 MLP,不需 env/EGL。
set -u
export PATH="/root/miniconda3/bin:$PATH"
export CUDA_VISIBLE_DEVICES=0 HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
cd /mnt/mnt/data/resfit || exit 3
RUN="conda run -n residual --no-capture-output python -u"
HDF5=resfit/dataset/two_arm_three_piece_assembly.hdf5
DS=ankile/dexmg-two-arm-three-piece-assembly
STAGES=outputs_chunk/three_piece_stages.npz
CACHE=outputs_chunk/three_piece_act_feat.npz
BASE=resfit/out/piecce/best
GCV=outputs_chunk/three_piece_gc_value_actfeat.pt
HIA=outputs_chunk/three_piece_high_actor_actfeat.pt
gate(){ [ -s "$1" ] || { echo "[driver] FATAL 产物缺失: $1 (上一步失败,停)"; exit 4; }; }

echo "[driver] START $(date '+%F %T')  GPU=$CUDA_VISIBLE_DEVICES"

echo "[driver] 1) act_feat gc_value(首跑建缓存,慢)... $(date '+%T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_hiql_gc_value \
  --state_mode act_feat --act_base_ckpt "$BASE" \
  --hdf5 "$HDF5" --dataset "$DS" --stage_cache "$STAGES" \
  --act_feat_cache "$CACHE" --output "$GCV"
gate "$GCV"; echo "[driver] 1) gc_value DONE $(date '+%T')"

echo "[driver] 2) act_feat high_actor(缓存命中)... $(date '+%T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
  --state_mode act_feat --act_base_ckpt "$BASE" \
  --hdf5 "$HDF5" --dataset "$DS" --stage_cache "$STAGES" \
  --act_feat_cache "$CACHE" --gc_value_ckpt "$GCV" --output "$HIA"
gate "$HIA"; echo "[driver] 2) high_actor DONE $(date '+%T')"

echo "[driver] ALL DONE $(date '+%F %T')"
echo "[driver] gc=$GCV"
echo "[driver] high=$HIA  cache=$CACHE"
