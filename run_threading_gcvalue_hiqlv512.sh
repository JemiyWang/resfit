#!/usr/bin/env bash
# threading 共享 gc_value:用与 three_piece 完全相同的 hiqlv512 配方重训
#   (concat / hidden=512 / layers=3 / LN / hiql loss+mask / geometric future / rep_dim 10)。
# 与 way_steps / bc 无关,两个 threading 实验(sg15 / sg8)共用这一份。
# 产物用新后缀 _hiqlv512,绝不覆盖旧的 two_arm_threading_gc_value_actfeat.pt(默认配方)。
# 用法: setsid bash run_threading_gcvalue_hiqlv512.sh <gpu> > threading_gcvalue_hiqlv512.log 2>&1 < /dev/null &
set -u
GPU=${1:-3}
export PATH="/root/miniconda3/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1 HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export PYTHONPATH=/mnt/mnt/data/resfit
cd /mnt/mnt/data/resfit || exit 3
RUN="conda run -n residual --no-capture-output python -u"
HDF5=resfit/dataset/two_arm_threading.hdf5
DS=ankile/dexmg-two-arm-threading
STAGES=outputs_chunk/two_arm_threading_stages.npz
CACHE=outputs_chunk/two_arm_threading_act_feat.npz
BASE=resfit/out/threading/best
GCV=outputs_chunk/two_arm_threading_gc_value_actfeat_hiqlv512.pt
gate(){ [ -s "$1" ] || { echo "[gcv] FATAL 缺失: $1"; exit 4; }; }
gate "$CACHE"; gate "$STAGES"

echo "[gcv] START threading gc_value hiqlv512  GPU=$CUDA_VISIBLE_DEVICES  $(date '+%F %T')"
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
echo "[gcv] DONE rc=$? $(date '+%F %T') -> $GCV"
