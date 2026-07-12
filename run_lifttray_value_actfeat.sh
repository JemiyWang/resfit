#!/usr/bin/env bash
# 训练单状态 act_feat value(lifttray),喂给 pothiql 当势函数 Φ(s)=V(image_feat+proprio)。
# 镜像 run_pouring_value_actfeat.sh:复用现有 act_feat 缓存(命中即跳 embed);超参镜像 eef lifttray_value_hdf5.pt(hidden256/默认超参)。
# 用法: setsid bash run_lifttray_value_actfeat.sh <gpu> > lifttray_value_actfeat.log 2>&1 < /dev/null &
set -eu
GPU=${1:-6}
export PATH="/root/miniconda3/bin:$PATH"
export PYTHONPATH=/mnt/mnt/data/resfit
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1
export CUDA_VISIBLE_DEVICES="$GPU"
cd /mnt/mnt/data/resfit
BASE="/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_e0o0sckj_best:v4"
echo "[lifttray_value_actfeat] START GPU=$CUDA_VISIBLE_DEVICES $(date '+%F %T')"
conda run -n residual --no-capture-output python -u \
  -m resfit.rl_finetuning.chunk_residual.train_hiql_value \
  --hdf5 resfit/dataset/two_arm_lift_tray.hdf5 \
  --dataset ankile/dexmg-two-arm-lift-tray \
  --state_mode act_feat \
  --act_feat_cache outputs_chunk/lifttray_act_feat_hdf5.npz \
  --act_base_ckpt "$BASE" \
  --value_hidden 256 \
  --output outputs_chunk/lifttray_value_actfeat_hdf5.pt
echo "[lifttray_value_actfeat] DONE $(date '+%F %T')"
