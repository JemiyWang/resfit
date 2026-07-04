#!/usr/bin/env bash
# 训练单状态 act_feat value(pouring),喂给 pothiql 当势函数 Φ(s)=V(image_feat+proprio)。
# 复用现有 act_feat 缓存(命中即跳过 embed);配方镜像 eef pouring_value_hdf5.pt(hidden256/默认超参)。
set -eu
export PATH="/root/miniconda3/bin:$PATH"
export PYTHONPATH=/mnt/mnt/data/resfit
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1
cd /mnt/mnt/data/resfit
BASE="/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2"
conda run -n residual --no-capture-output python -u \
  -m resfit.rl_finetuning.chunk_residual.train_hiql_value \
  --hdf5 resfit/dataset/two_arm_pouring.hdf5 \
  --dataset ankile/dexmg-two-arm-pouring \
  --state_mode act_feat \
  --act_feat_cache outputs_chunk/pouring_act_feat_hdf5.npz \
  --act_base_ckpt "$BASE" \
  --value_hidden 256 \
  --output outputs_chunk/pouring_value_actfeat_hdf5.pt
