#!/usr/bin/env bash
# 为 libero_10 task8("put both moka pots on the stove")建全套 pi0_feat 离线资产:
#   A) build pi0_feat cache(serve 逐帧 infer,~25min)→ B) train gc_value → C) train high_actor
# 复刻 task3 那套(误名 libero_task8_pi0_feat_*)的确切配置,只换任务语言/数据。
# 全程 residual env(纯 torch + openpi_client 连 serve,不 import openpi/libero);serve 须在 127.0.0.1:8000。
# GPU 用卡 5(serve 在卡 4)。命名用 t8moka 明确区分误名的 task3 资产。
set -euo pipefail
cd /mnt/mnt/data/resfit

export CUDA_VISIBLE_DEVICES=5
# 共享机~100/112核满载,默认抓满核 BLAS/cv2 线程会 thrash(实测 build 慢 ~55x:261ms→4.7ms/帧)。锁单线程。
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
PY=/mnt/mnt/data/envs/residual/bin/python
ROOT=/mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero
TASKLANG="put both moka pots on the stove"
CACHE=outputs_chunk/libero10_t8moka_pi0_feat.npz
GCV=outputs_chunk/libero10_t8moka_pi0_feat_gc_value.pt
HA=outputs_chunk/libero10_t8moka_pi0_feat_high_actor.pt
IMGK=observation.images.agentview,observation.images.robot0_eye_in_hand
DUMMY=/tmp/dummy_pi0feat.h5

# pi0_feat 短路不读 hdf5,但给个合法空 hdf5 兜底(防任何 os 检查/打开)
"$PY" -c "import h5py; h5py.File('$DUMMY','w').close()"

echo "===== A) build pi0_feat cache（serve 逐帧 infer，约 25 分钟） $(date +%T) ====="
PYTHONPATH=/mnt/mnt/data/resfit "$PY" -u -m resfit.rl_finetuning.chunk_residual.build_pi0_feat_cache_via_serve \
  --data_source libero --host 127.0.0.1 --port 8000 \
  --lerobot_root "$ROOT" --language "$TASKLANG" \
  --serve_ckpt_id pi0_libero --pooling last \
  --out_cache "$CACHE"

echo "===== B) train gc_value（pi0_feat, GPU5, 50k steps） $(date +%T) ====="
PYTHONPATH=/mnt/mnt/data/resfit "$PY" -u -m resfit.rl_finetuning.chunk_residual.train_hiql_gc_value \
  --state_mode pi0_feat --pi0_feat_cache "$CACHE" \
  --pi0_serve_ckpt_id pi0_libero --pi0_image_keys "$IMGK" \
  --pi0_proprio_key observation.state --pi0_pooling last --pi0_prompt "$TASKLANG" \
  --hdf5 "$DUMMY" --dataset physical-intelligence/libero \
  --device cuda --output "$GCV"

echo "===== C) train high_actor（pi0_feat, GPU5, 50k steps） $(date +%T) ====="
PYTHONPATH=/mnt/mnt/data/resfit "$PY" -u -m resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
  --state_mode pi0_feat --pi0_feat_cache "$CACHE" \
  --pi0_serve_ckpt_id pi0_libero --pi0_image_keys "$IMGK" \
  --pi0_proprio_key observation.state --pi0_pooling last --pi0_prompt "$TASKLANG" \
  --gc_value_ckpt "$GCV" \
  --hdf5 "$DUMMY" --dataset physical-intelligence/libero \
  --device cuda --output "$HA"

echo "===== ALL DONE $(date +%T) ====="
echo "cache=$CACHE"; echo "gc_value=$GCV"; echo "high_actor=$HA"
