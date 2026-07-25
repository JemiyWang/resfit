#!/usr/bin/env bash
set -euo pipefail

GPU=${1:?need GPU id}
PORT=${2:?need port}

cd /mnt/mnt/data/resfit
exec env \
  CUDA_VISIBLE_DEVICES="${GPU}" \
  PYTHONPATH=/mnt/mnt/data/data2/kai0/src:/mnt/mnt/data/data2/kai0 \
  /mnt/mnt/data/chj/openpi/.venv/bin/python \
  /mnt/mnt/data/resfit/pi0_serve/serve_block_awbc.py \
  --dir /mnt/mnt/data/data2/kai0/checkpoints/paper/19999 \
  --port "${PORT}" \
  --pooling mean \
  --default-prompt "put the paper roll on the holder" \
  --config-name pi05_paper_awbc \
  --repo-id /mnt/mnt/data/domains_rise/paper/paper_success \
  --asset-id pick_paper_all_merged \
  --serve-ckpt-id pi05_paper_awbc_19999 \
  --policy-state-dim 14
