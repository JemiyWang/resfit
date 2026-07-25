#!/usr/bin/env bash
set -euo pipefail

PORTS=("${1:?port0}" "${2:?port1}" "${3:?port2}")
ROOT=/mnt/mnt/data/domains_rise/cup
OUT=/mnt/mnt/data/resfit/outputs_chunk/cup_pi0feat_shards
PY=/mnt/mnt/data/envs/residual/bin/python
pids=()

cd /mnt/mnt/data/resfit
source resfit/lerobot/shell/torchcodec_env.sh 2>/dev/null || true
export PYTHONPATH="/mnt/mnt/data/resfit${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUT}"

# --shard_index 0 writes success_shard0.npz, then fail_shard0.npz.
# --shard_index 1 writes success_shard1.npz, then fail_shard1.npz.
# --shard_index 2 writes success_shard2.npz, then fail_shard2.npz.
build_shard() {
  local port=$1
  local shard=$2
  "${PY}" -m resfit.rl_finetuning.chunk_residual.build_pi0_feat_cache_via_serve \
    --host 127.0.0.1 --port "${port}" --data_source teleavatar \
    --lerobot_root "${ROOT}" --repo_id cup_success --prompt "pick cup" \
    --pooling mean --serve_ckpt_id pi05_pick_cup_awbc_49999 \
    --out_cache "${OUT}/success_shard${shard}.npz" \
    --num_shards 3 --shard_index "${shard}"
  "${PY}" -m resfit.rl_finetuning.chunk_residual.build_pi0_feat_cache_via_serve \
    --host 127.0.0.1 --port "${port}" --data_source teleavatar \
    --lerobot_root "${ROOT}" --repo_id cup_fail --prompt "pick cup" \
    --pooling mean --serve_ckpt_id pi05_pick_cup_awbc_49999 \
    --out_cache "${OUT}/fail_shard${shard}.npz" \
    --num_shards 3 --shard_index "${shard}"
}

for shard in 0 1 2; do
  build_shard "${PORTS[$shard]}" "${shard}" \
    >"${OUT}/shard${shard}.log" 2>&1 &
  pids[$shard]=$!
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done
