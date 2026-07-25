#!/usr/bin/env bash
set -euo pipefail

PORTS=(
  "${1:?port0}"
  "${2:?port1}"
  "${3:?port2}"
  "${4:?port3}"
)
ROOT=/mnt/mnt/data/domains_rise/paper
OUT=/mnt/mnt/data/resfit/outputs_chunk/paper_pi0feat_shards
LOGS=/mnt/mnt/data/resfit/outputs_chunk/paper_pi0feat_logs
PY=/mnt/mnt/data/envs/residual/bin/python
pids=()

cd /mnt/mnt/data/resfit
source resfit/lerobot/shell/torchcodec_env.sh 2>/dev/null || true
export PYTHONPATH="/mnt/mnt/data/resfit${PYTHONPATH:+:${PYTHONPATH}}"
mkdir -p "${OUT}" "${LOGS}"

build_shard() {
  local port=$1
  local shard=$2
  "${PY}" -m resfit.rl_finetuning.chunk_residual.build_pi0_feat_cache_via_serve \
    --host 127.0.0.1 --port "${port}" --data_source teleavatar \
    --lerobot_root "${ROOT}" --repo_id paper_success \
    --prompt "put the paper roll on the holder" \
    --pooling mean --serve_ckpt_id pi05_paper_awbc_19999 \
    --policy_state_dim 14 \
    --out_cache "${OUT}/success_shard${shard}.npz" \
    --num_shards 4 --shard_index "${shard}"
  "${PY}" -m resfit.rl_finetuning.chunk_residual.build_pi0_feat_cache_via_serve \
    --host 127.0.0.1 --port "${port}" --data_source teleavatar \
    --lerobot_root "${ROOT}" --repo_id paper_fail \
    --prompt "put the paper roll on the holder" \
    --pooling mean --serve_ckpt_id pi05_paper_awbc_19999 \
    --policy_state_dim 14 \
    --out_cache "${OUT}/fail_shard${shard}.npz" \
    --num_shards 4 --shard_index "${shard}"
}

for shard in 0 1 2 3; do
  build_shard "${PORTS[$shard]}" "${shard}" \
    >"${LOGS}/shard${shard}.log" 2>&1 &
  pids[$shard]=$!
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done
