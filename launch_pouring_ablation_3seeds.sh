#!/usr/bin/env bash
# Launch pouring ablation seeds 1/2/3 safely, auto-picking free GPUs.
# Modes:
#   nosubgoal          : full staged baseline minus subgoal(+joint).       [offcache: NEW, must build]
#   nostage            : staged reward shaping off (subgoal+joint kept).   [offcache: reuse existing complete]
#   nostage_nosubgoal  : staged off AND subgoal off (flat demo-anchored).  [offcache: reuse existing complete]
# GPU selection is automatic: the least-loaded card by nvidia-smi memory.used, skipping
# cards busy with other jobs (used >= FREE_MB, default 2000) and cards already reserved by
# a sibling seed or another concurrent launch. Reservations live under logs/.gpu_resv/,
# keyed by the training PID and auto-cleaned when that PID exits. Override with FREE_MB=<MiB>.
# If the mode-specific offcache does not exist, seed1 starts first; seed2/3 start after
# buffer_meta.json appears (so they reuse the cache instead of racing to rebuild it).
# Usage: bash launch_pouring_ablation_3seeds.sh <nosubgoal|nostage|nostage_nosubgoal>
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "Usage: bash launch_pouring_ablation_3seeds.sh <nosubgoal|nostage|nostage_nosubgoal>" >&2
  exit 2
fi
MODE=$1
SEEDS=(1 2 3)
FREE_MB=${FREE_MB:-2000}   # a GPU counts as "free" when memory.used < FREE_MB

case "$MODE" in
  nosubgoal)         OFFCACHE=outputs_chunk/pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_staged_nosubgoal_offcache ;;
  nostage)           OFFCACHE=outputs_chunk/pouring_actfeat_hdf5_bp_hiqlv512_sg15_as005_subgoal_joint_nobc_nopot_offcache ;;
  nostage_nosubgoal) OFFCACHE=outputs_chunk/pouring_hdf5_bp_bc01_as005_nosubgoal_nojoint_nopot_offcache ;;
  *)
    echo "Usage: bash launch_pouring_ablation_3seeds.sh <nosubgoal|nostage|nostage_nosubgoal>" >&2
    exit 2
    ;;
esac

cd /mnt/mnt/data/resfit || exit 3
mkdir -p logs
RESV_DIR=logs/.gpu_resv
mkdir -p "$RESV_DIR"
ASSIGNED=()   # GPUs picked by this launcher invocation

# Drop reservation files whose owning PID has exited.
clean_stale_resv(){
  local f rpid
  shopt -s nullglob
  for f in "$RESV_DIR"/gpu_*; do
    rpid=$(cat "$f" 2>/dev/null || true)
    if [ -z "$rpid" ] || ! kill -0 "$rpid" 2>/dev/null; then rm -f "$f"; fi
  done
  shopt -u nullglob
}

# Echo the index of the least-loaded GPU not already taken (in-batch ASSIGNED or a live
# reservation from a concurrent launch). Warns if even the best candidate is >= FREE_MB.
pick_gpu(){
  clean_stale_resv
  local reserved=" " f idx gi gm best="" best_mem=""
  shopt -s nullglob
  for f in "$RESV_DIR"/gpu_*; do reserved+="${f##*/gpu_} "; done
  shopt -u nullglob
  for idx in "${ASSIGNED[@]:-}"; do [ -n "$idx" ] && reserved+="$idx "; done
  while IFS=',' read -r gi gm; do
    gi=$(echo "$gi" | tr -d ' '); gm=$(echo "$gm" | tr -d ' ')
    [ -z "$gi" ] && continue
    [[ " $reserved " == *" $gi "* ]] && continue
    if [ -z "$best_mem" ] || [ "$gm" -lt "$best_mem" ]; then best="$gi"; best_mem="$gm"; fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  if [ -z "$best" ]; then
    echo "[launcher] no GPU available (all reserved/assigned)" >&2
    return 1
  fi
  if [ "$best_mem" -ge "$FREE_MB" ]; then
    echo "[launcher] WARN: freest free-GPU is $best with ${best_mem}MiB used (>= ${FREE_MB}MiB); using it anyway" >&2
  fi
  echo "$best"
}

# Pick a GPU, launch the seed on it detached, and reserve that GPU under the training PID.
launch_one(){
  local idx=$1
  local SEED=${SEEDS[$idx]}
  local GPU
  GPU=$(pick_gpu) || exit 6
  local LOG=logs/pouring_${MODE}_seed${SEED}.log
  echo "[launcher] start mode=$MODE seed=$SEED gpu=$GPU log=$LOG" >&2
  setsid bash run_pouring_ablation_seed.sh "$MODE" "$GPU" "$SEED" > "$LOG" 2>&1 < /dev/null &
  local cpid=$!
  echo "$cpid" > "$RESV_DIR/gpu_${GPU}"   # reserve for concurrent launches (survives subshell)
  ASSIGNED+=("$GPU")
  echo "$cpid"
}

if [ -s "$OFFCACHE/buffer_meta.json" ]; then
  echo "[launcher] offcache exists: $OFFCACHE; launching seeds 1/2/3 now."
  for i in 0 1 2; do launch_one "$i" >/dev/null; done
  echo "[launcher] submitted all seeds."
  exit 0
fi

echo "[launcher] offcache missing: $OFFCACHE"
echo "[launcher] launching seed1 first; seed2/3 will start after buffer_meta.json appears."
PID1=$(launch_one 0)
while true; do
  if [ -s "$OFFCACHE/buffer_meta.json" ]; then
    echo "[launcher] offcache is ready; launching seed2/seed3."
    launch_one 1 >/dev/null
    launch_one 2 >/dev/null
    echo "[launcher] submitted all seeds. seed1 pid=$PID1"
    exit 0
  fi
  if ! kill -0 "$PID1" 2>/dev/null; then
    echo "[launcher] seed1 exited before offcache became ready; inspect logs/pouring_${MODE}_seed1.log" >&2
    exit 5
  fi
  sleep 60
done
