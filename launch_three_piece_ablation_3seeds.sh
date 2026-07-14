#!/usr/bin/env bash
# Launch three_piece ablation seeds 1/2/3, with GPU placement control.
# Modes:
#   nosubgoal          : full staged baseline minus subgoal(+joint).
#   nostage            : staged reward shaping off (subgoal+joint kept).
#   nostage_nosubgoal  : staged off AND subgoal off (flat demo-anchored residual).
#
# GPU placement (env vars):
#   GPUS=0,1,2     restrict to these cards and round-robin seeds onto them (empty = auto-scan all).
#   PER_GPU=1      max concurrent runs to pack onto one card (e.g. PER_GPU=2 puts 2 runs/card).
#   FREE_MB=2000   auto-scan only: a card counts as "free" when memory.used < FREE_MB.
#   EVAL_NUM_ENVS  passed to each run (fewer eval envs => less GPU mem + CPU; 50-episode
#                  metric unchanged). Unset => code default 8.
# One run ~10GB GPU mem (base + 8 eval envs). On a 96GB card PER_GPU=2..4 is comfortable;
# the real caps are CPU cores (env stepping) and RAM (replay buffers ~30-60GB/run), not GPU mem.
# Examples:
#   bash launch_three_piece_ablation_3seeds.sh nosubgoal                 # auto, 1 run/card
#   GPUS=0,1 PER_GPU=2 bash launch_three_piece_ablation_3seeds.sh nostage    # 3 seeds -> 0,1,0
#   GPUS=0 PER_GPU=3 EVAL_NUM_ENVS=4 bash launch_three_piece_ablation_3seeds.sh nosubgoal  # all 3 on card 0
#
# Reservations under logs/.gpu_resv/ (file resv_<pid> = gpu index) let same-batch and
# concurrent launches count runs per card and honor PER_GPU; auto-cleaned when the PID exits.
# If the mode-specific offcache does not exist, seed1 starts first; seed2/3 start after
# buffer_meta.json appears (so they reuse the cache instead of racing to rebuild it).
# Usage: [GPUS=..] [PER_GPU=..] [EVAL_NUM_ENVS=..] bash launch_three_piece_ablation_3seeds.sh <nosubgoal|nostage|nostage_nosubgoal>
set -euo pipefail
if [ "$#" -ne 1 ]; then
  echo "Usage: [GPUS=..] [PER_GPU=..] [EVAL_NUM_ENVS=..] bash launch_three_piece_ablation_3seeds.sh <nosubgoal|nostage|nostage_nosubgoal>" >&2
  exit 2
fi
MODE=$1
SEEDS=(1 2 3)
FREE_MB=${FREE_MB:-2000}   # auto-scan: a GPU counts as "free" when memory.used < FREE_MB
GPUS="${GPUS:-}"           # e.g. GPUS=0,1,2 to restrict+round-robin; empty = auto-scan all
PER_GPU="${PER_GPU:-1}"    # max concurrent runs packed onto one card
[ -n "${EVAL_NUM_ENVS:-}" ] && export EVAL_NUM_ENVS   # propagate to the seed runner

case "$MODE" in
  nosubgoal)         OFFCACHE=outputs_chunk/three_piece_actfeat_bp_bc01_hiqlv512_sg15_staged_nosubgoal_offcache ;;
  nostage)           OFFCACHE=outputs_chunk/three_piece_actfeat_bp_bc01_hiqlv512_sg15_nostage_offcache ;;
  nostage_nosubgoal) OFFCACHE=outputs_chunk/three_piece_actfeat_bp_bc01_hiqlv512_sg15_nostage_nosubgoal_offcache ;;
  *)
    echo "Usage: [GPUS=..] [PER_GPU=..] [EVAL_NUM_ENVS=..] bash launch_three_piece_ablation_3seeds.sh <nosubgoal|nostage|nostage_nosubgoal>" >&2
    exit 2
    ;;
esac

cd /mnt/mnt/data/resfit || exit 3
mkdir -p logs
RESV_DIR=logs/.gpu_resv
mkdir -p "$RESV_DIR"

# Reservation file per run: name resv_<training_pid>, content = gpu index. Auto-cleaned
# when the training PID exits. This is the single source of truth for "runs per card".
clean_stale_resv(){
  local f pid
  shopt -s nullglob
  for f in "$RESV_DIR"/resv_*; do
    pid="${f##*/resv_}"
    if [ -z "$pid" ] || ! kill -0 "$pid" 2>/dev/null; then rm -f "$f"; fi
  done
  shopt -u nullglob
}

# How many live runs are currently reserved on gpu $1.
resv_count(){
  local target=$1 f g n=0
  shopt -s nullglob
  for f in "$RESV_DIR"/resv_*; do
    g=$(cat "$f" 2>/dev/null || true)
    [ "$g" = "$target" ] && n=$((n+1))
  done
  shopt -u nullglob
  echo "$n"
}

# Echo the GPU index for the next seed.
#  - GPUS set: round-robin over that set (fewest current runs first), up to PER_GPU per card.
#  - GPUS empty: least-loaded card by nvidia-smi with a free slot (< PER_GPU runs); warns if
#    even the freest card is >= FREE_MB used (no truly-free card left).
pick_gpu(){
  clean_stale_resv
  local gi gm best="" best_mem="" best_cnt="" c
  if [ -n "$GPUS" ]; then
    local -a pool
    IFS=',' read -ra pool <<< "$GPUS"
    for gi in "${pool[@]}"; do
      gi=$(echo "$gi" | tr -d ' '); [ -z "$gi" ] && continue
      c=$(resv_count "$gi")
      [ "$c" -ge "$PER_GPU" ] && continue
      if [ -z "$best_cnt" ] || [ "$c" -lt "$best_cnt" ]; then best="$gi"; best_cnt="$c"; fi
    done
    if [ -z "$best" ]; then
      echo "[launcher] no free slot: every card in GPUS=$GPUS already has PER_GPU=$PER_GPU runs" >&2
      return 1
    fi
    echo "$best"; return 0
  fi
  while IFS=',' read -r gi gm; do
    gi=$(echo "$gi" | tr -d ' '); gm=$(echo "$gm" | tr -d ' ')
    [ -z "$gi" ] && continue
    [ "$(resv_count "$gi")" -ge "$PER_GPU" ] && continue
    if [ -z "$best_mem" ] || [ "$gm" -lt "$best_mem" ]; then best="$gi"; best_mem="$gm"; fi
  done < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits)
  if [ -z "$best" ]; then
    echo "[launcher] no GPU available (every card already has PER_GPU=$PER_GPU runs)" >&2
    return 1
  fi
  if [ "$best_mem" -ge "$FREE_MB" ]; then
    echo "[launcher] WARN: no truly-free GPU (freest is gpu $best at ${best_mem}MiB >= ${FREE_MB}MiB); packing onto it" >&2
  fi
  echo "$best"
}

# Pick a GPU, launch the seed on it detached, and reserve that card under the training PID.
launch_one(){
  local idx=$1
  local SEED=${SEEDS[$idx]}
  local GPU
  GPU=$(pick_gpu) || exit 6
  local LOG=logs/three_piece_${MODE}_seed${SEED}.log
  echo "[launcher] start mode=$MODE seed=$SEED gpu=$GPU per_gpu=$PER_GPU eval_num_envs=${EVAL_NUM_ENVS:-8(default)} log=$LOG" >&2
  # nohup (not setsid): nohup exec's, so $! is the real long-lived training bash PID (setsid
  # fork-exits, leaving $! a dead pid -> reservations get cleaned early and the seed1 liveness
  # check false-fails). nohup also ignores SIGHUP so the run survives terminal close; for long
  # staggered runs, launch the launcher itself under tmux/`setsid` so Ctrl-C can't reach seeds.
  nohup bash run_three_piece_ablation_seed.sh "$MODE" "$GPU" "$SEED" > "$LOG" 2>&1 < /dev/null &
  local cpid=$!
  echo "$GPU" > "$RESV_DIR/resv_${cpid}"   # reserve this card under the training PID
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
    echo "[launcher] seed1 exited before offcache became ready; inspect logs/three_piece_${MODE}_seed1.log" >&2
    exit 5
  fi
  sleep 60
done
