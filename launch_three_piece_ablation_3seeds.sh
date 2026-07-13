#!/usr/bin/env bash
# Launch three_piece ablation seeds 1/2/3 safely.
# If the mode-specific offcache does not exist, seed1 starts first; seed2/3 start after buffer_meta.json appears.
# Usage: bash launch_three_piece_ablation_3seeds.sh <nobc|nostage> <gpu_seed1> <gpu_seed2> <gpu_seed3>
set -euo pipefail
if [ "$#" -ne 4 ]; then
  echo "Usage: bash launch_three_piece_ablation_3seeds.sh <nobc|nostage> <gpu_seed1> <gpu_seed2> <gpu_seed3>" >&2
  exit 2
fi
MODE=$1
GPUS=("$2" "$3" "$4")
SEEDS=(1 2 3)

case "$MODE" in
  nobc) OFFCACHE=outputs_chunk/three_piece_actfeat_bp_nobc_hiqlv512_sg15_staged_offcache ;;
  nostage) OFFCACHE=outputs_chunk/three_piece_actfeat_bp_bc01_hiqlv512_sg15_nostage_offcache ;;
  *)
    echo "Usage: bash launch_three_piece_ablation_3seeds.sh <nobc|nostage> <gpu_seed1> <gpu_seed2> <gpu_seed3>" >&2
    exit 2
    ;;
esac

cd /mnt/mnt/data/resfit || exit 3
mkdir -p logs

launch_one(){
  local idx=$1
  local SEED=${SEEDS[$idx]}
  local GPU=${GPUS[$idx]}
  local LOG=logs/three_piece_${MODE}_seed${SEED}.log
  echo "[launcher] start mode=$MODE seed=$SEED gpu=$GPU log=$LOG" >&2
  setsid bash run_three_piece_ablation_seed.sh "$MODE" "$GPU" "$SEED" > "$LOG" 2>&1 < /dev/null &
  echo $!
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
