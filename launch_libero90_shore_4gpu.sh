#!/usr/bin/env bash
set -euo pipefail
cd /mnt/mnt/data/resfit

PHASE=${PHASE:-all}
DRY_RUN=${DRY_RUN:-0}
TASKS=(57 60 63 64)
GPUS=(2 3 4 5)
RUNNER=/mnt/mnt/data/resfit/run_libero90_shore_task.sh
LOGROOT=/mnt/mnt/data/resfit/logs/libero90_shore
OPENPI_PY=/mnt/mnt/data/chj/openpi/.venv/bin/python
CKPT=/mnt/mnt/data/chj/openpi/checkpoints/pi0_libero/pi0_libero_is-dceq77jzghdxvjj2-devmachine-0_20260523_220232/29999
SERVE=/mnt/mnt/data/resfit/pi0_serve/serve_with_feat.py

if [[ ! "$PHASE" =~ ^(assets|train|all)$ ]]; then
  echo "PHASE must be assets|train|all" >&2
  exit 2
fi

if [[ "$DRY_RUN" == 1 ]]; then
  echo "serve: $SERVE --config pi0_libero --dir $CKPT --port 8000 --pooling last"
  if [[ "$PHASE" == assets || "$PHASE" == all ]]; then
    for i in "${!TASKS[@]}"; do
      echo "$RUNNER cache ${TASKS[$i]} ${GPUS[$i]}"
    done
    for i in "${!TASKS[@]}"; do
      echo "$RUNNER hierarchy ${TASKS[$i]} ${GPUS[$i]}"
    done
  fi
  if [[ "$PHASE" == train || "$PHASE" == all ]]; then
    for i in "${!TASKS[@]}"; do
      echo "$RUNNER train ${TASKS[$i]} ${GPUS[$i]}"
    done
  fi
  exit 0
fi

mkdir -p "$LOGROOT"
for path in "$RUNNER" "$OPENPI_PY" "$CKPT" "$SERVE"; do
  if [[ ! -e "$path" ]]; then
    echo "missing dependency: $path" >&2
    exit 2
  fi
done

port_is_listening() {
  ss -ltn | grep -q ':8000 '
}

if ! port_is_listening; then
  for gpu in "${GPUS[@]}"; do
    gpu_pids=$(nvidia-smi -i "$gpu" --query-compute-apps=pid --format=csv,noheader,nounits)
    if [[ -n "${gpu_pids//[[:space:]]/}" ]]; then
      echo "GPU$gpu has an existing compute process: $gpu_pids" >&2
      exit 2
    fi
  done
  if tmux has-session -t libero90_shore_serve 2>/dev/null; then
    echo "tmux session libero90_shore_serve exists but port 8000 is closed" >&2
    exit 2
  fi
  tmux new-session -d -s libero90_shore_serve \
    "cd /mnt/mnt/data/resfit && \
     unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; \
     export CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false; \
     exec $OPENPI_PY $SERVE --config pi0_libero --dir $CKPT --port 8000 \
       --pooling last >$LOGROOT/serve.log 2>&1"
  for _ in $(seq 1 120); do
    if port_is_listening; then
      break
    fi
    sleep 5
  done
  if ! port_is_listening; then
    echo "pi0 serve failed; inspect $LOGROOT/serve.log" >&2
    exit 2
  fi
else
  port_pid=$(ss -ltnp | sed -n 's/.*:8000 .*pid=\([0-9][0-9]*\).*/\1/p' | head -1)
  if [[ -z "$port_pid" ]]; then
    echo "port 8000 is occupied by an unidentified process" >&2
    exit 2
  fi
  port_cmd=$(ps -p "$port_pid" -o args=)
  if [[ "$port_cmd" != *"serve_with_feat.py"* || "$port_cmd" != *"pi0_libero"* ]]; then
    echo "port 8000 is not the expected pi0_libero feature server: $port_cmd" >&2
    exit 2
  fi
fi

if [[ "$PHASE" == assets || "$PHASE" == all ]]; then
  for i in "${!TASKS[@]}"; do
    "$RUNNER" cache "${TASKS[$i]}" "${GPUS[$i]}" \
      2>&1 | tee "$LOGROOT/task${TASKS[$i]}_cache.log"
  done

  pids=()
  for i in "${!TASKS[@]}"; do
    "$RUNNER" hierarchy "${TASKS[$i]}" "${GPUS[$i]}" \
      >"$LOGROOT/task${TASKS[$i]}_hierarchy.log" 2>&1 &
    pids+=("$!")
  done
  for pid in "${pids[@]}"; do
    wait "$pid"
  done
fi

if [[ "$PHASE" == train || "$PHASE" == all ]]; then
  for i in "${!TASKS[@]}"; do
    task=${TASKS[$i]}
    gpu=${GPUS[$i]}
    session="libero90_shore_t${task}"
    log="$LOGROOT/task${task}_train.log"
    if tmux has-session -t "$session" 2>/dev/null; then
      echo "training session already exists: $session" >&2
      exit 2
    fi
    tmux new-session -d -s "$session" \
      "cd /mnt/mnt/data/resfit && exec $RUNNER train $task $gpu >$log 2>&1"
    for _ in $(seq 1 720); do
      if grep -Eq '\[offline\] (已建|命中缓存)' "$log" 2>/dev/null; then
        break
      fi
      if ! tmux has-session -t "$session" 2>/dev/null; then
        echo "Task $task exited during startup; inspect $log" >&2
        exit 2
      fi
      sleep 5
    done
    if ! grep -Eq '\[offline\] (已建|命中缓存)' "$log"; then
      echo "Task $task offline-cache startup timed out" >&2
      exit 2
    fi
  done
fi
