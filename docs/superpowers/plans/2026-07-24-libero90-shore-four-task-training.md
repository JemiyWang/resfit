# LIBERO-90 SHORE Four-Task Training Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build and launch the task-specific SHORE asset and online-training pipeline for LIBERO-90 Tasks 57, 60, 63, and 64 on GPU2–5.

**Architecture:** A parameterized task runner owns one task's cache, hierarchy, and online commands. A fleet launcher owns the shared π0 feature server, sequential feature extraction, parallel hierarchy training, and staggered tmux online launches. All task data and outputs are isolated by task ID.

**Tech Stack:** Bash, tmux, Python/PyTorch, OpenPI/JAX websocket policy server, LIBERO, LeRobot v2, pytest.

## Global Constraints

- Work in `/mnt/mnt/data/resfit`; do not create a worktree.
- Keep `run_libero10_task8_pi0feat_bp_bc01_h10.sh` unchanged.
- Keep `/mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/meta/*` unchanged.
- GPU mapping is fixed: Task57→GPU2, Task60→GPU3, Task63→GPU4, Task64→GPU5.
- Do not stop or alter existing processes on GPU0, GPU1, GPU6, or GPU7.
- Base checkpoint is `/mnt/mnt/data/chj/openpi/checkpoints/pi0_libero/pi0_libero_is-dceq77jzghdxvjj2-devmachine-0_20260523_220232/29999`.
- Shared feature/base server is `127.0.0.1:8000`, config `pi0_libero`, pooling `last`.
- Online recipe remains H10, offline fraction 0.5, demo BC coefficient 0.1, 500k environment steps, seed 0.

---

### Task 1: Parameterized single-task runner

**Files:**
- Create: `run_libero90_shore_task.sh`
- Create: `resfit/rl_finetuning/chunk_residual/tests/test_libero90_shore_run_scripts.py`

**Interfaces:**
- Consumes: `run_libero90_shore_task.sh <cache|hierarchy|train> <task_id> <gpu_id>`.
- Produces: task-specific `.npz`, `gc_value.pt`, `high_actor.pt`, offline cache, and online output.
- Test-only interface: `DRY_RUN=1` prints shell-escaped commands without running them.

- [ ] **Step 1: Write failing task-runner tests**

```python
from pathlib import Path
import os
import subprocess

ROOT = Path(__file__).resolve().parents[4]
RUNNER = ROOT / "run_libero90_shore_task.sh"


def run(*args):
    return subprocess.run(
        ["bash", str(RUNNER), *map(str, args)],
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1"},
        text=True,
        capture_output=True,
    )


def test_task57_train_dry_run_uses_task_local_inputs():
    p = run("train", 57, 2)
    assert p.returncode == 0, p.stderr
    assert "--libero_suite libero_90" in p.stdout
    assert "--libero_task_id 57" in p.stdout
    assert "converted/libero_90/task57/meta/stats.json" in p.stdout
    assert "libero90_task57_pi0_feat.npz" in p.stdout
    assert "libero90_task57_pi0_feat_gc_value.pt" in p.stdout
    assert "libero90_task57_pi0_feat_high_actor.pt" in p.stdout
    assert "CUDA_VISIBLE_DEVICES=2" in p.stdout


def test_all_supported_task_mappings():
    for task_id, gpu_id in [(57, 2), (60, 3), (63, 4), (64, 5)]:
        p = run("cache", task_id, gpu_id)
        assert p.returncode == 0, p.stderr
        assert f"task{task_id}" in p.stdout


def test_wrong_gpu_mapping_fails():
    p = run("train", 63, 5)
    assert p.returncode != 0
    assert "expected GPU4" in p.stderr


def test_unknown_mode_fails():
    p = run("unknown", 57, 2)
    assert p.returncode != 0
```

- [ ] **Step 2: Run tests and confirm RED**

Run:

```bash
/mnt/mnt/data/envs/resfit-libero/bin/python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_libero90_shore_run_scripts.py -q
```

Expected: FAIL because `run_libero90_shore_task.sh` does not exist.

- [ ] **Step 3: Implement the single-task runner**

Create `run_libero90_shore_task.sh` with:

```bash
#!/usr/bin/env bash
set -euo pipefail

MODE=${1:-}
TASK_ID=${2:-}
GPU_ID=${3:-}
ROOT=/mnt/mnt/data/resfit
DATA_BASE=/mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/converted/libero_90
OUT="$ROOT/outputs_chunk"
PY_RES=/mnt/mnt/data/envs/residual/bin/python
PY_LIB=/mnt/mnt/data/envs/resfit-libero/bin/python
DRY_RUN=${DRY_RUN:-0}

case "$TASK_ID" in
  57) EXPECTED_GPU=2; TASK_LANG="pick up the cream cheese and put it in the tray" ;;
  60) EXPECTED_GPU=3; TASK_LANG="pick up the black bowl on the left and put it in the tray" ;;
  63) EXPECTED_GPU=4; TASK_LANG="stack the left bowl on the right bowl and place them in the tray" ;;
  64) EXPECTED_GPU=5; TASK_LANG="stack the right bowl on the left bowl and place them in the tray" ;;
  *) echo "unsupported task: $TASK_ID" >&2; exit 2 ;;
esac
[[ "$GPU_ID" == "$EXPECTED_GPU" ]] || {
  echo "Task $TASK_ID expected GPU$EXPECTED_GPU, got GPU$GPU_ID" >&2
  exit 2
}

DATA="$DATA_BASE/task$TASK_ID"
STATS="$DATA/meta/stats.json"
CACHE="$OUT/libero90_task${TASK_ID}_pi0_feat.npz"
GCV="$OUT/libero90_task${TASK_ID}_pi0_feat_gc_value.pt"
HA="$OUT/libero90_task${TASK_ID}_pi0_feat_high_actor.pt"
DUMMY="/tmp/libero90_task${TASK_ID}_pi0feat.h5"
OFFCACHE="$OUT/libero90_task${TASK_ID}_pi0feat_bp_bc01_h10_offcache"
ONLINE_OUT="$OUT/libero90_task${TASK_ID}_pi0feat_bp_bc01_h10"
IMGK=observation.images.agentview,observation.images.robot0_eye_in_hand

run() {
  if [[ "$DRY_RUN" == 1 ]]; then
    printf 'RUN'
    printf ' %q' "$@"
    printf '\n'
  else
    "$@"
  fi
}

if [[ "$DRY_RUN" != 1 ]]; then
  [[ -f "$STATS" ]] || { echo "missing task data: $STATS" >&2; exit 2; }
  mkdir -p "$OUT"
fi

export CUDA_VISIBLE_DEVICES="$GPU_ID"
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
export MUJOCO_EGL_DEVICE_ID="$GPU_ID"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONPATH="/mnt/mnt/data/wam-b1k/third_party/LIBERO:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY

case "$MODE" in
  cache)
    run env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY_RES" -u -m \
      resfit.rl_finetuning.chunk_residual.build_pi0_feat_cache_via_serve \
      --data_source libero --host 127.0.0.1 --port 8000 \
      --lerobot_root "$DATA" --language "$TASK_LANG" \
      --serve_ckpt_id pi0_libero --pooling last --out_cache "$CACHE"
    ;;
  hierarchy)
    if [[ "$DRY_RUN" != 1 ]]; then
      [[ -f "$CACHE" ]] || { echo "missing cache: $CACHE" >&2; exit 2; }
      "$PY_RES" -c "import h5py; h5py.File('$DUMMY', 'w').close()"
    fi
    run env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY_RES" -u -m \
      resfit.rl_finetuning.chunk_residual.train_hiql_gc_value \
      --state_mode pi0_feat --pi0_feat_cache "$CACHE" \
      --pi0_serve_ckpt_id pi0_libero --pi0_image_keys "$IMGK" \
      --pi0_proprio_key observation.state --pi0_pooling last \
      --pi0_prompt "$TASK_LANG" --hdf5 "$DUMMY" \
      --dataset "physical-intelligence/libero90-task$TASK_ID" \
      --device cuda --output "$GCV"
    run env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY_RES" -u -m \
      resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
      --state_mode pi0_feat --pi0_feat_cache "$CACHE" \
      --pi0_serve_ckpt_id pi0_libero --pi0_image_keys "$IMGK" \
      --pi0_proprio_key observation.state --pi0_pooling last \
      --pi0_prompt "$TASK_LANG" --gc_value_ckpt "$GCV" \
      --hdf5 "$DUMMY" --dataset "physical-intelligence/libero90-task$TASK_ID" \
      --device cuda --output "$HA"
    ;;
  train)
    if [[ "$DRY_RUN" != 1 ]]; then
      for f in "$CACHE" "$GCV" "$HA"; do
        [[ -f "$f" ]] || { echo "missing asset: $f" >&2; exit 2; }
      done
    fi
    run env CUDA_VISIBLE_DEVICES="$GPU_ID" "$PY_LIB" -u -m \
      resfit.rl_finetuning.chunk_residual.train_chunk_residual \
      --env_family libero --libero_suite libero_90 --libero_task_id "$TASK_ID" \
      --libero_stats_json "$STATS" --actor raw --chunk_length 1 \
      --action_scale 0.05 --min_range_per_dim 0.1 --gamma 0.99 --n_step 3 \
      --offline_base_mode base_policy --base_policy_type pi05 --base_action_mode queue \
      --pi0_host 127.0.0.1 --pi0_port 8000 --pi0_action_dim 7 \
      --pi0_execute_horizon 10 --offline_fraction 0.5 --demo_bc_coef 0.1 \
      --offline_buffer_cache "$OFFCACHE" --subgoal_conditioned \
      --gc_value_ckpt "$GCV" --high_actor_ckpt "$HA" --pi0_feat_cache "$CACHE" \
      --subgoal_way_steps 25 --renorm_subgoal \
      --eval_num_envs 8 --eval_num_episodes 10 \
      --total_env_steps 500000 --learning_starts 10000 --utd 4 \
      --device cuda --seed 0 --output_dir "$ONLINE_OUT" \
      --wandb_mode online --wandb_project dexmg-chunk-residual \
      --wandb_name "libero90_task${TASK_ID}_pi0feat_bp_bc01_h10"
    ;;
  *) echo "usage: $0 <cache|hierarchy|train> <task_id> <gpu_id>" >&2; exit 2 ;;
esac
```

- [ ] **Step 4: Run tests and shell syntax check**

Run:

```bash
bash -n run_libero90_shore_task.sh
/mnt/mnt/data/envs/resfit-libero/bin/python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_libero90_shore_run_scripts.py -q
```

Expected: syntax exit 0 and `4 passed`.

- [ ] **Step 5: Commit Task 1**

```bash
git add run_libero90_shore_task.sh \
  resfit/rl_finetuning/chunk_residual/tests/test_libero90_shore_run_scripts.py
git commit -m "feat: add LIBERO-90 SHORE task runner"
```

### Task 2: Fleet launcher and orchestration tests

**Files:**
- Create: `launch_libero90_shore_4gpu.sh`
- Modify: `resfit/rl_finetuning/chunk_residual/tests/test_libero90_shore_run_scripts.py`

**Interfaces:**
- Consumes: `PHASE=assets|train|all`, default `all`; calls the Task 1 runner.
- Produces: tmux sessions `libero90_shore_serve`, `libero90_shore_orchestrator`, and `libero90_shore_t<ID>`.
- Test-only interface: `DRY_RUN=1` prints the planned task order and exits without tmux/GPU mutation.

- [ ] **Step 1: Add failing launcher tests**

```python
LAUNCHER = ROOT / "launch_libero90_shore_4gpu.sh"


def test_launcher_dry_run_contains_all_task_gpu_pairs():
    p = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1", "PHASE": "all"},
        text=True,
        capture_output=True,
    )
    assert p.returncode == 0, p.stderr
    for task_id, gpu_id in [(57, 2), (60, 3), (63, 4), (64, 5)]:
        assert f"cache {task_id} {gpu_id}" in p.stdout
        assert f"hierarchy {task_id} {gpu_id}" in p.stdout
        assert f"train {task_id} {gpu_id}" in p.stdout
    assert "pi0_serve/serve_with_feat.py" in p.stdout


def test_launcher_rejects_unknown_phase():
    p = subprocess.run(
        ["bash", str(LAUNCHER)],
        cwd=ROOT,
        env={**os.environ, "DRY_RUN": "1", "PHASE": "bad"},
        text=True,
        capture_output=True,
    )
    assert p.returncode != 0
```

- [ ] **Step 2: Run tests and confirm RED**

Run the same pytest file. Expected: launcher tests fail because the launcher does not exist.

- [ ] **Step 3: Implement fleet launcher**

The launcher must implement these exact operations:

```bash
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

[[ "$PHASE" =~ ^(assets|train|all)$ ]] || {
  echo "PHASE must be assets|train|all" >&2
  exit 2
}

if [[ "$DRY_RUN" == 1 ]]; then
  echo "serve: $SERVE --config pi0_libero --dir $CKPT --port 8000 --pooling last"
  if [[ "$PHASE" == assets || "$PHASE" == all ]]; then
    for i in "${!TASKS[@]}"; do echo "$RUNNER cache ${TASKS[$i]} ${GPUS[$i]}"; done
    for i in "${!TASKS[@]}"; do echo "$RUNNER hierarchy ${TASKS[$i]} ${GPUS[$i]}"; done
  fi
  if [[ "$PHASE" == train || "$PHASE" == all ]]; then
    for i in "${!TASKS[@]}"; do echo "$RUNNER train ${TASKS[$i]} ${GPUS[$i]}"; done
  fi
  exit 0
fi

mkdir -p "$LOGROOT"
for path in "$RUNNER" "$OPENPI_PY" "$CKPT" "$SERVE"; do
  [[ -e "$path" ]] || { echo "missing dependency: $path" >&2; exit 2; }
done

if ! ss -ltn | grep -q ':8000 '; then
  tmux has-session -t libero90_shore_serve 2>/dev/null && {
    echo "tmux session exists but port 8000 is closed" >&2
    exit 2
  }
  tmux new-session -d -s libero90_shore_serve \
    "cd /mnt/mnt/data/resfit && unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY; \
     export CUDA_VISIBLE_DEVICES=2 XLA_PYTHON_CLIENT_PREALLOCATE=false; \
     exec $OPENPI_PY $SERVE --config pi0_libero --dir $CKPT --port 8000 \
       --pooling last >$LOGROOT/serve.log 2>&1"
  for _ in $(seq 1 120); do
    ss -ltn | grep -q ':8000 ' && break
    sleep 5
  done
  ss -ltn | grep -q ':8000 ' || { echo "pi0 serve failed" >&2; exit 2; }
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
  for pid in "${pids[@]}"; do wait "$pid"; done
fi

if [[ "$PHASE" == train || "$PHASE" == all ]]; then
  for i in "${!TASKS[@]}"; do
    task=${TASKS[$i]}
    gpu=${GPUS[$i]}
    session="libero90_shore_t${task}"
    log="$LOGROOT/task${task}_train.log"
    tmux has-session -t "$session" 2>/dev/null && {
      echo "training session already exists: $session" >&2
      exit 2
    }
    tmux new-session -d -s "$session" \
      "cd /mnt/mnt/data/resfit && exec $RUNNER train $task $gpu >$log 2>&1"
    for _ in $(seq 1 720); do
      grep -Eq '\[offline\] (已建|命中缓存)' "$log" 2>/dev/null && break
      tmux has-session -t "$session" 2>/dev/null || {
        echo "Task $task exited during startup; inspect $log" >&2
        exit 2
      }
      sleep 5
    done
    grep -Eq '\[offline\] (已建|命中缓存)' "$log" || {
      echo "Task $task offline-cache startup timed out" >&2
      exit 2
    }
  done
fi
```

- [ ] **Step 4: Run syntax and tests**

```bash
bash -n launch_libero90_shore_4gpu.sh
/mnt/mnt/data/envs/resfit-libero/bin/python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_libero90_shore_run_scripts.py -q
```

Expected: syntax exit 0 and `6 passed`.

- [ ] **Step 5: Commit Task 2**

```bash
git add launch_libero90_shore_4gpu.sh \
  resfit/rl_finetuning/chunk_residual/tests/test_libero90_shore_run_scripts.py
git commit -m "feat: orchestrate four LIBERO-90 SHORE runs"
```

### Task 3: Preflight, launch, and live verification

**Files:**
- Runtime logs: `logs/libero90_shore/*.log`
- Runtime assets: `outputs_chunk/libero90_task{57,60,63,64}_pi0_feat*`
- Runtime sessions: tmux only; no tracked files.

**Interfaces:**
- Consumes: Task 1 and Task 2 scripts, four converted datasets, π0 checkpoint.
- Produces: running asset/online pipeline and inspectable logs.

- [ ] **Step 1: Re-run relevant regressions**

```bash
/mnt/mnt/data/envs/resfit-libero/bin/python -m pytest \
  resfit/rl_finetuning/chunk_residual/tests/test_libero_offline.py \
  resfit/rl_finetuning/chunk_residual/tests/test_libero90_shore_run_scripts.py -q
```

Expected: all non-optional tests pass.

- [ ] **Step 2: Verify GPU and port preflight**

```bash
nvidia-smi
ss -ltnp
tmux ls
```

Expected: GPU2–5 have no unknown compute processes and port 8000 is free or owned by
`libero90_shore_serve`. Do not kill any conflicting process; report instead.

- [ ] **Step 3: Start the resumable orchestrator in tmux**

```bash
tmux new-session -d -s libero90_shore_orchestrator \
  "cd /mnt/mnt/data/resfit && PHASE=all bash launch_libero90_shore_4gpu.sh \
   >logs/libero90_shore/orchestrator.log 2>&1"
```

Expected: session exists, `libero90_shore_serve` appears, port 8000 begins listening,
and Task 57 feature-cache log starts advancing.

- [ ] **Step 4: Verify initial live state**

```bash
tmux ls
tail -80 logs/libero90_shore/orchestrator.log
tail -80 logs/libero90_shore/serve.log
nvidia-smi
```

Expected: no traceback/OOM; serve is on GPU2; cache extraction is making progress.

- [ ] **Step 5: Monitor phase transitions**

Inspect the orchestrator at bounded intervals. The required checkpoints are:

1. Four `.npz` caches load with 50 sequences and task-correct signatures.
2. Four `gc_value.pt` and four `high_actor.pt` files load successfully.
3. tmux sessions `libero90_shore_t57`, `t60`, `t63`, and `t64` exist.
4. Each training log reports the expected offline transition count:
   Task57=7647, Task60=6126, Task63=10721, Task64=11684.
5. Each log advances into the online loop without traceback/OOM.

- [ ] **Step 6: Final live report**

Report task→GPU→PID/session/log/output mapping and the current phase. Do not claim
online training is running until all four logs have passed offline-cache construction.
