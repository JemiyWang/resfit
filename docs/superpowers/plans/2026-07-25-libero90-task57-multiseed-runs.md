# LIBERO-90 Task 57 Multi-Seed Runs Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop task 60, 63, and 64, preserve task 57 seed 0, and launch task 57 seeds 1 and 2 with otherwise identical training parameters.

**Architecture:** Treat the live task 57 seed 0 command as the configuration source of truth. Launch two independent trainer processes in seed-specific tmux sessions and output directories; reuse task 57's immutable prepared assets and shared offline cache.

**Tech Stack:** Bash, tmux, NVIDIA CUDA, Python, LIBERO, TorchRL, W&B.

## Global Constraints

- Keep task 57 seed 0 on GPU2 and PI0 service port 8000 alive.
- Run task 57 seed 1 on GPU3 and task 57 seed 2 on GPU4.
- Leave GPU5 unused after stopping task 64.
- Change only GPU/EGL device, seed, output directory, and W&B run name relative to task 57 seed 0.
- Preserve all task 60, 63, and 64 outputs and checkpoints.
- Reuse task 57 data, statistics, PI0 feature cache, offline replay cache, goal-conditioned value checkpoint, and high-level actor checkpoint read-only.

---

### Task 1: Preflight the Replacement Runs

**Files:**
- Read: `/mnt/mnt/data/resfit/run_libero90_shore_task.sh`
- Read: `/mnt/mnt/data/resfit/outputs_chunk/libero90_task57_pi0_feat.npz`
- Read: `/mnt/mnt/data/resfit/outputs_chunk/libero90_task57_pi0_feat_gc_value.pt`
- Read: `/mnt/mnt/data/resfit/outputs_chunk/libero90_task57_pi0_feat_high_actor.pt`
- Read: `/mnt/mnt/data/resfit/outputs_chunk/libero90_task57_pi0feat_bp_bc01_h10_offcache`

**Interfaces:**
- Consumes: live task 57 seed 0 process command and prepared task 57 assets.
- Produces: a pass/fail decision to stop old runs; no files are changed.

- [ ] **Step 1: Confirm seed 0 and the PI0 service are alive**

Run:

```bash
ps -p 2316336,2144856 -o pid,ppid,stat,cmd
ss -ltnp | rg ':8000\b'
```

Expected: PID 2316336 is task 57 with `--seed 0`; PID 2144856 is the PI0
service and port 8000 is listening.

- [ ] **Step 2: Confirm the three old trainer targets are exact**

Run:

```bash
ps -p 2320719,2323541,2331436 -o pid,ppid,stat,cmd
```

Expected: the three commands contain task IDs 60, 63, and 64 respectively.

- [ ] **Step 3: Reject session or output collisions**

Run:

```bash
tmux has-session -t libero90_shore_t57_s1
tmux has-session -t libero90_shore_t57_s2
ls -ld \
  /mnt/mnt/data/resfit/outputs_chunk/libero90_task57_pi0feat_bp_bc01_h10_seed1 \
  /mnt/mnt/data/resfit/outputs_chunk/libero90_task57_pi0feat_bp_bc01_h10_seed2
```

Expected: all four checks report that the targets do not exist. Stop and
request direction instead of overwriting if any target exists.

### Task 2: Stop Task 60, 63, and 64 Gracefully

**Files:**
- Preserve: `/mnt/mnt/data/resfit/outputs_chunk/libero90_task60_pi0feat_bp_bc01_h10`
- Preserve: `/mnt/mnt/data/resfit/outputs_chunk/libero90_task63_pi0feat_bp_bc01_h10`
- Preserve: `/mnt/mnt/data/resfit/outputs_chunk/libero90_task64_pi0feat_bp_bc01_h10`

**Interfaces:**
- Consumes: validated tmux sessions and trainer PIDs from Task 1.
- Produces: free GPU3, GPU4, and GPU5 with the old processes exited.

- [ ] **Step 1: Send interrupt to each exact tmux session**

Run:

```bash
tmux send-keys -t libero90_shore_t60 C-c
tmux send-keys -t libero90_shore_t63 C-c
tmux send-keys -t libero90_shore_t64 C-c
```

Expected: each command exits 0.

- [ ] **Step 2: Verify the old trainers exit**

Run:

```bash
ps -p 2320719,2323541,2331436 -o pid,ppid,stat,cmd
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
```

Expected: no old trainer rows remain; GPU3-GPU5 release their trainer and
worker allocations. If a trainer remains after graceful shutdown, send
`Ctrl-C` once more and inspect its process state before escalating.

- [ ] **Step 3: Reconfirm protected processes**

Run:

```bash
ps -p 2316336,2144856 -o pid,ppid,stat,cmd
```

Expected: task 57 seed 0 and the PI0 service remain alive.

### Task 3: Launch Task 57 Seed 1 on GPU3

**Files:**
- Create at runtime: `/mnt/mnt/data/resfit/logs/libero90_shore/task57_seed1_train.log`
- Create at runtime: `/mnt/mnt/data/resfit/outputs_chunk/libero90_task57_pi0feat_bp_bc01_h10_seed1`

**Interfaces:**
- Consumes: task 57 prepared assets and PI0 service on port 8000.
- Produces: tmux session `libero90_shore_t57_s1`.

- [ ] **Step 1: Start the seed 1 tmux session**

Run:

```bash
tmux new-session -d -s libero90_shore_t57_s1 \
  "cd /mnt/mnt/data/resfit && exec env \
  -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  CUDA_VISIBLE_DEVICES=3 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  MUJOCO_EGL_DEVICE_ID=3 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=/mnt/mnt/data/wam-b1k/third_party/LIBERO:/mnt/mnt/data/resfit \
  /mnt/mnt/data/envs/resfit-libero/bin/python -u -m \
  resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --env_family libero --libero_suite libero_90 --libero_task_id 57 \
  --libero_stats_json /mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/converted/libero_90/task57/meta/stats.json \
  --actor raw --chunk_length 1 --action_scale 0.05 --min_range_per_dim 0.1 \
  --gamma 0.99 --n_step 3 --offline_base_mode base_policy \
  --base_policy_type pi05 --base_action_mode queue \
  --pi0_host 127.0.0.1 --pi0_port 8000 --pi0_action_dim 7 \
  --pi0_execute_horizon 10 --offline_fraction 0.5 --demo_bc_coef 0.1 \
  --offline_buffer_cache /mnt/mnt/data/resfit/outputs_chunk/libero90_task57_pi0feat_bp_bc01_h10_offcache \
  --subgoal_conditioned \
  --gc_value_ckpt /mnt/mnt/data/resfit/outputs_chunk/libero90_task57_pi0_feat_gc_value.pt \
  --high_actor_ckpt /mnt/mnt/data/resfit/outputs_chunk/libero90_task57_pi0_feat_high_actor.pt \
  --pi0_feat_cache /mnt/mnt/data/resfit/outputs_chunk/libero90_task57_pi0_feat.npz \
  --subgoal_way_steps 25 --renorm_subgoal --eval_num_envs 8 \
  --eval_num_episodes 10 --total_env_steps 500000 --learning_starts 10000 \
  --utd 4 --device cuda --seed 1 \
  --output_dir /mnt/mnt/data/resfit/outputs_chunk/libero90_task57_pi0feat_bp_bc01_h10_seed1 \
  --wandb_mode online --wandb_project dexmg-chunk-residual \
  --wandb_name libero90_task57_pi0feat_bp_bc01_h10_seed1 \
  >/mnt/mnt/data/resfit/logs/libero90_shore/task57_seed1_train.log 2>&1"
```

Expected: command exits 0 and creates the tmux session.

- [ ] **Step 2: Verify seed 1 startup**

Run:

```bash
tmux list-panes -t libero90_shore_t57_s1 \
  -F '#{pane_pid}|#{pane_dead}|#{pane_current_command}'
ps -eo pid,ppid,stat,cmd | rg 'libero_task_id 57.*--seed 1'
tail -80 /mnt/mnt/data/resfit/logs/libero90_shore/task57_seed1_train.log
```

Expected: pane is alive, process command has task 57 and seed 1, and the log
has no traceback, OOM, missing asset, or port error.

### Task 4: Launch Task 57 Seed 2 on GPU4

**Files:**
- Create at runtime: `/mnt/mnt/data/resfit/logs/libero90_shore/task57_seed2_train.log`
- Create at runtime: `/mnt/mnt/data/resfit/outputs_chunk/libero90_task57_pi0feat_bp_bc01_h10_seed2`

**Interfaces:**
- Consumes: task 57 prepared assets and PI0 service on port 8000.
- Produces: tmux session `libero90_shore_t57_s2`.

- [ ] **Step 1: Start the seed 2 tmux session**

Run:

```bash
tmux new-session -d -s libero90_shore_t57_s2 \
  "cd /mnt/mnt/data/resfit && exec env \
  -u http_proxy -u https_proxy -u HTTP_PROXY -u HTTPS_PROXY \
  CUDA_VISIBLE_DEVICES=4 MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
  MUJOCO_EGL_DEVICE_ID=4 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 \
  PYTHONPATH=/mnt/mnt/data/wam-b1k/third_party/LIBERO:/mnt/mnt/data/resfit \
  /mnt/mnt/data/envs/resfit-libero/bin/python -u -m \
  resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --env_family libero --libero_suite libero_90 --libero_task_id 57 \
  --libero_stats_json /mnt/mnt/data/chj/lerobot_cache/physical-intelligence/libero/converted/libero_90/task57/meta/stats.json \
  --actor raw --chunk_length 1 --action_scale 0.05 --min_range_per_dim 0.1 \
  --gamma 0.99 --n_step 3 --offline_base_mode base_policy \
  --base_policy_type pi05 --base_action_mode queue \
  --pi0_host 127.0.0.1 --pi0_port 8000 --pi0_action_dim 7 \
  --pi0_execute_horizon 10 --offline_fraction 0.5 --demo_bc_coef 0.1 \
  --offline_buffer_cache /mnt/mnt/data/resfit/outputs_chunk/libero90_task57_pi0feat_bp_bc01_h10_offcache \
  --subgoal_conditioned \
  --gc_value_ckpt /mnt/mnt/data/resfit/outputs_chunk/libero90_task57_pi0_feat_gc_value.pt \
  --high_actor_ckpt /mnt/mnt/data/resfit/outputs_chunk/libero90_task57_pi0_feat_high_actor.pt \
  --pi0_feat_cache /mnt/mnt/data/resfit/outputs_chunk/libero90_task57_pi0_feat.npz \
  --subgoal_way_steps 25 --renorm_subgoal --eval_num_envs 8 \
  --eval_num_episodes 10 --total_env_steps 500000 --learning_starts 10000 \
  --utd 4 --device cuda --seed 2 \
  --output_dir /mnt/mnt/data/resfit/outputs_chunk/libero90_task57_pi0feat_bp_bc01_h10_seed2 \
  --wandb_mode online --wandb_project dexmg-chunk-residual \
  --wandb_name libero90_task57_pi0feat_bp_bc01_h10_seed2 \
  >/mnt/mnt/data/resfit/logs/libero90_shore/task57_seed2_train.log 2>&1"
```

Expected: command exits 0 and creates tmux session
`libero90_shore_t57_s2`.

- [ ] **Step 2: Verify seed 2 startup**

Run:

```bash
tmux list-panes -t libero90_shore_t57_s2 \
  -F '#{pane_pid}|#{pane_dead}|#{pane_current_command}'
ps -eo pid,ppid,stat,cmd | rg 'libero_task_id 57.*--seed 2'
tail -80 /mnt/mnt/data/resfit/logs/libero90_shore/task57_seed2_train.log
```

Expected: pane is alive, process command has task 57 and seed 2, and the log
has no traceback, OOM, missing asset, or port error.

### Task 5: Compare Configuration and Verify the Final Allocation

**Files:**
- Read: `/mnt/mnt/data/resfit/logs/libero90_shore/task57_seed1_train.log`
- Read: `/mnt/mnt/data/resfit/logs/libero90_shore/task57_seed2_train.log`

**Interfaces:**
- Consumes: three live task 57 process commands.
- Produces: verified seed 0/1/2 task 57 allocation and final status report.

- [ ] **Step 1: Capture all three task 57 commands**

Run:

```bash
ps -eo pid,ppid,stat,cmd | rg 'train_chunk_residual.*libero_task_id 57'
```

Expected: exactly three trainer roots with seeds 0, 1, and 2 on GPU2, GPU3,
and GPU4 respectively.

- [ ] **Step 2: Confirm GPU and service allocation**

Run:

```bash
nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv,noheader
ss -ltnp | rg ':8000\b'
```

Expected: GPU2-GPU4 have the intended runs, GPU5 has no compute allocation,
and the PI0 service still listens on port 8000.

- [ ] **Step 3: Scan new logs for fatal startup errors**

Run:

```bash
rg -n -i \
  'Traceback|CUDA out of memory|out of memory|Exception|missing asset|connection refused|nan' \
  /mnt/mnt/data/resfit/logs/libero90_shore/task57_seed1_train.log \
  /mnt/mnt/data/resfit/logs/libero90_shore/task57_seed2_train.log
```

Expected: no matches.
