#!/bin/bash
# Launch a residual-TD3 RL run for one of the paper tasks on this headless server.
#
# Wraps the bare official paper_runs/<task>/1_*_residual_rl.sh with the env this
# server needs (GPU pin + EGL offscreen rendering + the `residual` conda env) and
# a tee'd log, then starts it in a detached tmux session so it survives logout.
#
# Usage:   bash resfit/rl_finetuning/shell/run_rl.sh <task> <gpu>
#   <task>  one of: can square boxcleanup cansorting coffee
#   <gpu>   CUDA device index, e.g. 5
#
# Before launching a task make sure its base_policy.wandb_id is set in
# resfit/rl_finetuning/config/residual_td3.py and that its BC has produced a
# `_best` artifact (i.e. the BC log shows "New best success-rate").
set -euo pipefail

TASK="${1:-}"
GPU="${2:-}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
PAPER_RUNS="$REPO_ROOT/resfit/rl_finetuning/shell/paper_runs"

tasks_list() { ls -1d "$PAPER_RUNS"/*/ 2>/dev/null | xargs -n1 basename | tr '\n' ' '; }

if [[ -z "$TASK" || -z "$GPU" ]]; then
    echo "Usage: bash $0 <task> <gpu>"
    echo "  <task>: $(tasks_list)"
    echo "  <gpu> : CUDA device index, e.g. 5"
    exit 1
fi

# Resolve the residual-RL script (glob handles odd names like box_cleanup).
SCRIPT="$(ls "$PAPER_RUNS/$TASK"/1_*_residual_rl.sh 2>/dev/null | head -1 || true)"
if [[ -z "$SCRIPT" ]]; then
    echo "ERROR: no residual-RL script for task '$TASK' under $PAPER_RUNS/$TASK/"
    echo "Valid tasks: $(tasks_list)"
    exit 1
fi

SESSION="${TASK}_rl"
LOG="$REPO_ROOT/${TASK}_rl_train.log"

# Never clobber a run already in progress under the same session name.
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' already exists (a run may be in progress)."
    echo "  Attach : tmux attach -t $SESSION"
    echo "  Kill   : tmux kill-session -t $SESSION   # only if you are sure"
    exit 1
fi

echo "Task    : $TASK"
echo "GPU     : $GPU"
echo "Script  : ${SCRIPT#"$REPO_ROOT"/}"
echo "Session : $SESSION"
echo "Log     : ${LOG#"$REPO_ROOT"/}"

tmux new-session -d -s "$SESSION" \
    "cd '$REPO_ROOT' && CUDA_VISIBLE_DEVICES='$GPU' MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
     conda run -n residual --no-capture-output bash '$SCRIPT' 2>&1 | tee '$LOG'"

echo
echo "Started. Attach: tmux attach -t $SESSION  (Ctrl-b d to detach) | Tail: tail -f $LOG"
