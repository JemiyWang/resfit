#!/bin/bash
# 把某个 BC 任务(launch_act_<...>.sh)丢进 detached tmux 后台跑,并 tee 到日志、防重复启动。
# 这是 RL 那个 run_rl.sh 的 BC 版。各 launch_act_*.sh 已自带 cd/EGL/conda/GPU 默认值,
# 本脚本只负责: tmux 后台 + 日志 + 防双开 + 可选覆盖 GPU。
#
# Usage:  bash resfit/lerobot/shell/run_bc.sh <task> [gpu]
#   <task>  匹配 launch_act_<task>.sh;若无精确匹配,则尝试唯一包含该词的 launch_act_*<task>*.sh
#           例: piece(=twoarmthreepieceassembly) / drawercleanup / boxcleanup / twoarmcoffee ...
#   [gpu]   可选 CUDA 卡号;不给就用 launch_act 脚本里自带的默认值
set -euo pipefail

TASK="${1:-}"
GPU="${2:-}"

SHELL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SHELL_DIR/../../.." && pwd)"

list_scripts() {
    ls -1 "$SHELL_DIR"/launch_act_*.sh 2>/dev/null \
        | xargs -n1 basename 2>/dev/null \
        | sed 's/^launch_act_//; s/\.sh$//' | tr '\n' ' ' || true
}

if [[ -z "$TASK" ]]; then
    echo "Usage: bash $0 <task> [gpu]"
    echo "  <task>: $(list_scripts)"
    echo "  [gpu] : CUDA 卡号(可选,不给用脚本默认)"
    exit 1
fi

# 1) 先精确匹配 launch_act_<task>.sh
SCRIPT="$SHELL_DIR/launch_act_${TASK}.sh"
if [[ ! -f "$SCRIPT" ]]; then
    # 2) 否则唯一模糊匹配 launch_act_*<task>*.sh
    matches=( "$SHELL_DIR"/launch_act_*"${TASK}"*.sh )
    if [[ ${#matches[@]} -eq 1 && -f "${matches[0]}" ]]; then
        SCRIPT="${matches[0]}"
    else
        echo "ERROR: 没有唯一匹配 '$TASK' 的 BC 脚本。"
        echo "  可选 task: $(list_scripts)"
        echo "  (也可直接传完整名,如 twoarmthreepieceassembly)"
        exit 1
    fi
fi

SESSION="${TASK}_bc"
LOG="$REPO_ROOT/${TASK}_bc_train.log"

# 同名 session 已存在就拒绝,防止两个进程抢同一张卡
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' 已存在(可能正在跑)。"
    echo "  Attach: tmux attach -t $SESSION"
    echo "  Kill  : tmux kill-session -t $SESSION   # 确认要重开再 kill"
    exit 1
fi

# 可选覆盖 GPU:launch_act 里是 CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-N}",这里给了就生效
GPU_PREFIX=""
[[ -n "$GPU" ]] && GPU_PREFIX="CUDA_VISIBLE_DEVICES='$GPU' "

echo "Task    : $TASK"
echo "Script  : ${SCRIPT#"$REPO_ROOT"/}"
echo "GPU     : ${GPU:-(脚本默认)}"
echo "Session : $SESSION"
echo "Log     : ${LOG#"$REPO_ROOT"/}"

tmux new-session -d -s "$SESSION" \
    "${GPU_PREFIX}bash '$SCRIPT' 2>&1 | tee '$LOG'"

echo
echo "Started. Attach: tmux attach -t $SESSION  (Ctrl-b d 退出) | Tail: tail -f $LOG"
