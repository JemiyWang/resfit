#!/bin/bash
# M1.3 / M2.3 — chunk 级残差 RL 训练(raw 或 flow actor),按任务自动配 dataset/ae/base。
#
# 照 shell/run_rl.sh 的范式:GPU pin + EGL 离屏渲染(eval 要渲染) + residual conda env
# + tee 日志 + tmux detached(长训练,扛得住登出) + session 重名保护。
# 按 <task> 自动选 env 名 / dataset / 默认 AE ckpt / ACT 基座 wandb_id,与 train_ae.sh 风格统一。
#
# Usage:   bash resfit/rl_finetuning/chunk_residual/run_chunk_residual.sh <actor> <task> <gpu> [extra...]
#   <actor>  raw | flow
#   <task>   boxcleanup | threading
#   <gpu>    CUDA device index,例 5
#   extra    透传给 train_chunk_residual(在脚本默认之后,故可覆盖任何同名 flag),例:
#            正式:    --seed 0 --total_env_steps 300000
#            冒烟:    --smoke --eval_num_envs 2 --eval_num_episodes 2
#            覆盖基座: --base_wandb_id dexmg-twoarmthreading-bc/xxxx
#
# flow 会自动补该任务的默认 --ae_ckpt(ckpt/ae_<task>.pt),除非 extra 已显式给。
# 公平 A/B:raw 与 flow 用同 --task 同 --seed 同 --total_env_steps,只差 --actor。
set -euo pipefail

ACTOR="${1:-}"
TASK="${2:-}"
GPU="${3:-}"
shift 3 2>/dev/null || true
EXTRA=("$@")

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CKPT_DIR="resfit/rl_finetuning/chunk_residual/ckpt"

usage() {
    echo "Usage: bash $0 <actor:raw|flow> <task:boxcleanup|threading> <gpu> [extra...]"
    echo "  正式: bash $0 flow threading 5 --seed 0 --total_env_steps 300000"
    echo "  冒烟: bash $0 raw boxcleanup 5 --smoke --eval_num_envs 2 --eval_num_episodes 2"
}

if [[ "$ACTOR" != "raw" && "$ACTOR" != "flow" ]] || [[ -z "$TASK" || -z "$GPU" ]]; then
    usage; exit 1
fi

# 任务 -> env 名 / dataset / ACT 基座 wandb_id(与 residual_td3.py 各 Config 对齐)
case "$TASK" in
    boxcleanup)
        ENV="TwoArmBoxCleanup"
        DATASET="ankile/dexmg-two-arm-box-cleanup"
        BASE_WANDB="dexmg-boxcleanup-bc/d59wny58" ;;
    threading)
        ENV="TwoArmThreading"
        DATASET="ankile/dexmg-two-arm-threading"
        BASE_WANDB="dexmg-twoarmthreading-bc/TODO_FILL_BC_RUN_ID" ;;
    *)
        echo "ERROR: 未知任务 '$TASK'。已知: boxcleanup | threading"; usage; exit 1 ;;
esac
AE_CKPT_DEFAULT="$CKPT_DIR/ae_${TASK}.pt"

has_flag() { [[ " ${EXTRA[*]:-} " == *" $1 "* ]]; }

# 基座 wandb_id 还是占位 且 extra 没覆盖 -> 早失败,提示先训 BC 填 id。
if [[ "$BASE_WANDB" == *TODO* ]] && ! has_flag "--base_wandb_id"; then
    echo "ERROR: 任务 '$TASK' 的 ACT 基座 wandb_id 还是占位($BASE_WANDB)。"
    echo "  先训好它的 BC: bash resfit/lerobot/shell/run_bc.sh twoarmthreading $GPU"
    echo "  出 >0% 的 _best 后,用 extra 传入: --base_wandb_id dexmg-twoarmthreading-bc/xxxx"
    exit 1
fi

# 脚本默认放命令前段;extra 在后,argparse 后者覆盖前者,故 extra 可覆盖任何同名项。
DEFAULTS=(--task "$ENV" --dataset "$DATASET" --base_wandb_id "$BASE_WANDB")

# flow 需要 AE:extra 未给 --ae_ckpt 就补该任务默认,并检查文件存在。
AE_ARGS=()
if [[ "$ACTOR" == "flow" ]] && ! has_flag "--ae_ckpt"; then
    AE_ARGS=(--ae_ckpt "$AE_CKPT_DEFAULT")
    if [[ ! -f "$REPO_ROOT/$AE_CKPT_DEFAULT" ]]; then
        echo "ERROR: flow 需要 AE ckpt,但没找到 $AE_CKPT_DEFAULT"
        echo "  先跑: bash resfit/rl_finetuning/chunk_residual/train_ae.sh $TASK $GPU"
        exit 1
    fi
fi

SESSION="chunk_${TASK}_${ACTOR}"
LOG="$REPO_ROOT/chunk_${TASK}_${ACTOR}_train.log"

# 不覆盖同名进行中的 run。
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ERROR: tmux session '$SESSION' already exists(可能有 run 在跑)。"
    echo "  Attach : tmux attach -t $SESSION"
    echo "  Kill   : tmux kill-session -t $SESSION   # 确定要重跑再 kill"
    exit 1
fi

if has_flag "--base_wandb_id"; then BASE_SHOW="(由 extra --base_wandb_id 覆盖)"; else BASE_SHOW="$BASE_WANDB"; fi
echo "Actor   : $ACTOR"
echo "Task    : $TASK ($ENV)"
echo "GPU     : $GPU"
echo "Dataset : $DATASET"
echo "Base    : $BASE_SHOW"
echo "AE      : ${AE_ARGS[*]:-(none / raw)}"
echo "Extra   : ${EXTRA[*]:-(none)}"
echo "Session : $SESSION"
echo "Log     : ${LOG#"$REPO_ROOT"/}"

tmux new-session -d -s "$SESSION" \
    "cd '$REPO_ROOT' && CUDA_VISIBLE_DEVICES='$GPU' MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
     conda run -n residual --no-capture-output \
     python -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
       --actor '$ACTOR' ${DEFAULTS[*]} ${AE_ARGS[*]:-} ${EXTRA[*]:-} 2>&1 | tee '$LOG'"

echo
echo "Started. Attach: tmux attach -t $SESSION  (Ctrl-b d 脱离) | Tail: tail -f $LOG"
echo "Curve  : grep 'eval success_rate' $LOG"
