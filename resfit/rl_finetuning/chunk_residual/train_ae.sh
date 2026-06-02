#!/bin/bash
# M0.3 — 离线预训练并冻结动作自编码器(AE),供 residual_flow actor(M2)使用。
#
# 纯离线训练:只读 dexmg 数据集的动作、不开仿真,所以不需要 EGL 离屏渲染,只用
# CUDA_VISIBLE_DEVICES 把训练 pin 到一张卡(train_action_ae 自身默认 --device cuda)。
# 前台跑 + tee 日志,方便直接盯末尾的 go/no-go 指标。
#
# Usage:   bash resfit/rl_finetuning/chunk_residual/train_ae.sh <task|dataset> [gpu] [out_ckpt]
#   <task|dataset>  任务简称(dexmg 双臂),或完整 HF dataset id:
#                     boxcleanup coffee cansort drawer piece threading transport
#                   也可直接传完整 id(含 '/'),如 ankile/dexmg-two-arm-coffee
#   [gpu]           CUDA device index,默认 0
#   [out_ckpt]      AE 输出路径,默认 ckpt/ae_<task>.pt(按任务自动命名,避免互相覆盖)
#
# go/no-go:末尾 `[VAL] recon_l1` 应 < ~0.05(归一化到 [-1,1] 后)。
#          不达标就加 --steps / --hidden_dim 重训。--action_scale 必须与 RL 端一致(0.2)。
set -euo pipefail

TASK="${1:-}"
GPU="${2:-0}"

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"

if [[ -z "$TASK" ]]; then
    echo "Usage: bash $0 <task|dataset> [gpu] [out_ckpt]"
    echo "  <task>: boxcleanup|coffee|cansort|drawer|piece|threading|transport (或完整 HF dataset id)"
    echo "  例:    bash $0 drawer 3"
    exit 1
fi

# 任务简称 -> HF dataset(与 residual_td3.py 各 Config 对齐);含 '/' 视为已是完整 id。
# SLUG 用于命名 ckpt/日志,保持规范简称,避免任务间互相覆盖。
case "$TASK" in
    boxcleanup|box)                  DATASET="ankile/dexmg-two-arm-box-cleanup";          SLUG="boxcleanup" ;;
    coffee)                          DATASET="ankile/dexmg-two-arm-coffee";               SLUG="coffee"     ;;
    cansort|cansorting)              DATASET="ankile/dexmg-two-arm-can-sort-random";      SLUG="cansort"    ;;
    drawer|drawercleanup)            DATASET="ankile/dexmg-two-arm-drawer-cleanup";       SLUG="drawer"     ;;
    piece|threepiece|pieceassembly)  DATASET="ankile/dexmg-two-arm-three-piece-assembly"; SLUG="piece"      ;;
    threading|twoarmthreading)       DATASET="ankile/dexmg-two-arm-threading";            SLUG="threading"  ;;
    transport)                       DATASET="ankile/dexmg-two-arm-transport";            SLUG="transport"  ;;
    */*)                             DATASET="$TASK"; SLUG="$(basename "$TASK")" ;;
    *)
        echo "ERROR: 未知任务简称 '$TASK'。"
        echo "  已知: boxcleanup | coffee | cansort | drawer | piece | threading | transport"
        echo "  或直接传完整 HF dataset id(含 '/'),如 ankile/dexmg-two-arm-coffee"
        exit 1
        ;;
esac

OUT="${3:-resfit/rl_finetuning/chunk_residual/ckpt/ae_${SLUG}.pt}"
LOG="$REPO_ROOT/chunk_ae_${SLUG}_train.log"

cd "$REPO_ROOT"
mkdir -p "$(dirname "$OUT")"

echo "Task    : $TASK"
echo "Dataset : $DATASET"
echo "GPU     : $GPU"
echo "Out     : $OUT"
echo "Log     : ${LOG#"$REPO_ROOT"/}"
echo

CUDA_VISIBLE_DEVICES="$GPU" conda run -n residual --no-capture-output \
    python -m resfit.rl_finetuning.chunk_residual.train_action_ae \
        --dataset "$DATASET" \
        --chunk_length 20 --action_scale 0.2 --min_range_per_dim 0.1 \
        --steps 20000 --batch_size 256 \
        --out "$OUT" 2>&1 | tee "$LOG"

echo
echo "Done. go/no-go: grep recon_l1 '$LOG'   (want [VAL] recon_l1 < ~0.05)"
echo "若担心 SSH 断线,可改成: tmux new -s chunk_ae_${SLUG} \"bash $0 $TASK $GPU\""
