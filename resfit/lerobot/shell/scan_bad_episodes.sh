#!/usr/bin/env bash
# 扫描某个 lerobot 数据集本地缓存里"损坏的视频",输出逗号分隔的坏 episode 索引,
# 直接喂给 train_bc_dexmg 的 --exclude_episodes。
#
# 背景:部分 ankile dexmg 数据集的少量 mp4 是 AV1 源文件损坏(libdav1d/libaom 都报
#       Corrupt frame),训练时 DataLoader 采到就崩。用全量 ffmpeg 解码找出来排除即可。
#       (已验证 ffmpeg 全解 == torchcodec 全解,结果一致。)
#
# 用法:  bash resfit/lerobot/shell/scan_bad_episodes.sh ankile/dexmg-two-arm-threading
#   前提:该数据集已下载到本地缓存(跑过一次 BC、或被 LeRobotDataset 拉过即有)。
#   输出(stdout):  17,540,581  (逗号分隔;无坏文件则为空)
#   进度/统计走 stderr,不污染 stdout,可直接 EXCL=$(bash scan_bad_episodes.sh <repo>)。
set -uo pipefail

REPO="${1:-}"
[ -n "$REPO" ] || { echo "用法: bash $0 <hf-repo-id, 如 ankile/dexmg-two-arm-threading>" >&2; exit 1; }

FF="$(conda run -n residual which ffmpeg 2>/dev/null || command -v ffmpeg || echo ffmpeg)"
BASE="$HOME/.cache/huggingface/lerobot/$REPO/videos"
[ -d "$BASE" ] || { echo "ERROR: 找不到本地缓存 $BASE(先下载该数据集)" >&2; exit 1; }

n_total=$(find "$BASE" -name "*.mp4" | wc -l)
echo "扫描 $REPO 的 $n_total 个视频(ffmpeg 全量解码)..." >&2

TMP="$(mktemp)"
find "$BASE" -name "*.mp4" -print0 \
  | xargs -0 -P 16 -I{} bash -c '
      if "'"$FF"'" -v error -xerror -i "{}" -f null - >/dev/null 2>&1; then :; else echo "{}"; fi
    ' > "$TMP"

n_bad=$(wc -l < "$TMP")
echo "坏文件数: $n_bad" >&2
grep -oE "episode_[0-9]+" "$TMP" | sort -u | sed "s/episode_0*//" | paste -sd, -
rm -f "$TMP"
