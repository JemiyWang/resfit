#!/usr/bin/env bash
# ============================================================================
# block 的 pi0_feat V(s) —— 4 卡全量并行。放 tmux 里跑。
#
# 前置(你已完成):4 个 serve_block_awbc 已起,监听 8001/8002/8003/8004。
# 本脚本:4 分片并行建 ψ 缓存(success 295集 + failure 50集)→ 合并训 V。
# 预计:建缓存 ~4h(serve-bound,4卡并行) + 训 V ~几分钟。
#
# 用法:  bash run_block_v_4gpu.sh
#        SMOKE=1 bash run_block_v_4gpu.sh   # 每分片只 2 集,~10min 验证全流程
# ============================================================================
set -uo pipefail

# ---- 配置 ----
PORTS=(8001 8002 8003 8004)              # 4 个 serve 的端口(与你起的对齐)
NUM_SHARDS=4
BLOCK_ROOT=/mnt/mnt/data/domains_rise/block
POOLING=mean
SERVE_CKPT_ID=pi05_block_awbc_49999
PROMPT="build block"
OUT=/mnt/mnt/data/resfit/outputs_chunk
CACHE_DIR=$OUT/block_pi0feat_shards        # 分片缓存落这里
LOG_DIR=$OUT/block_pi0feat_logs
V=/mnt/mnt/data/envs/residual/bin/python   # full-stack:openpi_client + resfit + torchcodec
mkdir -p "$CACHE_DIR" "$LOG_DIR"

# torchcodec 的 LD_LIBRARY_PATH(读 block mp4 必需)
source /mnt/mnt/data/resfit/resfit/lerobot/shell/torchcodec_env.sh 2>/dev/null || true

NUM_DEMOS_ARG=""
[ -n "${SMOKE:-}" ] && NUM_DEMOS_ARG="--num_demos 2" && echo "[SMOKE] 每分片只 2 集"

cd /mnt/mnt/data/resfit

# ---- 一个分片建缓存的函数 ----
build_shard() {  # $1=split(block_success/block_fail) $2=shard_index $3=port
  local split=$1 shard=$2 port=$3
  $V -m resfit.rl_finetuning.chunk_residual.build_pi0_feat_cache_via_serve \
    --host 127.0.0.1 --port "$port" \
    --data_source teleavatar \
    --lerobot_root "$BLOCK_ROOT" --repo_id "$split" \
    --prompt "$PROMPT" --pooling "$POOLING" \
    --serve_ckpt_id "$SERVE_CKPT_ID" \
    --num_shards "$NUM_SHARDS" --shard_index "$shard" \
    --out_cache "$CACHE_DIR/${split#block_}_shard${shard}.npz" \
    $NUM_DEMOS_ARG \
    > "$LOG_DIR/${split#block_}_shard${shard}.log" 2>&1
}

# ---- 第 1 步:success 4 分片并行(4 serve) ----
echo "=== [1/3] 建 success 缓存(4 分片并行) $(date +%H:%M:%S) ==="
pids=()
for i in 0 1 2 3; do
  build_shard block_success "$i" "${PORTS[$i]}" &
  pids+=($!)
  echo "  shard $i → port ${PORTS[$i]} (pid $!)"
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 1 ] && { echo "✗ success 建缓存有分片失败,查 $LOG_DIR/success_shard*.log"; exit 1; }
echo "  success 完成 $(date +%H:%M:%S)"

# ---- 第 2 步:failure 4 分片并行 ----
echo "=== [2/3] 建 failure 缓存(4 分片并行) $(date +%H:%M:%S) ==="
pids=()
for i in 0 1 2 3; do
  build_shard block_fail "$i" "${PORTS[$i]}" &
  pids+=($!)
done
fail=0
for p in "${pids[@]}"; do wait "$p" || fail=1; done
[ "$fail" -eq 1 ] && { echo "✗ failure 建缓存有分片失败,查 $LOG_DIR/fail_shard*.log"; exit 1; }
echo "  failure 完成 $(date +%H:%M:%S)"

echo "=== 缓存清单 ==="
ls -la "$CACHE_DIR"/*.npz

# ---- 第 3 步:训 V(4 succ 分片 + 4 fail 分片 → success_signed ±1) ----
echo "=== [3/3] 训 block V $(date +%H:%M:%S) ==="
SUCC_ARGS=""; for i in 0 1 2 3; do SUCC_ARGS="$SUCC_ARGS --success_dataset $CACHE_DIR/success_shard${i}.npz"; done
FAIL_ARGS=""; for i in 0 1 2 3; do FAIL_ARGS="$FAIL_ARGS --failure_dataset $CACHE_DIR/fail_shard${i}.npz"; done

/mnt/mnt/data/chj/conda_envs/residual/bin/python \
  -m resfit.rl_finetuning.chunk_residual.train_hiql_value \
  --dataset "$BLOCK_ROOT/block_success" \
  --output "$OUT/block_value_pi0feat.pt" \
  --state_mode pi0_feat \
  --pi0_serve_ckpt_id "$SERVE_CKPT_ID" \
  --pi0_image_keys top_head,hand_left,hand_right \
  --pi0_proprio_key observation.state \
  --pi0_prompt "$PROMPT" --pi0_pooling "$POOLING" \
  --terminal_reward_mode success_signed \
  $SUCC_ARGS $FAIL_ARGS \
  --steps 50000 --seed 0 2>&1 | tee "$LOG_DIR/train_v.log"

echo "=== 完成: $OUT/block_value_pi0feat.pt  $(date +%H:%M:%S) ==="
