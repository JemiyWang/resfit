#!/usr/bin/env bash
# 看门狗:等 run_hiql_align_ab.sh 训完(driver 日志出现 ALL DONE),自动跑离线 gate 重验。
# 5 条:②clamp/基线 near 对照 + ①LN/基线 gc_value 三诊断 + ①LN high_actor final gate。
# 仅对存在的产物跑;每条各自落 *.verify/near.log,总日志 run_hiql_align_ab.verify.log。
# verify 走 read_per_demo_states(hdf5 直读,不渲染、不碰 GPU)。
set -u
export PATH="/root/miniconda3/bin:$PATH"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd /mnt/mnt/data/resfit || exit 3
RUN="conda run -n residual python -u -m"
DRIVERLOG=run_hiql_align_ab.driver.log

echo "[verify] WAIT 训练完成(轮询 $DRIVERLOG 的 ALL DONE)... $(date '+%F %T')"
for i in $(seq 1 720); do            # 最多等 6h(720×30s)
  if grep -q "ALL DONE" "$DRIVERLOG" 2>/dev/null; then
    echo "[verify] 检测到训练 ALL DONE(第 $i 次轮询)$(date '+%T')"; break
  fi
  sleep 30
done
if ! grep -q "ALL DONE" "$DRIVERLOG" 2>/dev/null; then
  echo "[verify] 超时未见 ALL DONE,改为 best-effort 验已存在产物 $(date '+%T')"
fi

run_one() {  # $1=描述 $2=日志 ;后续为命令
  local desc="$1" log="$2"; shift 2
  echo "[verify] >>> $desc -> $log  $(date '+%T')"
  "$@" > "$log" 2>&1
  echo "[verify] <<< $desc rc=$? $(date '+%T')"
}

# ===== ② clamp near 塌缩诊断:clamp vs fixed 基线 =====
if [ -f outputs_chunk/three_piece_high_actor_geom_clamp.pt ]; then
  run_one "② clamp near" three_piece_high_actor_geom_clamp.near.log \
    $RUN resfit.rl_finetuning.chunk_residual.verify_high_actor \
    --pt outputs_chunk/three_piece_high_actor_geom_clamp.pt --num_demos 40 --goal_mode near
else
  echo "[verify] 跳过 ② clamp near(产物缺失)"
fi
run_one "② fixed 基线 near" three_piece_high_actor_geom.near.log \
  $RUN resfit.rl_finetuning.chunk_residual.verify_high_actor \
  --pt outputs_chunk/three_piece_high_actor_geom.pt --num_demos 40 --goal_mode near

# ===== ① LN gc_value 三诊断:LN vs geom 基线 =====
if [ -f outputs_chunk/three_piece_gc_value_geom_ln.pt ]; then
  run_one "① LN gc_value gate" three_piece_gc_value_geom_ln.verify.log \
    $RUN resfit.rl_finetuning.chunk_residual.verify_gc_value \
    --pt outputs_chunk/three_piece_gc_value_geom_ln.pt --num_demos 40
else
  echo "[verify] 跳过 ① LN gc_value(产物缺失)"
fi
run_one "① geom 基线 gc_value gate" three_piece_gc_value_geom.verify.log \
  $RUN resfit.rl_finetuning.chunk_residual.verify_gc_value \
  --pt outputs_chunk/three_piece_gc_value_geom.pt --num_demos 40

# ===== ① LN high_actor final gate(确认 LN 没把高层带崩)=====
if [ -f outputs_chunk/three_piece_high_actor_geom_ln.pt ]; then
  run_one "① LN high_actor final gate" three_piece_high_actor_geom_ln.verify.log \
    $RUN resfit.rl_finetuning.chunk_residual.verify_high_actor \
    --pt outputs_chunk/three_piece_high_actor_geom_ln.pt --num_demos 40 \
    --out outputs_chunk/high_actor_verify_geom_ln.png
else
  echo "[verify] 跳过 ① LN high_actor final(产物缺失)"
fi

echo "[verify] ALL VERIFY DONE $(date '+%F %T')"
echo "[verify] 结果日志:three_piece_high_actor_geom_clamp.near.log / three_piece_high_actor_geom.near.log / three_piece_gc_value_geom_ln.verify.log / three_piece_gc_value_geom.verify.log / three_piece_high_actor_geom_ln.verify.log"
