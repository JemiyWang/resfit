#!/usr/bin/env bash
# HIQL 对齐三改 A/B 训练驱动(detached,关掉对话不被杀)。
# ②clamp:在现 geom value 上训 high_actor(后台并发)。
# ①LN:先训 gc_value_ln,成功后再在其上训 high_actor_ln(级联,串行)。
# 纯 CPU 训练,不抢 GPU。日志各自落 three_piece_*.log,总日志 run_hiql_align_ab.driver.log。
set -u
export PATH="/root/miniconda3/bin:$PATH"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
# 限 torch/BLAS 线程,防两进程并发抢核超额订阅(小 MLP 训练 8 线程足够)
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
cd /mnt/mnt/data/resfit || exit 3

HDF5=resfit/dataset/two_arm_three_piece_assembly.hdf5
DS=ankile/dexmg-two-arm-three-piece-assembly
STAGES=outputs_chunk/three_piece_stages.npz
STATE30=outputs_chunk/three_piece_state30.npz
RUN="conda run -n residual python -u -m"

echo "[driver] START $(date '+%F %T')"

# ===== A/B ② clamp:high_actor on existing geom value(后台并发)=====
$RUN resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
  --hdf5 "$HDF5" --dataset "$DS" --stage_cache "$STAGES" --state30_cache "$STATE30" \
  --gc_value_ckpt outputs_chunk/three_piece_gc_value_geom.pt \
  --way_steps 25 --target_mode clamp_to_goal \
  --output outputs_chunk/three_piece_high_actor_geom_clamp.pt \
  > three_piece_high_actor_geom_clamp.log 2>&1 &
PID_CLAMP=$!
echo "[driver] ② clamp high_actor 已起 pid=$PID_CLAMP -> three_piece_high_actor_geom_clamp.log"

# ===== A/B ① LN:gc_value_ln 先(串行)=====
echo "[driver] ① 训 gc_value_ln ... $(date '+%T')"
$RUN resfit.rl_finetuning.chunk_residual.train_hiql_gc_value \
  --hdf5 "$HDF5" --dataset "$DS" --stage_cache "$STAGES" \
  --goal_future_mode geometric --use_layer_norm 1 \
  --output outputs_chunk/three_piece_gc_value_geom_ln.pt \
  > three_piece_gc_value_geom_ln.log 2>&1
RC_LNV=$?
echo "[driver] ① gc_value_ln 完成 rc=$RC_LNV $(date '+%T')"

# ===== ① LN 级联:在 ln value 上训 high_actor_ln(默认 fixed_waypoint,隔离 LN 单变量)=====
if [ "$RC_LNV" -eq 0 ] && [ -f outputs_chunk/three_piece_gc_value_geom_ln.pt ]; then
  echo "[driver] ① 训 high_actor_ln(挂 ln value)... $(date '+%T')"
  $RUN resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
    --hdf5 "$HDF5" --dataset "$DS" --stage_cache "$STAGES" --state30_cache "$STATE30" \
    --gc_value_ckpt outputs_chunk/three_piece_gc_value_geom_ln.pt \
    --way_steps 25 \
    --output outputs_chunk/three_piece_high_actor_geom_ln.pt \
    > three_piece_high_actor_geom_ln.log 2>&1
  echo "[driver] ① high_actor_ln 完成 rc=$? $(date '+%T')"
else
  echo "[driver] ① 跳过 high_actor_ln(gc_value_ln 失败 rc=$RC_LNV)"
fi

# ===== 等 ②clamp 后台收尾 =====
wait "$PID_CLAMP"
echo "[driver] ② clamp high_actor 完成 rc=$? $(date '+%T')"

echo "[driver] 产物:"
ls -la outputs_chunk/three_piece_high_actor_geom_clamp.pt \
       outputs_chunk/three_piece_gc_value_geom_ln.pt \
       outputs_chunk/three_piece_high_actor_geom_ln.pt 2>&1
echo "[driver] ALL DONE $(date '+%F %T')"
