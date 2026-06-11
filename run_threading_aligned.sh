#!/usr/bin/env bash
# threading 的"对齐 three_piece_aligned_bp_bc01"实验:HIQL 子目标分层(路线B)
# 链式产出依赖 + 跑对齐主训。用法: bash run_threading_aligned.sh <gpu>
# 逐级 gate:任一步失败或产物缺失即停。全程限线程(防 value/replay CPU 段霸核)。
set -u
GPU=${1:?usage: run_threading_aligned.sh <gpu>}
export PATH="/root/miniconda3/bin:$PATH"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1
cd /mnt/mnt/data/resfit || exit 3
RUN="conda run -n residual --no-capture-output python -u"

HDF5=resfit/dataset/two_arm_threading.hdf5
DS=ankile/dexmg-two-arm-threading
STAGES=outputs_chunk/two_arm_threading_stages.npz
STATE30=outputs_chunk/two_arm_threading_state30.npz
GCV=outputs_chunk/two_arm_threading_gc_value_aligned.pt
HIA=outputs_chunk/two_arm_threading_high_actor_aligned.pt
gate(){ [ -s "$1" ] || { echo "[driver] FATAL 产物缺失: $1 (上一步失败,停)"; exit 4; }; }

echo "[driver] START $(date '+%F %T')  GPU=$GPU"

# --- 1) state30 缓存(high_actor + 主训 subgoal_state30 都要)---
if [ -s "$STATE30" ]; then echo "[driver] 1) state30 已存,跳过"; else
  echo "[driver] 1) 建 state30 缓存(~83min replay)... $(date '+%T')"
  $RUN -c "from resfit.rl_finetuning.chunk_residual.state30_cache import load_or_build_state30; load_or_build_state30('$HDF5','$DS',None,'$STATE30'); print('STATE30_DONE')" \
    > two_arm_threading_state30.log 2>&1
  gate "$STATE30"; echo "[driver] 1) state30 完成 $(date '+%T')"
fi

# --- 2) gc_value(对齐默认 geom+LN+hiql,不带 flag)---
echo "[driver] 2) 训 gc_value_aligned ... $(date '+%T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_hiql_gc_value \
  --hdf5 "$HDF5" --dataset "$DS" --stage_cache "$STAGES" --output "$GCV" \
  > two_arm_threading_gc_value_aligned.log 2>&1
gate "$GCV"; echo "[driver] 2) gc_value 完成 $(date '+%T')"

# --- 3) high_actor(挂 gc_value + state30,way_steps/target_mode 用对齐默认)---
echo "[driver] 3) 训 high_actor_aligned ... $(date '+%T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
  --hdf5 "$HDF5" --dataset "$DS" --stage_cache "$STAGES" --state30_cache "$STATE30" \
  --gc_value_ckpt "$GCV" --output "$HIA" \
  > two_arm_threading_high_actor_aligned.log 2>&1
gate "$HIA"; echo "[driver] 3) high_actor 完成 $(date '+%T')"

# --- 4) 对齐主训(subgoal 分层;对齐 three_piece_aligned_bp_bc01)---
echo "[driver] 4) 对齐主训长跑 ... $(date '+%T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreading --base_wandb_id resfit/out/threading/best --dataset "$DS" \
  --offline_dataset_path "$HDF5" --offline_stage_cache "$STAGES" \
  --subgoal_conditioned --gc_value_ckpt "$GCV" --high_actor_ckpt "$HIA" \
  --subgoal_state30_cache "$STATE30" \
  --reward_shaping none \
  --demo_bc_coef 0.1 --action_scale 0.05 --offline_fraction 0.5 --base_n_action_steps 10 \
  --offline_buffer_cache outputs_chunk/threading_aligned_offcache \
  --wandb_project dexmg-chunk-residual --wandb_name threading_aligned_bp_bc01 \
  --output_dir outputs_chunk/threading_aligned_bp_bc01 \
  > threading_aligned_bp_bc01.log 2>&1
echo "[driver] 4) 主训退出 rc=$? $(date '+%T')"
echo "[driver] ALL DONE $(date '+%F %T')"
