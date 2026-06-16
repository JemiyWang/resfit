#!/usr/bin/env bash
# threading 重启专用：只跑 stage3 主训，复用已训好的 stage_cache / act_feat cache / gc_value / high_actor /
# offline buffer cache。主训参数逐字照搬 run_act_feat_threading.sh 的 stage3（行39-49，不改）；区别仅：
# 跳过 stage1(gc_value)/stage2(high_actor) 重训，且复用 offcache —— 不重建那 26G、不再写盘（磁盘紧）。
# threading 是有 stage 的任务，故保留 --offline_stage_cache（与 no-stage 的 pouring/lifttray 不同）。
# 用法: bash run_act_feat_threading_stage3_restart.sh <gpu>
set -u
GPU=${1:?usage: $0 <gpu>}
export PATH="/root/miniconda3/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com PYTHONPATH=/mnt/mnt/data/resfit
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
cd /mnt/mnt/data/resfit || exit 3
RUN="conda run -n residual --no-capture-output python -u"
HDF5=resfit/dataset/two_arm_threading.hdf5
DS=ankile/dexmg-two-arm-threading
STAGES=outputs_chunk/two_arm_threading_stages.npz
CACHE=outputs_chunk/two_arm_threading_act_feat.npz
BASE=resfit/out/threading/best
GCV=outputs_chunk/two_arm_threading_gc_value_actfeat.pt
HIA=outputs_chunk/two_arm_threading_high_actor_actfeat.pt
OFFCACHE=outputs_chunk/threading_actfeat_offcache

# stage3-only：前置产物必须齐全（本脚本不重训 gc/high、不重建 cache/stage）
for f in "$STAGES" "$CACHE" "$GCV" "$HIA" "$OFFCACHE"; do
  [ -e "$f" ] || { echo "[thr-restart] FATAL 缺前置产物: $f → 改跑完整 run_act_feat_threading.sh"; exit 4; }
done
echo "[thr-restart] stage3-only GPU=$CUDA_VISIBLE_DEVICES $(date '+%F %T')"
echo "[thr-restart] 复用 STAGES=$STAGES"
echo "[thr-restart] 复用 CACHE=$CACHE  GCV=$GCV  HIA=$HIA"
echo "[thr-restart] 复用 OFFCACHE=$OFFCACHE（应秒级加载，不重建）"

$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreading --base_wandb_id "$BASE" --dataset "$DS" \
  --offline_dataset_path "$HDF5" --offline_stage_cache "$STAGES" \
  --subgoal_conditioned --gc_value_ckpt "$GCV" --high_actor_ckpt "$HIA" \
  --act_feat_cache "$CACHE" \
  --reward_shaping none --demo_bc_coef 0.1 --action_scale 0.05 \
  --offline_fraction 0.5 --base_n_action_steps 10 \
  --offline_buffer_cache "$OFFCACHE" \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual --wandb_name threading_actfeat_bp_bc01 \
  --output_dir outputs_chunk/threading_actfeat_bp_bc01
echo "[thr-restart] EXIT rc=$? $(date '+%F %T')"
