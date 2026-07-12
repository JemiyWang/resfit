#!/usr/bin/env bash
# lifttray bcfixed01:对齐刚跑的 pouring as005(action_scale 0.05 + bc 固定 0.1 不衰减)。
#   lifttray 本就用 action_scale 0.05,与 as005 同;唯一相对现有 lifttray_*_bcdecay(bc 0.1→0)的改动:
#   bc 改为"固定 0.1 不衰减"(删 --bc_coef_final,不传=固定 demo_bc_coef)。
# stage3-only:复用现有 lifttray 三件套(gc_value/high_actor/act_feat 与 action_scale/bc 均无关)
#   + 复用现有 valid offcache(action_scale 0.05、gc_value/way_steps 全同、bc 不进签名 → 命中秒读,
#   不重建那 62G;offcache 静态无写入者,只读 memmap 与实验2并发安全)。
# lifttray 特有保持不动:hdf5 路 + no-stage(绝不加 --stage_balanced/--offline_stage_cache)。
# 用法: setsid bash run_lifttray_hiqlv512_sg15_bcfixed01.sh <gpu> > lifttray_hiqlv512_sg15_bcfixed01.log 2>&1 < /dev/null &
set -u
GPU=${1:-2}
export PATH="/root/miniconda3/bin:$PATH"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1
export PYTHONPATH=/mnt/mnt/data/resfit
cd /mnt/mnt/data/resfit || exit 3
RUN="conda run -n residual --no-capture-output python -u"

TASK=TwoArmLiftTray
DS=ankile/dexmg-two-arm-lift-tray
HDF5=resfit/dataset/two_arm_lift_tray.hdf5
BASE=/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_e0o0sckj_best:v4
CACHE=outputs_chunk/lifttray_act_feat_hdf5.npz                                   # 复用
GC=outputs_chunk/lifttray_gc_value_actfeat_hdf5_hiqlv512.pt                      # 复用
HA=outputs_chunk/lifttray_high_actor_actfeat_hdf5_hiqlv512.pt                    # 复用
OFFCACHE=outputs_chunk/lifttray_actfeat_hdf5_bp_hiqlv512_sg15_offcache           # 复用现有 valid offcache(命中,不重建)
OUT=outputs_chunk/lifttray_actfeat_hdf5_bp_bcfixed01_hiqlv512_sg15              # 新(bc 固定 0.1)
ACTION_SCALE=0.05
gate(){ [ -s "$1" ] || { echo "[lt-fix] FATAL 复用产物缺失: $1"; exit 4; }; }

echo "[lt-fix] gate 复用产物(gc_value/high_actor/act_feat + offcache meta)..."
gate "$CACHE"; gate "$GC"; gate "$HA"; gate "$OFFCACHE/buffer_meta.json"
echo "[lt-fix] START lifttray ($TASK) hiqlv512 sg15 bc固定0.1 action_scale=$ACTION_SCALE  GPU=$CUDA_VISIBLE_DEVICES  $(date '+%F %T')"

# === 主训(与现有 lifttray bcdecay 逐字一致,仅 bc 改固定 / wandb_name / output_dir)===
$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task "$TASK" --base_wandb_id "$BASE" --dataset "$DS" \
  --offline_dataset_path "$HDF5" \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale "$ACTION_SCALE" --actor_lr 1e-6 --actor raw \
  --reward_shaping none \
  --offline_fraction 0.5 --offline_base_mode base_policy --demo_bc_coef 0.1 \
  --offline_buffer_cache "$OFFCACHE" \
  --subgoal_conditioned --gc_value_ckpt "$GC" --high_actor_ckpt "$HA" --act_feat_cache "$CACHE" \
  --subgoal_way_steps 15 \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual --wandb_name lifttray_actfeat_hdf5_bp_bcfixed01_hiqlv512_sg15 \
  --output_dir "$OUT"
echo "[lt-fix] ALL DONE lifttray hiqlv512 sg15 bc固定0.1 $(date '+%F %T')  out=$OUT"
