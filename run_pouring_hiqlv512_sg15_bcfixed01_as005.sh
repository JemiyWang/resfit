#!/usr/bin/env bash
# pouring as005:与 run_pouring_hiqlv512_sg15_bcdecay_as01.sh 对齐,仅两处改:
#   (1) bc 固定 0.1 不衰减 -> 删掉 --bc_coef_final(不传=固定 demo_bc_coef,逐位等价,见 train 脚本 help)。
#   (2) action_scale 0.1 -> 0.05。
# stage3-only:复用 as01/as02 已训好的 gc_value/high_actor/act_feat(均与 action_scale 无关),
#   只重建 offcache(action_scale 进签名+被 action_scaler.scale 烘进存储动作,0.1/0.2 的 cache 对 0.05 无效;
#   bc_coef 不进 offcache 签名,故仅 action_scale 触发重建)。offcache 用全新路径 _as005_offcache,
#   主训启动时自动构建(~40min,~41G)。
# 用法: setsid bash run_pouring_hiqlv512_sg15_bcfixed01_as005.sh <gpu> > pouring_hiqlv512_sg15_bcfixed01_as005.log 2>&1 < /dev/null &
set -u
GPU=${1:-0}
export PATH="/root/miniconda3/bin:$PATH"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1
export PYTHONPATH=/mnt/mnt/data/resfit
cd /mnt/mnt/data/resfit || exit 3
RUN="conda run -n residual --no-capture-output python -u"

TASK=TwoArmPouring
DS=ankile/dexmg-two-arm-pouring
HDF5=resfit/dataset/two_arm_pouring.hdf5
BASE=/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2
CACHE=outputs_chunk/pouring_act_feat_hdf5.npz                                     # 复用(与 action_scale 无关)
GC=outputs_chunk/pouring_gc_value_actfeat_hdf5_hiqlv512.pt                        # 复用(与 action_scale 无关)
HA=outputs_chunk/pouring_high_actor_actfeat_hdf5_hiqlv512.pt                      # 复用(与 action_scale 无关)
OFFCACHE=outputs_chunk/pouring_actfeat_hdf5_bp_hiqlv512_sg15_as005_offcache       # 新路径,会重建(action_scale 0.05)
OUT=outputs_chunk/pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005                # 新(bc 固定 0.1,无 bcdecay)
ACTION_SCALE=0.05
gate(){ [ -s "$1" ] || { echo "[as005] FATAL 复用产物缺失: $1"; exit 4; }; }

echo "[as005] gate 复用产物(gc_value/high_actor/act_feat)..."
gate "$CACHE"; gate "$GC"; gate "$HA"
echo "[as005] START pouring ($TASK) hiqlv512 sg15 bc固定0.1 action_scale=$ACTION_SCALE  GPU=$CUDA_VISIBLE_DEVICES  $(date '+%F %T')"
echo "[as005] 将自动构建新 offcache: $OFFCACHE"

# === 主训(与 as01 step3 逐字一致,仅 action_scale / 删bc_coef_final / offcache / wandb_name / output_dir 改)===
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
  --wandb_project dexmg-chunk-residual --wandb_name pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005 \
  --output_dir "$OUT"
echo "[as005] ALL DONE pouring hiqlv512 sg15 bc固定0.1 as005 $(date '+%F %T')  out=$OUT"
