#!/usr/bin/env bash
# pouring as01:在 as02 基础上把 action_scale 0.2 -> 0.1(as02 已塌方:0.76@10k -> 0.00)。
# stage3-only:复用 as02 已训好的 gc_value/high_actor/act_feat(均与 action_scale 无关),
#   只重建 offcache(action_scale 进签名+被 action_scaler.scale 烘进存储动作,0.2 的 cache 对 0.1 无效)。
#   offcache 用全新路径 _as01_offcache,主训启动时自动构建(~40min,~41G)。
# 其余 step3 参数与 run_pouring_hiqlv512_sg15_bcdecay_as02.sh 第66-78行逐字一致。
# 用法: setsid bash run_pouring_hiqlv512_sg15_bcdecay_as01.sh <gpu> > pouring_hiqlv512_sg15_bcdecay_as01.log 2>&1 < /dev/null &
set -u
GPU=${1:-7}
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
CACHE=outputs_chunk/pouring_act_feat_hdf5.npz                                    # 复用(与 action_scale 无关)
GC=outputs_chunk/pouring_gc_value_actfeat_hdf5_hiqlv512.pt                       # 复用(与 action_scale 无关)
HA=outputs_chunk/pouring_high_actor_actfeat_hdf5_hiqlv512.pt                     # 复用(与 action_scale 无关)
OFFCACHE=outputs_chunk/pouring_actfeat_hdf5_bp_hiqlv512_sg15_as01_offcache      # 新路径,会重建(action_scale 0.1)
OUT=outputs_chunk/pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_bcdecay_as01        # 新
ACTION_SCALE=0.1
gate(){ [ -s "$1" ] || { echo "[as01] FATAL 复用产物缺失: $1"; exit 4; }; }

echo "[as01] gate 复用产物(gc_value/high_actor/act_feat)..."
gate "$CACHE"; gate "$GC"; gate "$HA"
echo "[as01] START pouring ($TASK) hiqlv512 sg15 bcdecay action_scale=$ACTION_SCALE  GPU=$CUDA_VISIBLE_DEVICES  $(date '+%F %T')"
echo "[as01] 将自动构建新 offcache: $OFFCACHE"

# === 主训(与 as02 step3 逐字一致,仅 action_scale / offcache / wandb_name / output_dir 改)===
$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task "$TASK" --base_wandb_id "$BASE" --dataset "$DS" \
  --offline_dataset_path "$HDF5" \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale "$ACTION_SCALE" --actor_lr 1e-6 --actor raw \
  --reward_shaping none \
  --offline_fraction 0.5 --offline_base_mode base_policy --demo_bc_coef 0.1 --bc_coef_final 0.0 \
  --offline_buffer_cache "$OFFCACHE" \
  --subgoal_conditioned --gc_value_ckpt "$GC" --high_actor_ckpt "$HA" --act_feat_cache "$CACHE" \
  --subgoal_way_steps 15 \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual --wandb_name pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_bcdecay_as01 \
  --output_dir "$OUT"
echo "[as01] ALL DONE pouring hiqlv512 sg15 bcdecay as01 $(date '+%F %T')  out=$OUT"
