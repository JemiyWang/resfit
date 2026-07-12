#!/usr/bin/env bash
# lifttray stage3-only 重启:复用已存在的 gc_value/high_actor/act_feat/offcache,
#   只重跑第 3 步主训。step3 命令与 run_lifttray_hiqlv512_sg15_bcdecay.sh 第61-73行逐字一致。
# 背景:2026-06-19 02:12 上一轮主进程被 SIGKILL(非磁盘写错),在 env_steps ~50k 处中断;
#       已先备份上一轮 best.pt -> best.pt.run1_eval0840_20260618(无断点续训,会从头覆盖 best.pt)。
# 用法: setsid bash run_lifttray_hiqlv512_sg15_bcdecay_restart.sh <gpu> > lifttray_hiqlv512_sg15_bcdecay_restart.log 2>&1 < /dev/null &
set -u
GPU=${1:-1}
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
CACHE=outputs_chunk/lifttray_act_feat_hdf5.npz                                   # 复用现有特征缓存(1004M)
GC=outputs_chunk/lifttray_gc_value_actfeat_hdf5_hiqlv512.pt                      # 复用(上一轮所用)
HA=outputs_chunk/lifttray_high_actor_actfeat_hdf5_hiqlv512.pt                    # 复用(上一轮所用)
OFFCACHE=outputs_chunk/lifttray_actfeat_hdf5_bp_hiqlv512_sg15_offcache          # 复用现有 62G(签名已核对吻合)
OUT=outputs_chunk/lifttray_actfeat_hdf5_bp_bc01_hiqlv512_sg15_bcdecay           # 同上一轮
gate(){ [ -s "$1" ] || { echo "[restart] FATAL 复用产物缺失: $1"; exit 4; }; }

echo "[restart] gate 复用产物..."
gate "$CACHE"; gate "$GC"; gate "$HA"; gate "$OFFCACHE/buffer_meta.json"
echo "[restart] START lifttray ($TASK) hiqlv512 sg15 bcdecay  GPU=$CUDA_VISIBLE_DEVICES  $(date '+%F %T')"

# === 主训(与原脚本 step3 逐字一致)===
$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task "$TASK" --base_wandb_id "$BASE" --dataset "$DS" \
  --offline_dataset_path "$HDF5" \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw \
  --reward_shaping none \
  --offline_fraction 0.5 --offline_base_mode base_policy --demo_bc_coef 0.1 --bc_coef_final 0.0 \
  --offline_buffer_cache "$OFFCACHE" \
  --subgoal_conditioned --gc_value_ckpt "$GC" --high_actor_ckpt "$HA" --act_feat_cache "$CACHE" \
  --subgoal_way_steps 15 \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual --wandb_name lifttray_actfeat_hdf5_bp_bc01_hiqlv512_sg15_bcdecay \
  --output_dir "$OUT"
echo "[restart] ALL DONE lifttray hiqlv512 sg15 bcdecay $(date '+%F %T')  out=$OUT"
