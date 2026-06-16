#!/usr/bin/env bash
# 重启专用：只跑 stage3 主训，复用已训好的 act_feat cache / gc_value / high_actor / offline buffer cache。
# 与 run_act_feat_hdf5_full_bp.sh 的 stage3 超参逐字一致（对齐 three_piece_actfeat_bp_bc01）；
# 区别仅：跳过 stage1(gc_value)/stage2(high_actor) 重训（产物已存在），且复用 offcache —— 不重建那
# 41G/62G、不再写 103G（pouring/lifttray 第一次正是建 offcache 把盘写满而崩）。
# pouring/lifttray 是 no-stage 任务，故不带 --stage_balanced / --offline_stage_cache（three_piece 专属）。
# 用法: CUDA_VISIBLE_DEVICES=N bash run_act_feat_hdf5_stage3_restart.sh <pouring|lifttray>
set -euo pipefail
KEY="${1:?用法: CUDA_VISIBLE_DEVICES=N bash $0 pouring|lifttray}"
case "$KEY" in
  pouring)  TASK=TwoArmPouring;  SUB=dexmg-two-arm-pouring;   HDF5=resfit/dataset/two_arm_pouring.hdf5;   BASE=/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2 ;;
  lifttray) TASK=TwoArmLiftTray; SUB=dexmg-two-arm-lift-tray; HDF5=resfit/dataset/two_arm_lift_tray.hdf5; BASE=/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_e0o0sckj_best:v4 ;;
  *) echo "未知 task: $KEY (只支持 pouring|lifttray)"; exit 2 ;;
esac

export PYTHONPATH=/mnt/mnt/data/resfit
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /mnt/mnt/data/resfit

DS=ankile/$SUB
CACHE=outputs_chunk/${KEY}_act_feat_hdf5.npz
GC=outputs_chunk/${KEY}_gc_value_actfeat_hdf5.pt
HA=outputs_chunk/${KEY}_high_actor_actfeat_hdf5.pt
OFFCACHE=outputs_chunk/${KEY}_actfeat_hdf5_bp_offcache
OUT=outputs_chunk/${KEY}_actfeat_hdf5_bp_bc01
RUN="conda run -n residual --no-capture-output python -u"

# stage3-only：前置产物必须齐全（本脚本不重训 gc/high、不重建 cache）
for f in "$CACHE" "$GC" "$HA" "$OFFCACHE"; do
  [ -e "$f" ] || { echo "[restart] FATAL 缺前置产物: $f → 该跑完整 run_act_feat_hdf5_full_bp.sh"; exit 4; }
done
echo "[restart] $KEY ($TASK) stage3-only GPU=${CUDA_VISIBLE_DEVICES:-?} $(date '+%F %T')"
echo "[restart] 复用 CACHE=$CACHE"
echo "[restart] 复用 GC=$GC  HA=$HA"
echo "[restart] 复用 OFFCACHE=$OFFCACHE（应秒级加载，不重建）"

$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task "$TASK" --base_wandb_id "$BASE" --dataset "$DS" \
  --offline_dataset_path "$HDF5" \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw \
  --reward_shaping none \
  --offline_fraction 0.5 --offline_base_mode base_policy --demo_bc_coef 0.1 \
  --offline_buffer_cache "$OFFCACHE" \
  --subgoal_conditioned --gc_value_ckpt "$GC" --high_actor_ckpt "$HA" --act_feat_cache "$CACHE" \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual --wandb_name "${KEY}_actfeat_hdf5_bp_bc01" \
  --output_dir "$OUT"
echo "[restart] DONE $KEY $(date '+%F %T')  out=$OUT"
