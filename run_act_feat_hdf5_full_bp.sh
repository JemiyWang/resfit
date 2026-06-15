#!/usr/bin/env bash
# 全集正式实验:act_feat × hdf5 路线 × base_policy,超参对齐 three_piece_actfeat_bp_bc01。
# 与现有 run_act_feat_lerobot_full_bp.sh 的区别:离线 cache/主训全走 hdf5(--hdf5/--offline_dataset_path),
# 去 --data_source lerobot;真同源在线 env + ~100x 提速(省 pyav 解码)。
# 用法: CUDA_VISIBLE_DEVICES=<卡> bash run_act_feat_hdf5_full_bp.sh <pouring|lifttray>
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
gate(){ [ -s "$1" ] || { echo "[driver] FATAL 产物缺失: $1 (上一步失败,停)"; exit 4; }; }

echo "[driver] START $KEY ($TASK) hdf5 路 GPU=${CUDA_VISIBLE_DEVICES:-?} $(date '+%F %T')"

echo "[driver] 1) gc_value 全集(hdf5,首跑建 act_feat 缓存,~2min)... $(date '+%T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_hiql_gc_value \
  --hdf5 "$HDF5" --state_mode act_feat \
  --act_base_ckpt "$BASE" --dataset "$DS" \
  --act_feat_cache "$CACHE" --output "$GC"
gate "$CACHE"; gate "$GC"; echo "[driver] 1) gc_value DONE $(date '+%T')"

echo "[driver] 2) high_actor(命中缓存)... $(date '+%T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
  --hdf5 "$HDF5" --state_mode act_feat \
  --act_base_ckpt "$BASE" --dataset "$DS" \
  --act_feat_cache "$CACHE" --gc_value_ckpt "$GC" --output "$HA"
gate "$HA"; echo "[driver] 2) high_actor DONE $(date '+%T')"

echo "[driver] 3) 主训 500k(hdf5,base_policy,对齐 three_piece 超参,no-stage)... $(date '+%T')"
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
echo "[driver] ALL DONE $KEY $(date '+%F %T')  out=$OUT"
