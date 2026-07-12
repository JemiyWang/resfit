#!/usr/bin/env bash
# HIQL-aligned 全集正式实验:act_feat × lerobot × base_policy。
# coffee 用本地 LeRobot 数据目录;ACT base 默认用本地 best,也可由 COFFEE_ACT_BASE 覆盖。
# gc_value 阶段显式对齐参考 HIQL V: concat rep, hidden=512, layers=3, LN, hiql loss/mask, geometric future。
# 用法:
#   CUDA_VISIBLE_DEVICES=<卡> bash run_act_feat_lerobot_hiql_aligned_bp_bc01.sh coffee
#   CUDA_VISIBLE_DEVICES=<卡> bash run_act_feat_lerobot_hiql_aligned_bp_bc01.sh <pouring|lifttray|piece>
set -euo pipefail
KEY="${1:?用法: CUDA_VISIBLE_DEVICES=N bash $0 coffee|pouring|lifttray|piece}"
case "$KEY" in
  coffee)   TASK=TwoArmCoffee; SUB=dexmg-two-arm-coffee; BASE="${COFFEE_ACT_BASE:-/mnt/mnt/data/resfit/resfit/out/coffee/best}" ;;
  pouring)  TASK=TwoArmPouring;  SUB=dexmg-two-arm-pouring;   BASE=/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2 ;;
  lifttray) TASK=TwoArmLiftTray; SUB=dexmg-two-arm-lift-tray; BASE=/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_e0o0sckj_best:v4 ;;
  piece)    TASK=TwoArmThreePieceAssembly; SUB=dexmg-two-arm-three-piece-assembly; BASE=/mnt/mnt/data/resfit/resfit/out/piecce/best ;;
  boxcleanup) TASK=TwoArmBoxCleanup; SUB=dexmg-two-arm-box-cleanup; BASE=/mnt/mnt/data/resfit/resfit/out/boxcleanup/best ;;
  square)   TASK=Square; SUB=robomimic-mh-square-image; BASE=/mnt/mnt/data/resfit/resfit/out/square/best ;;  # 单臂 robomimic;整跑前须先过 stage1/stage3 smoke
  *) echo "未知 task: $KEY (只支持 coffee|pouring|lifttray|piece)"; exit 2 ;;
esac

export PYTHONPATH=/mnt/mnt/data/resfit
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /mnt/mnt/data/resfit
source resfit/lerobot/shell/torchcodec_env.sh

ROOT=/mnt/mnt/data/resfit/resfit/dataset/ankile/$SUB
[ "$KEY" = piece ] && ROOT=/root/.cache/huggingface/lerobot/ankile/$SUB
DS=ankile/$SUB
DUMMY=/tmp/UNUSED_dummy.hdf5
CACHE=outputs_chunk/${KEY}_act_feat_lerobot_hiql_aligned.npz
GC=outputs_chunk/${KEY}_gc_value_actfeat_lerobot_hiql_aligned.pt
HA=outputs_chunk/${KEY}_high_actor_actfeat_lerobot_hiql_aligned.pt
OFFCACHE=outputs_chunk/${KEY}_actfeat_lerobot_hiql_aligned_bp_offcache
OUT=outputs_chunk/${KEY}_actfeat_lerobot_hiql_aligned_bp_bc01
RUN="conda run -n residual --no-capture-output python -u"
gate(){ [ -s "$1" ] || { echo "[driver] FATAL 产物缺失: $1 (上一步失败,停)"; exit 4; }; }

echo "[driver] START $KEY ($TASK) lerobot HIQL-aligned 路 GPU=${CUDA_VISIBLE_DEVICES:-?} $(date '+%F %T')"
echo "[driver] ROOT=$ROOT"
echo "[driver] BASE=$BASE"

echo "[driver] 1) gc_value HIQL-aligned 全集(首跑建 act_feat 缓存,慢)... $(date '+%T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_hiql_gc_value \
  --hdf5 "$DUMMY" --state_mode act_feat --data_source lerobot --lerobot_root "$ROOT" \
  --act_base_ckpt "$BASE" --dataset "$DS" \
  --act_feat_cache "$CACHE" --output "$GC" \
  --rep_dim 10 \
  --value_rep_mode concat \
  --value_hidden 512 \
  --value_layers 3 \
  --use_layer_norm 1 \
  --value_loss_mode hiql \
  --value_mask_mode hiql \
  --goal_future_mode geometric
gate "$CACHE"; gate "$GC"; echo "[driver] 1) gc_value DONE $(date '+%T')"

echo "[driver] 2) high_actor(命中 hiql_aligned 缓存)... $(date '+%T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
  --hdf5 "$DUMMY" --state_mode act_feat --data_source lerobot --lerobot_root "$ROOT" \
  --act_base_ckpt "$BASE" --dataset "$DS" \
  --act_feat_cache "$CACHE" --gc_value_ckpt "$GC" --output "$HA"
gate "$HA"; echo "[driver] 2) high_actor DONE $(date '+%T')"

echo "[driver] 3) 主训 500k(base_policy,对齐 cansort 主训参数,no-stage)... $(date '+%T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task "$TASK" --base_wandb_id "$BASE" --dataset "$DS" \
  --data_source lerobot --lerobot_root "$ROOT" \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw \
  --reward_shaping none \
  --offline_fraction 0.5 --offline_base_mode base_policy --demo_bc_coef 0.1 \
  --offline_buffer_cache "$OFFCACHE" \
  --subgoal_conditioned --gc_value_ckpt "$GC" --high_actor_ckpt "$HA" --act_feat_cache "$CACHE" \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual --wandb_name "${KEY}_actfeat_lerobot_hiql_aligned_bp_bc01" \
  --output_dir "$OUT"
echo "[driver] ALL DONE $KEY $(date '+%F %T')  out=$OUT"
