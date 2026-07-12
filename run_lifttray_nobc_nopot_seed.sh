#!/bin/bash
# lift 版 nobc_nopot(纯 subgoal:无 bc loss、reward_shaping none、有 subgoal+joint),多 seed 版。
# 对齐 pouring_actfeat_hdf5_bp_hiqlv512_sg15_as005_subgoal_joint_nobc_nopot,任务换 TwoArmLiftTray:
#   - 非任务超参逐字照搬 pouring 模板(reward none / 无 demo_bc / subgoal_conditioned / joint /
#     offline_fraction 0.5 / offline_base_mode base_policy / subgoal_way_steps 15 / chunk1 / as0.05 ...);
#   - 任务相关换 lift(task/base e0o0sckj:v4/dataset/hdf5/gc_value/high_actor/act_feat/offcache/name)。
# offcache 与 seed 无关 → 多 seed 共享同一个,第1个 seed 建(~62G, base_policy 重跑),后续秒级复用。
# 用法: setsid bash run_lifttray_nobc_nopot_seed.sh <gpu> <seed> > lifttray_nobc_nopot_seed<seed>.log 2>&1 < /dev/null &
set -euo pipefail
GPU=${1:?用法: run_lifttray_nobc_nopot_seed.sh <gpu> <seed>}
SEED=${2:?用法: run_lifttray_nobc_nopot_seed.sh <gpu> <seed>}
export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1
export PYTHONPATH=/mnt/mnt/data/resfit
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
cd /mnt/mnt/data/resfit
PY=/mnt/mnt/data/envs/residual/bin/python

NAME=lifttray_actfeat_hdf5_bp_hiqlv512_sg15_subgoal_joint_nobc_nopot_seed${SEED}
GCV=outputs_chunk/lifttray_gc_value_actfeat_hdf5_hiqlv512.pt
HIA=outputs_chunk/lifttray_high_actor_actfeat_hdf5_hiqlv512.pt
CACHE=outputs_chunk/lifttray_act_feat_hdf5.npz
gate(){ [ -s "$1" ] || { echo "[lift_nobc_nopot_seed${SEED}] FATAL 复用产物缺失: $1"; exit 4; }; }
echo "[lift_nobc_nopot_seed${SEED}] gate 复用产物(gc_value/high_actor/act_feat)..."
gate "$GCV"; gate "$HIA"; gate "$CACHE"
echo "[lift_nobc_nopot_seed${SEED}] START GPU=$GPU SEED=$SEED  $(date '+%F %T')"

exec $PY -u -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmLiftTray \
  --base_wandb_id /mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_e0o0sckj_best:v4 \
  --dataset ankile/dexmg-two-arm-lift-tray \
  --offline_dataset_path resfit/dataset/two_arm_lift_tray.hdf5 \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw \
  --reward_shaping none \
  --offline_fraction 0.5 --offline_base_mode base_policy \
  --offline_buffer_cache outputs_chunk/lifttray_actfeat_hdf5_bp_hiqlv512_sg15_subgoal_joint_nobc_nopot_offcache \
  --subgoal_conditioned --gc_value_ckpt "$GCV" --high_actor_ckpt "$HIA" --act_feat_cache "$CACHE" \
  --subgoal_way_steps 15 \
  --online_finetune_value --online_finetune_high_actor \
  --seed ${SEED} \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual \
  --wandb_name ${NAME} \
  --output_dir outputs_chunk/${NAME}
