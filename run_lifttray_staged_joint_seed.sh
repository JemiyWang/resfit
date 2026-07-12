#!/bin/bash
# lifttray staged + 联合训练，多 seed 版：逐字对齐 run_lifttray_staged_joint.sh，
# 仅把 seed / gpu / wandb_name / output_dir 参数化(原脚本无 --seed 走默认0、name无后缀)。
# offcache / gc_value / high_actor / act_feat 均与 seed 无关，3 个 seed 只读共享不重建。
# 用法: setsid bash run_lifttray_staged_joint_seed.sh <gpu> <seed> > lifttray_staged_joint_seed<seed>.log 2>&1 < /dev/null &
set -euo pipefail
GPU=${1:?用法: run_lifttray_staged_joint_seed.sh <gpu> <seed>}
SEED=${2:?用法: run_lifttray_staged_joint_seed.sh <gpu> <seed>}
export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1
export PYTHONPATH=/mnt/mnt/data/resfit
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
cd /mnt/mnt/data/resfit
PY=/mnt/mnt/data/envs/residual/bin/python

NAME=lifttray_actfeat_hdf5_bp_bc01_hiqlv512_sg15_staged_joint_seed${SEED}
GCV=outputs_chunk/lifttray_gc_value_actfeat_hdf5_hiqlv512.pt
HIA=outputs_chunk/lifttray_high_actor_actfeat_hdf5_hiqlv512.pt
CACHE=outputs_chunk/lifttray_act_feat_hdf5.npz
gate(){ [ -s "$1" ] || { echo "[lifttray_staged_seed${SEED}] FATAL 复用产物缺失: $1"; exit 4; }; }
echo "[lifttray_staged_seed${SEED}] gate 复用产物(gc_value/high_actor/act_feat)..."
gate "$GCV"; gate "$HIA"; gate "$CACHE"
echo "[lifttray_staged_seed${SEED}] START GPU=$GPU SEED=$SEED  $(date '+%F %T')"

exec $PY -u -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmLiftTray \
  --base_wandb_id /mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_e0o0sckj_best:v4 \
  --dataset ankile/dexmg-two-arm-lift-tray \
  --offline_dataset_path resfit/dataset/two_arm_lift_tray.hdf5 \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw \
  --reward_shaping staged --stage_reward_bonus 1.0 --stage_balanced \
  --offline_fraction 0.5 --offline_stage_cache outputs_chunk/two_arm_lift_tray_stages.npz \
  --offline_base_mode base_policy --demo_bc_coef 0.1 \
  --offline_buffer_cache outputs_chunk/lifttray_actfeat_hdf5_bp_hiqlv512_sg15_staged_offcache \
  --subgoal_conditioned --gc_value_ckpt "$GCV" --high_actor_ckpt "$HIA" --act_feat_cache "$CACHE" \
  --subgoal_way_steps 15 \
  --online_finetune_value --online_finetune_high_actor \
  --seed ${SEED} \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual \
  --wandb_name ${NAME} \
  --output_dir outputs_chunk/${NAME}
