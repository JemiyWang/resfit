#!/bin/bash
# lifttray staged + 联合训练。参照 run_lifttray_pothiql_joint_rerun0629.sh,改用 staged reward:
#   ① 改 reward_shaping 为 staged,移除 hiql 相关 flags
#   ② 加 stage_reward_bonus 1.0 / stage_balanced / offline_stage_cache(不存在→首跑 sim-replay 自动生成)
#   ③ 新 offcache(staged 签名重建)+ 新 wandb_name/output_dir
# 其余逐字照搬 pothiql。用法: setsid bash run_lifttray_staged_joint.sh <gpu> > lifttray_staged_joint.log 2>&1 < /dev/null &
set -euo pipefail
GPU=${1:?用法: run_lifttray_staged_joint.sh <gpu>}
export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1
export PYTHONPATH=/mnt/mnt/data/resfit
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
cd /mnt/mnt/data/resfit
PY=/mnt/mnt/data/envs/residual/bin/python

GCV=outputs_chunk/lifttray_gc_value_actfeat_hdf5_hiqlv512.pt
HIA=outputs_chunk/lifttray_high_actor_actfeat_hdf5_hiqlv512.pt
CACHE=outputs_chunk/lifttray_act_feat_hdf5.npz
gate(){ [ -s "$1" ] || { echo "[lifttray_staged] FATAL 复用产物缺失: $1"; exit 4; }; }
echo "[lifttray_staged] gate 复用产物(gc_value/high_actor/act_feat)..."
gate "$GCV"; gate "$HIA"; gate "$CACHE"
echo "[lifttray_staged] START GPU=$GPU  $(date '+%F %T')"

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
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual \
  --wandb_name lifttray_actfeat_hdf5_bp_bc01_hiqlv512_sg15_staged_joint \
  --output_dir outputs_chunk/lifttray_actfeat_hdf5_bp_bc01_hiqlv512_sg15_staged_joint
