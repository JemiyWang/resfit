#!/bin/bash
# lifttray pothiql **act_feat 版**:镜像 run_lifttray_pothiql_joint_rerun0629.sh,仅改 3 处:
#   1) --hiql_value_ckpt: lifttray_value_hdf5.pt(eef,38) → lifttray_value_actfeat_hdf5.pt(act_feat,550)  ← 势函数 V 输入 eef→act_feat
#   2) --offline_buffer_cache 换新路径(势函数输入变→离线 reward 变→旧 offcache 失效,必首建)
#   3) --wandb_name / --output_dir 后缀 _pothiql_joint_rerun0629 → _pothiql_actfeat_joint
# 其余 100% 一致:bcfixed0.1 / subgoal sg15(gc_value+high_actor act_feat)/ joint online_finetune /
#   chunk1 / queue / base_n10 / action_scale0.05 / actor_lr1e-6 / actor raw / offline_fraction0.5 / base_policy 锚 / 500k。
# 用法: setsid bash run_lifttray_pothiql_actfeat_joint.sh <gpu> > lifttray_pothiql_actfeat_joint.log 2>&1 < /dev/null &
set -euo pipefail
GPU=${1:-4}
export CUDA_VISIBLE_DEVICES="$GPU"
export MUJOCO_GL=egl
export HF_HUB_OFFLINE=1
export PYTHONPATH=/mnt/mnt/data/resfit
export LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64
cd /mnt/mnt/data/resfit
VAL=outputs_chunk/lifttray_value_actfeat_hdf5.pt
[ -s "$VAL" ] || { echo "[pothiql_actfeat_joint] FATAL act_feat 势函数 value 未就绪: $VAL (先跑 run_lifttray_value_actfeat.sh)"; exit 4; }
echo "[pothiql_actfeat_joint] START lifttray pothiql act_feat V + joint  GPU=$CUDA_VISIBLE_DEVICES  $(date '+%F %T')  (offcache 首建)"
exec /mnt/mnt/data/envs/residual/bin/python -u -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmLiftTray \
  --base_wandb_id /mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_e0o0sckj_best:v4 \
  --dataset ankile/dexmg-two-arm-lift-tray \
  --offline_dataset_path resfit/dataset/two_arm_lift_tray.hdf5 \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw \
  --reward_shaping potential --potential_source hiql \
  --hiql_value_ckpt outputs_chunk/lifttray_value_actfeat_hdf5.pt \
  --offline_fraction 0.5 --offline_base_mode base_policy --demo_bc_coef 0.1 \
  --offline_buffer_cache outputs_chunk/lifttray_actfeat_hdf5_bp_hiqlv512_sg15_pothiql_actfeat_joint_offcache \
  --subgoal_conditioned \
  --gc_value_ckpt outputs_chunk/lifttray_gc_value_actfeat_hdf5_hiqlv512.pt \
  --high_actor_ckpt outputs_chunk/lifttray_high_actor_actfeat_hdf5_hiqlv512.pt \
  --act_feat_cache outputs_chunk/lifttray_act_feat_hdf5.npz \
  --subgoal_way_steps 15 --online_finetune_value --online_finetune_high_actor \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual \
  --wandb_name lifttray_actfeat_hdf5_bp_bcfixed01_hiqlv512_sg15_pothiql_actfeat_joint \
  --output_dir outputs_chunk/lifttray_actfeat_hdf5_bp_bcfixed01_hiqlv512_sg15_pothiql_actfeat_joint
