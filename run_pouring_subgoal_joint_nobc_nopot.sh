#!/usr/bin/env bash
# pouring subgoal+joint 消融:对齐 run_pouring_pothiql_actfeat_joint.sh,仅做 2 处减法。
#   删掉:1) potential hiql 整形(--reward_shaping potential --potential_source hiql --hiql_value_ckpt ...)
#              → 改成 --reward_shaping none(无任何 reward shaping)
#          2) bc loss(--demo_bc_coef 0.1)→ 撤掉(argparse 默认 0.0,BC 分支被 demo_bc_coef>0 门控,不计)
#   其余 100% 一致:act_feat / hdf5 / base_policy 锚(offline_fraction0.5,无 bc)/
#     chunk1 / queue / base_n10 / action_scale0.05 / actor_lr1e-6 / actor raw /
#     subgoal sg15(gc_value+high_actor)/ joint online_finetune(V+high_actor 微调 subgoal 机制,与 potential 无关)/ 500k。
#   OFFCACHE 换新路径:签名含 reward_shaping(potential→none),旧 offcache 失效 → 首建(~41G,~80min base_policy CPU forward),非复用。
# 用法: setsid bash run_pouring_subgoal_joint_nobc_nopot.sh <gpu> > pouring_subgoal_joint_nobc_nopot.log 2>&1 < /dev/null &
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

OFFCACHE=outputs_chunk/pouring_actfeat_hdf5_bp_hiqlv512_sg15_as005_subgoal_joint_nobc_nopot_offcache   # 新路径→必重建(签名 reward_shaping=none)
BASE="/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2"
gate(){ [ -s "$1" ] || { echo "[subgoal_joint_nobc_nopot] FATAL 复用产物缺失: $1"; exit 4; }; }
gate "outputs_chunk/pouring_gc_value_actfeat_hdf5_hiqlv512.pt"
gate "outputs_chunk/pouring_high_actor_actfeat_hdf5_hiqlv512.pt"
gate "outputs_chunk/pouring_act_feat_hdf5.npz"
gate "$BASE/policy/model.safetensors"
echo "[subgoal_joint_nobc_nopot] START pouring subgoal+joint (no bc, no potential)  GPU=$CUDA_VISIBLE_DEVICES  $(date '+%F %T')  (offcache 首建)"

$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmPouring --base_wandb_id "$BASE" \
  --dataset ankile/dexmg-two-arm-pouring \
  --offline_dataset_path resfit/dataset/two_arm_pouring.hdf5 \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw \
  --reward_shaping none \
  --offline_fraction 0.5 --offline_base_mode base_policy \
  --offline_buffer_cache "$OFFCACHE" \
  --subgoal_conditioned \
  --gc_value_ckpt outputs_chunk/pouring_gc_value_actfeat_hdf5_hiqlv512.pt \
  --high_actor_ckpt outputs_chunk/pouring_high_actor_actfeat_hdf5_hiqlv512.pt \
  --act_feat_cache outputs_chunk/pouring_act_feat_hdf5.npz \
  --subgoal_way_steps 15 --online_finetune_value --online_finetune_high_actor \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual \
  --wandb_name pouring_actfeat_hdf5_bp_hiqlv512_sg15_as005_subgoal_joint_nobc_nopot \
  --output_dir outputs_chunk/pouring_actfeat_hdf5_bp_hiqlv512_sg15_as005_subgoal_joint_nobc_nopot
echo "[subgoal_joint_nobc_nopot] ALL DONE $(date '+%F %T')"
