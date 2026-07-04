#!/usr/bin/env bash
# pouring pothiql_actfeat_joint A/B run(act_feat 势函数,与 rerun0703 逐字对照)。
#   逐字复现 run_pouring_pothiql_joint_rerun0703.sh,仅改 4 处:
#     1) OFFCACHE 换新路径(势函数输入变了→离线 reward 变→旧 offcache 失效,必重建,非只读复用)
#     2) --hiql_value_ckpt 换 Task 1 产出的 act_feat 版 value(pouring_value_actfeat_hdf5.pt)
#     3) --wandb_name 后缀 _pothiql_joint_rerun0703 → _pothiql_actfeat_joint
#     4) --output_dir 同步改名
#   训练超参 100% 一致:pothiql(单状态 V,--potential_source hiql)/
#     chunk1 / queue / base_n10 / action_scale0.05 / actor_lr1e-6 / actor raw / offline_fraction0.5 /
#     base_policy 锚 + bc0.1 / subgoal sg15(gc_value+high_actor)/ joint online_finetune / 500k。
#   offcache 首建(签名含新势函数,~41G,~80min base_policy CPU forward),非复用。
# 用法: setsid bash run_pouring_pothiql_actfeat_joint.sh <gpu> > pouring_pothiql_actfeat_joint.log 2>&1 < /dev/null &
set -u
GPU=${1:-4}
export PATH="/root/miniconda3/bin:$PATH"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1
export PYTHONPATH=/mnt/mnt/data/resfit
cd /mnt/mnt/data/resfit || exit 3
RUN="conda run -n residual --no-capture-output python -u"

OFFCACHE=outputs_chunk/pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_pothiql_actfeat_joint_offcache   # 新路径→必重建(首建,非只读复用)
BASE="/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2"
gate(){ [ -s "$1" ] || { echo "[pothiql_actfeat_joint] FATAL 复用产物缺失: $1"; exit 4; }; }
gate "outputs_chunk/pouring_value_actfeat_hdf5.pt"
gate "outputs_chunk/pouring_gc_value_actfeat_hdf5_hiqlv512.pt"
gate "outputs_chunk/pouring_high_actor_actfeat_hdf5_hiqlv512.pt"
gate "outputs_chunk/pouring_act_feat_hdf5.npz"
gate "$BASE/policy/model.safetensors"
echo "[pothiql_actfeat_joint] START pouring pothiql+joint act_feat A/B  GPU=$CUDA_VISIBLE_DEVICES  $(date '+%F %T')  (offcache 首建)"

$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmPouring --base_wandb_id "$BASE" \
  --dataset ankile/dexmg-two-arm-pouring \
  --offline_dataset_path resfit/dataset/two_arm_pouring.hdf5 \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw \
  --reward_shaping potential --potential_source hiql \
  --hiql_value_ckpt outputs_chunk/pouring_value_actfeat_hdf5.pt \
  --offline_fraction 0.5 --offline_base_mode base_policy --demo_bc_coef 0.1 \
  --offline_buffer_cache "$OFFCACHE" \
  --subgoal_conditioned \
  --gc_value_ckpt outputs_chunk/pouring_gc_value_actfeat_hdf5_hiqlv512.pt \
  --high_actor_ckpt outputs_chunk/pouring_high_actor_actfeat_hdf5_hiqlv512.pt \
  --act_feat_cache outputs_chunk/pouring_act_feat_hdf5.npz \
  --subgoal_way_steps 15 --online_finetune_value --online_finetune_high_actor \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual \
  --wandb_name pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_pothiql_actfeat_joint \
  --output_dir outputs_chunk/pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_pothiql_actfeat_joint
echo "[pothiql_actfeat_joint] ALL DONE $(date '+%F %T')"
