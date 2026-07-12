#!/usr/bin/env bash
# pouring pothiql_joint 复现跑(rerun0703)。
#   逐字复现 pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_pothiql_joint_rerun0629,
#   参数取自该 run 的 wandb metadata(run-20260630_114318-u2na8cd8,与首启 u3a5v3cs 53 项逐字一致)。
#   唯一改动:output_dir/wandb_name 后缀 _rerun0629 → _rerun0703(避免覆盖 7-1 已跑完的 best.pt)。
#   训练超参 100% 一致:pothiql(单状态 V,--potential_source hiql + hiql_value_ckpt pouring_value_hdf5.pt)/
#     chunk1 / queue / base_n10 / action_scale0.05 / actor_lr1e-6 / actor raw / offline_fraction0.5 /
#     base_policy 锚 + bc0.1 / subgoal sg15(gc_value+high_actor)/ joint online_finetune / 500k。
#   offcache 复用现有 pouring_actfeat_hdf5_bp_hiqlv512_sg15_as005_pothiql_joint_offcache
#     (buffer_meta.json 签名与本参数逐位一致,n_transitions=337508 → 秒读,省 ~80min + 41G;只读不改)。
# 用法: setsid bash run_pouring_pothiql_joint_rerun0703.sh <gpu> > pouring_pothiql_joint_rerun0703.log 2>&1 < /dev/null &
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

OFFCACHE=outputs_chunk/pouring_actfeat_hdf5_bp_hiqlv512_sg15_as005_pothiql_joint_offcache   # 只读复用(签名一致)
BASE="/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2"
gate(){ [ -s "$1" ] || { echo "[pothiql_rerun0703] FATAL 复用产物缺失: $1"; exit 4; }; }
gate "$OFFCACHE/buffer_meta.json"
gate "outputs_chunk/pouring_value_hdf5.pt"
gate "outputs_chunk/pouring_gc_value_actfeat_hdf5_hiqlv512.pt"
gate "outputs_chunk/pouring_high_actor_actfeat_hdf5_hiqlv512.pt"
gate "outputs_chunk/pouring_act_feat_hdf5.npz"
gate "$BASE/policy/model.safetensors"
echo "[pothiql_rerun0703] START pouring pothiql+joint 复现  GPU=$CUDA_VISIBLE_DEVICES  $(date '+%F %T')  (offcache 复用)"

$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmPouring --base_wandb_id "$BASE" \
  --dataset ankile/dexmg-two-arm-pouring \
  --offline_dataset_path resfit/dataset/two_arm_pouring.hdf5 \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw \
  --reward_shaping potential --potential_source hiql \
  --hiql_value_ckpt outputs_chunk/pouring_value_hdf5.pt \
  --offline_fraction 0.5 --offline_base_mode base_policy --demo_bc_coef 0.1 \
  --offline_buffer_cache "$OFFCACHE" \
  --subgoal_conditioned \
  --gc_value_ckpt outputs_chunk/pouring_gc_value_actfeat_hdf5_hiqlv512.pt \
  --high_actor_ckpt outputs_chunk/pouring_high_actor_actfeat_hdf5_hiqlv512.pt \
  --act_feat_cache outputs_chunk/pouring_act_feat_hdf5.npz \
  --subgoal_way_steps 15 --online_finetune_value --online_finetune_high_actor \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual \
  --wandb_name pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_pothiql_joint_rerun0703 \
  --output_dir outputs_chunk/pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_pothiql_joint_rerun0703
echo "[pothiql_rerun0703] ALL DONE $(date '+%F %T')"
