#!/usr/bin/env bash
# pouring 消融:基于 GPU0 arm(run_pouring_subgoal_joint_nobc_nopot.sh),只把 bc 加回 0.1,其余不变。
#   = 参照 pothiql_actfeat_joint **减去 potential**:保留 bc0.1 + subgoal sg15 + joint,reward_shaping none(无 potential)。
#   相对 GPU0 arm 唯一差异:+ --demo_bc_coef 0.1。
#   注:bc 不进 offcache 签名 → 本 run offcache 内容与 GPU0 arm 逐位相同,但为避免两 live 训练进程共享同一 offcache(PER 优先级就地写风险)用独立新路径,独立重建。
#   其余一致:act_feat/hdf5/chunk1/queue/base_n10/action_scale0.05/actor_lr1e-6/actor raw/offline_fraction0.5/base_policy 锚/subgoal(gc_value+high_actor)/joint online_finetune/500k。
# 用法: setsid bash run_pouring_subgoal_joint_bc01_nopot.sh <gpu> > pouring_subgoal_joint_bc01_nopot.log 2>&1 < /dev/null &
set -u
GPU=${1:-7}
export PATH="/root/miniconda3/bin:$PATH"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1
export PYTHONPATH=/mnt/mnt/data/resfit
cd /mnt/mnt/data/resfit || exit 3
RUN="conda run -n residual --no-capture-output python -u"

OFFCACHE=outputs_chunk/pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_subgoal_joint_nopot_offcache   # 独立新路径(bc 不进签名但避免共享冲突)
BASE="/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2"
gate(){ [ -s "$1" ] || { echo "[subgoal_joint_bc01_nopot] FATAL 复用产物缺失: $1"; exit 4; }; }
gate "outputs_chunk/pouring_gc_value_actfeat_hdf5_hiqlv512.pt"
gate "outputs_chunk/pouring_high_actor_actfeat_hdf5_hiqlv512.pt"
gate "outputs_chunk/pouring_act_feat_hdf5.npz"
gate "$BASE/policy/model.safetensors"
echo "[subgoal_joint_bc01_nopot] START pouring subgoal+joint+bc0.1 (no potential)  GPU=$CUDA_VISIBLE_DEVICES  $(date '+%F %T')  (offcache 首建)"

$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmPouring --base_wandb_id "$BASE" \
  --dataset ankile/dexmg-two-arm-pouring \
  --offline_dataset_path resfit/dataset/two_arm_pouring.hdf5 \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw \
  --reward_shaping none \
  --offline_fraction 0.5 --offline_base_mode base_policy --demo_bc_coef 0.1 \
  --offline_buffer_cache "$OFFCACHE" \
  --subgoal_conditioned \
  --gc_value_ckpt outputs_chunk/pouring_gc_value_actfeat_hdf5_hiqlv512.pt \
  --high_actor_ckpt outputs_chunk/pouring_high_actor_actfeat_hdf5_hiqlv512.pt \
  --act_feat_cache outputs_chunk/pouring_act_feat_hdf5.npz \
  --subgoal_way_steps 15 --online_finetune_value --online_finetune_high_actor \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual \
  --wandb_name pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_subgoal_joint_nopot \
  --output_dir outputs_chunk/pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_subgoal_joint_nopot
echo "[subgoal_joint_bc01_nopot] ALL DONE $(date '+%F %T')"
