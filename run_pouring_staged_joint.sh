#!/usr/bin/env bash
# pouring staged + 联合训练(staged_joint)。
#   逐字对齐 pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_pothiql_joint_rerun0703,仅改:
#     ① reward: potential(单状态 hiql V) → staged(stage 整数净加 bonus;不需要 value.pt)
#     ② 删势函数来源与 value ckpt 相关 flags;加 --stage_reward_bonus 1.0 --stage_balanced
#        --offline_stage_cache outputs_chunk/two_arm_pouring_stages.npz(不存在→首跑 sim-replay 生成 5 段)
#     ③ 新 OFFCACHE(staged 签名→重建)/ 新 wandb_name / output_dir
#   其余 100% 一致:chunk1 / queue / base_n10 / action_scale0.05 / actor_lr1e-6 / actor raw /
#     offline_fraction0.5 / base_policy锚 + bc0.1 / subgoal sg15(gc_value+high_actor+act_feat)/
#     joint online_finetune / 500k。
#   ⚠ 上 500k 前先 smoke 掉 stages npz 在 GR1+hdf5 上的首次生成(见 verify_pouring_stages.py)。
# 用法: setsid bash run_pouring_staged_joint.sh <gpu> > pouring_staged_joint.log 2>&1 < /dev/null &
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

OFFCACHE=outputs_chunk/pouring_actfeat_hdf5_bp_hiqlv512_sg15_as005_staged_joint_offcache   # 新(staged 签名→重建)
STAGES=outputs_chunk/two_arm_pouring_stages.npz
BASE="/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2"
gate(){ [ -s "$1" ] || { echo "[pouring_staged_joint] FATAL 复用产物缺失: $1"; exit 4; }; }
gate "outputs_chunk/pouring_gc_value_actfeat_hdf5_hiqlv512.pt"
gate "outputs_chunk/pouring_high_actor_actfeat_hdf5_hiqlv512.pt"
gate "outputs_chunk/pouring_act_feat_hdf5.npz"
gate "$BASE/policy/model.safetensors"
[ -e "$STAGES" ] && echo "[pouring_staged_joint] 注意:$STAGES 已存在,将直接读(确认是5段);若想重生成请先删它" \
                  || echo "[pouring_staged_joint] $STAGES 不存在 → 首次将 sim-replay 生成 5 段(GR1,较慢,一次性)"
echo "[pouring_staged_joint] START pouring staged+joint  GPU=$CUDA_VISIBLE_DEVICES  $(date '+%F %T')  (offcache 首建)"

$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmPouring --base_wandb_id "$BASE" \
  --dataset ankile/dexmg-two-arm-pouring \
  --offline_dataset_path resfit/dataset/two_arm_pouring.hdf5 \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw --stage_balanced \
  --reward_shaping staged --stage_reward_bonus 1.0 \
  --offline_stage_cache "$STAGES" \
  --offline_fraction 0.5 --offline_base_mode base_policy --demo_bc_coef 0.1 \
  --offline_buffer_cache "$OFFCACHE" \
  --subgoal_conditioned \
  --gc_value_ckpt outputs_chunk/pouring_gc_value_actfeat_hdf5_hiqlv512.pt \
  --high_actor_ckpt outputs_chunk/pouring_high_actor_actfeat_hdf5_hiqlv512.pt \
  --act_feat_cache outputs_chunk/pouring_act_feat_hdf5.npz \
  --subgoal_way_steps 15 --online_finetune_value --online_finetune_high_actor \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual \
  --wandb_name pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_staged_joint \
  --output_dir outputs_chunk/pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_staged_joint
echo "[pouring_staged_joint] ALL DONE $(date '+%F %T')"
