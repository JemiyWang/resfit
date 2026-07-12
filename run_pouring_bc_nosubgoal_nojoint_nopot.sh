#!/usr/bin/env bash
# pouring 消融:对齐 run_pouring_pothiql_actfeat_joint.sh,去 subgoal + potential(bc 保留)。
#   删掉:1) potential hiql 整形(--reward_shaping potential --potential_source hiql --hiql_value_ckpt ...)→ --reward_shaping none
#          2) --subgoal_conditioned(子目标注入)→ 连带删 gc_value/high_actor/act_feat_cache/subgoal_way_steps
#          3) joint 在线微调(--online_finetune_value/high_actor)→ **必删**:代码断言"在线微调需 --subgoal_conditioned"(train_chunk_residual.py:62),去 subgoal 不删 joint 会起手 assert 崩。
#   **保留 bc**(用户未要求去 bc):--demo_bc_coef 0.1。
#   其余一致:chunk1/queue/base_n10/action_scale0.05/actor_lr1e-6/actor raw/offline_fraction0.5/base_policy 锚/500k。
#   等价:这就是 base_policy 残差 RL + BC 锚 + 离线混采、稀疏奖励、无 subgoal/无 joint/无 shaping 的基线臂。
#   OFFCACHE 换新路径:签名含 reward_shaping+subgoal,potential→none & subgoal true→false → 旧 offcache 失效,必首建(~41G,~80min base_policy CPU forward)。
# 用法: setsid bash run_pouring_bc_nosubgoal_nojoint_nopot.sh <gpu> > pouring_bc_nosubgoal_nojoint_nopot.log 2>&1 < /dev/null &
set -u
GPU=${1:-2}
export PATH="/root/miniconda3/bin:$PATH"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1
export PYTHONPATH=/mnt/mnt/data/resfit
cd /mnt/mnt/data/resfit || exit 3
RUN="conda run -n residual --no-capture-output python -u"

OFFCACHE=outputs_chunk/pouring_hdf5_bp_bc01_as005_nosubgoal_nojoint_nopot_offcache   # 新路径→必重建(签名 reward_shaping=none & subgoal=false)
BASE="/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2"
gate(){ [ -s "$1" ] || { echo "[bc_nosubgoal_nojoint_nopot] FATAL 复用产物缺失: $1"; exit 4; }; }
gate "$BASE/policy/model.safetensors"
echo "[bc_nosubgoal_nojoint_nopot] START pouring bc baseline (no subgoal, no joint, no potential)  GPU=$CUDA_VISIBLE_DEVICES  $(date '+%F %T')  (offcache 首建)"

$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmPouring --base_wandb_id "$BASE" \
  --dataset ankile/dexmg-two-arm-pouring \
  --offline_dataset_path resfit/dataset/two_arm_pouring.hdf5 \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw \
  --reward_shaping none \
  --offline_fraction 0.5 --offline_base_mode base_policy --demo_bc_coef 0.1 \
  --offline_buffer_cache "$OFFCACHE" \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual \
  --wandb_name pouring_hdf5_bp_bc01_as005_nosubgoal_nojoint_nopot \
  --output_dir outputs_chunk/pouring_hdf5_bp_bc01_as005_nosubgoal_nojoint_nopot
echo "[bc_nosubgoal_nojoint_nopot] ALL DONE $(date '+%F %T')"
