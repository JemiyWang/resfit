#!/usr/bin/env bash
# pouring 消融:对齐 run_pouring_pothiql_actfeat_joint.sh,去 subgoal + bc(potential 保留)。
#   删掉:1) --subgoal_conditioned(子目标注入)→ 连带删 gc_value/high_actor/subgoal_way_steps
#          2) joint 在线微调(--online_finetune_value/high_actor)→ **必删**:断言"在线微调需 --subgoal_conditioned"(train_chunk_residual.py:62)
#          3) bc loss(--demo_bc_coef 0.1)→ 撤掉(默认 0.0,BC 分支门控跳过)
#   **保留 potential hiql(act_feat 版)**:--reward_shaping potential --potential_source hiql --hiql_value_ckpt pouring_value_actfeat_hdf5.pt
#          → 因势函数是 act_feat,**必须保留 --act_feat_cache**(断言 _validate_actfeat_potential_cache/L515;_needs_act_feat_cache 对 act_feat potential 返回 True,与 subgoal 无关)。
#   其余一致:chunk1/queue/base_n10/action_scale0.05/actor_lr1e-6/actor raw/offline_fraction0.5/base_policy 锚/500k。
#   等价:base_policy 残差 RL + potential(hiql,act_feat)整形、无 subgoal/无 joint/无 bc 的"纯势函数臂"。
#   OFFCACHE 换新路径:签名含 subgoal(true→false)→ 旧 offcache 失效,必首建(~41G,~80min base_policy CPU forward)。
# 用法: setsid bash run_pouring_pothiql_nosubgoal_nojoint_nobc.sh <gpu> > pouring_pothiql_nosubgoal_nojoint_nobc.log 2>&1 < /dev/null &
set -u
GPU=${1:-3}
export PATH="/root/miniconda3/bin:$PATH"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1
export PYTHONPATH=/mnt/mnt/data/resfit
cd /mnt/mnt/data/resfit || exit 3
RUN="conda run -n residual --no-capture-output python -u"

OFFCACHE=outputs_chunk/pouring_actfeat_hdf5_bp_as005_pothiql_nosubgoal_nojoint_nobc_offcache   # 新路径→必重建(签名 subgoal=false)
BASE="/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2"
gate(){ [ -s "$1" ] || { echo "[pothiql_nosubgoal_nojoint_nobc] FATAL 复用产物缺失: $1"; exit 4; }; }
gate "outputs_chunk/pouring_value_actfeat_hdf5.pt"
gate "outputs_chunk/pouring_act_feat_hdf5.npz"
gate "$BASE/policy/model.safetensors"
echo "[pothiql_nosubgoal_nojoint_nobc] START pouring potential-only (no subgoal, no joint, no bc)  GPU=$CUDA_VISIBLE_DEVICES  $(date '+%F %T')  (offcache 首建)"

$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmPouring --base_wandb_id "$BASE" \
  --dataset ankile/dexmg-two-arm-pouring \
  --offline_dataset_path resfit/dataset/two_arm_pouring.hdf5 \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw \
  --reward_shaping potential --potential_source hiql \
  --hiql_value_ckpt outputs_chunk/pouring_value_actfeat_hdf5.pt \
  --act_feat_cache outputs_chunk/pouring_act_feat_hdf5.npz \
  --offline_fraction 0.5 --offline_base_mode base_policy \
  --offline_buffer_cache "$OFFCACHE" \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual \
  --wandb_name pouring_actfeat_hdf5_bp_as005_pothiql_nosubgoal_nojoint_nobc \
  --output_dir outputs_chunk/pouring_actfeat_hdf5_bp_as005_pothiql_nosubgoal_nojoint_nobc
echo "[pothiql_nosubgoal_nojoint_nobc] ALL DONE $(date '+%F %T')"
