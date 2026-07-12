#!/usr/bin/env bash
# pouring pothiql_joint + OAC 探索(A/B 的 +OAC 臂)。
# 与 baseline pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_pothiql_joint_rerun0629 逐字一致,
# 仅:追加 --oac_explore --oac_beta_ub 2.0 --oac_delta 0.5;改 wandb_name/output_dir。
# offcache 只读复用 baseline 的(OAC 只改在线探索采样,不改离线 buffer 签名)。
# 用法: setsid bash run_pouring_pothiql_joint_oac.sh <gpu> > pouring_pothiql_joint_oac.log 2>&1 < /dev/null &
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

OFFCACHE=outputs_chunk/pouring_actfeat_hdf5_bp_hiqlv512_sg15_as005_pothiql_joint_offcache   # 只读复用 baseline
[ -s "$OFFCACHE/buffer_meta.json" ] || { echo "[pothiql_oac] FATAL offcache 未建完: $OFFCACHE"; exit 4; }
echo "[pothiql_oac] START GPU=$CUDA_VISIBLE_DEVICES  $(date '+%F %T')  (offcache 复用 baseline)"

$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmPouring --base_wandb_id /mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2 \
  --dataset ankile/dexmg-two-arm-pouring --offline_dataset_path resfit/dataset/two_arm_pouring.hdf5 \
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
  --oac_explore --oac_beta_ub 2.0 --oac_delta 0.5 \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual \
  --wandb_name pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_pothiql_joint_oac \
  --output_dir outputs_chunk/pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_as005_pothiql_joint_oac
echo "[pothiql_oac] DONE $(date '+%F %T')"
