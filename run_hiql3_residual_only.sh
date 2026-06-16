#!/usr/bin/env bash
# hiql3 residual run。两件套 *_hiql3.pt 已就绪(17:22/17:28),跳过重训,只跑 residual。
# 与 three_piece_aligned_bp_bc01 配置完全一致,只有三处差异:
#   1) 挂 hiql3 两件套(承载三新默认 mask=hiql / rep=goal_only / adv=mean)
#   2) --bc_coef_final 0.01  (demo-BC 系数从 0.1 线性衰减到 0.01,区间 0→500k)
#   3) 独立 offcache outputs_chunk/hiql3_bp_offcache
#      (subgoal z 由 hiql3 两件套算后烤进 buffer,内容与 aligned 不同,必须重建;
#       且不能覆盖 bc0/bc01 共用的 aligned_bp_offcache)
# detached,卡 7,限线程防超订。
set -u
export PATH="/root/miniconda3/bin:$PATH"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=7
export PYTHONPATH=/mnt/mnt/data/resfit
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1
cd /mnt/mnt/data/resfit || exit 3
PY=/mnt/mnt/data/envs/residual/bin/python

echo "[hiql3-run] START $(date '+%F %T')  CUDA=$CUDA_VISIBLE_DEVICES"
$PY -u -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreePieceAssembly \
  --base_wandb_id /mnt/mnt/data/resfit/resfit/out/piecce/best \
  --dataset ankile/dexmg-two-arm-three-piece-assembly \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw --stage_balanced --reward_shaping none \
  --offline_dataset_path resfit/dataset/two_arm_three_piece_assembly.hdf5 \
  --offline_fraction 0.5 --offline_stage_cache outputs_chunk/three_piece_stages.npz \
  --offline_base_mode base_policy --demo_bc_coef 0.1 --bc_coef_final 0.01 \
  --offline_buffer_cache outputs_chunk/hiql3_bp_offcache \
  --subgoal_conditioned \
  --gc_value_ckpt outputs_chunk/three_piece_gc_value_hiql3.pt \
  --high_actor_ckpt outputs_chunk/three_piece_high_actor_hiql3.pt \
  --subgoal_state30_cache outputs_chunk/three_piece_state30.npz \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual --wandb_name three_piece_hiql3_bp_bc01 \
  --output_dir outputs_chunk/three_piece_hiql3_bp_bc01 \
  > outputs_chunk/three_piece_hiql3_bp_bc01.log 2>&1
echo "[hiql3-run] EXIT rc=$? $(date '+%F %T')"
