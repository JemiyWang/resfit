#!/usr/bin/env bash
# bp_bc01 + BC衰减 消融。与 three_piece_aligned_bp_bc01 配置完全一致,唯一差异:
#   加 --bc_coef_final 0.01 (demo-BC 从 0.1 线性衰减到 0.01,区间 0→500k)。
# 两件套仍用 aligned(不换),故 subgoal z 与 bp_bc01 相同,且 bc_coef_final 不入 offcache 签名
#   → 直接命中复用 outputs_chunk/aligned_bp_offcache(秒读,不重建,只读不影响 bc0/bc01)。
# detached,卡 1,限线程防超订。
set -u
export PATH="/root/miniconda3/bin:$PATH"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH=/mnt/mnt/data/resfit
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1
cd /mnt/mnt/data/resfit || exit 3
PY=/mnt/mnt/data/envs/residual/bin/python

echo "[bc01decay] START $(date '+%F %T')  CUDA=$CUDA_VISIBLE_DEVICES"
$PY -u -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreePieceAssembly \
  --base_wandb_id /mnt/mnt/data/resfit/resfit/out/piecce/best \
  --dataset ankile/dexmg-two-arm-three-piece-assembly \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw --stage_balanced --reward_shaping none \
  --offline_dataset_path resfit/dataset/two_arm_three_piece_assembly.hdf5 \
  --offline_fraction 0.5 --offline_stage_cache outputs_chunk/three_piece_stages.npz \
  --offline_base_mode base_policy --demo_bc_coef 0.1 --bc_coef_final 0.01 \
  --offline_buffer_cache outputs_chunk/aligned_bp_offcache \
  --subgoal_conditioned \
  --gc_value_ckpt outputs_chunk/three_piece_gc_value_aligned.pt \
  --high_actor_ckpt outputs_chunk/three_piece_high_actor_aligned.pt \
  --subgoal_state30_cache outputs_chunk/three_piece_state30.npz \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual --wandb_name three_piece_aligned_bp_bc01_decay \
  --output_dir outputs_chunk/three_piece_aligned_bp_bc01_decay \
  > outputs_chunk/three_piece_aligned_bp_bc01_decay.log 2>&1
echo "[bc01decay] EXIT rc=$? $(date '+%F %T')"
