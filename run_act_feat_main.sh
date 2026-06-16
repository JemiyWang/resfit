#!/usr/bin/env bash
# act_feat 主训:等配置 three_piece_aligned_bp_bc01,仅 value 输入换 act_feat。
#   gc/high → act_feat 版 + 加 --act_feat_cache + 去掉 --subgoal_state30_cache(act_feat 不用)。
#   base_wandb_id 与 act_base_ckpt 同一 ACT(resfit/out/piecce/best)→ 同源、不触发警告。
# 在线 rollout 需 env 渲染 → EGL。用法: bash run_act_feat_main.sh <gpu>
# 由 launcher 以 nohup 完全分离启动(关对话不杀进程)。
set -u
GPU=${1:?usage: run_act_feat_main.sh <gpu>}
export PATH="/root/miniconda3/bin:$PATH"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1
export PYTHONPATH=/mnt/mnt/data/resfit
cd /mnt/mnt/data/resfit || exit 3
PY="conda run -n residual --no-capture-output python -u"

echo "[actfeat-main] START $(date '+%F %T')  GPU=$CUDA_VISIBLE_DEVICES"
$PY -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreePieceAssembly --base_wandb_id resfit/out/piecce/best \
  --dataset ankile/dexmg-two-arm-three-piece-assembly \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw --stage_balanced --reward_shaping none \
  --offline_dataset_path resfit/dataset/two_arm_three_piece_assembly.hdf5 \
  --offline_fraction 0.5 --offline_stage_cache outputs_chunk/three_piece_stages.npz \
  --offline_base_mode base_policy --demo_bc_coef 0.1 \
  --offline_buffer_cache outputs_chunk/actfeat_bp_offcache \
  --subgoal_conditioned \
  --gc_value_ckpt outputs_chunk/three_piece_gc_value_actfeat.pt \
  --high_actor_ckpt outputs_chunk/three_piece_high_actor_actfeat.pt \
  --act_feat_cache outputs_chunk/three_piece_act_feat.npz \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual --wandb_name three_piece_actfeat_bp_bc01 \
  --output_dir outputs_chunk/three_piece_actfeat_bp_bc01
echo "[actfeat-main] EXIT rc=$? $(date '+%F %T')"
