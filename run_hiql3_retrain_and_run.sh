#!/usr/bin/env bash
# HIQL 三新默认(value_mask_mode=hiql / value_rep_mode=goal_only / adv_agg=mean)生效:
# 纯默认重训两件套(=aligned 其余配置 geom/LN/hiql/clamp/p0.3 + 三新默认),再起 residual run
# (照搬 three_piece_aligned_bp_bc01 的金标准,只换两件套为 hiql3、复用 offcache、新 run 名)。
# detached(关对话不被杀)。全程卡 2,限线程防超订。
set -u
export PATH="/root/miniconda3/bin:$PATH"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=/mnt/mnt/data/resfit
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1
cd /mnt/mnt/data/resfit || exit 3
PY=/mnt/mnt/data/envs/residual/bin/python

HDF5=resfit/dataset/two_arm_three_piece_assembly.hdf5
DS=ankile/dexmg-two-arm-three-piece-assembly
STAGES=outputs_chunk/three_piece_stages.npz
STATE30=outputs_chunk/three_piece_state30.npz
GCV=outputs_chunk/three_piece_gc_value_hiql3.pt
HA=outputs_chunk/three_piece_high_actor_hiql3.pt

echo "[driver] START $(date '+%F %T')  CUDA=$CUDA_VISIBLE_DEVICES"

# ===== 1) 重训 gc_value(纯默认: geom + LN + value_loss hiql + mask hiql + rep goal_only)=====
echo "[driver] 1/3 gc_value 重训(eef_piece replay,限线程~十几分钟)... $(date '+%T')"
$PY -u -m resfit.rl_finetuning.chunk_residual.train_hiql_gc_value \
  --hdf5 "$HDF5" --dataset "$DS" --stage_cache "$STAGES" --output "$GCV" \
  > outputs_chunk/three_piece_gc_value_hiql3.train.log 2>&1
RC=$?; echo "[driver] gc_value rc=$RC $(date '+%T')"
if [ $RC -ne 0 ] || [ ! -f "$GCV" ]; then echo "[driver] gc_value 失败,中止"; exit 1; fi

# ===== 2) 级联重训 high_actor(纯默认: clamp_to_goal + p_randomgoal0.3 + adv_agg mean,挂 hiql3 value)=====
echo "[driver] 2/3 high_actor 重训(state30 缓存,较快)... $(date '+%T')"
$PY -u -m resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
  --hdf5 "$HDF5" --dataset "$DS" --stage_cache "$STAGES" --state30_cache "$STATE30" \
  --gc_value_ckpt "$GCV" --output "$HA" \
  > outputs_chunk/three_piece_high_actor_hiql3.train.log 2>&1
RC=$?; echo "[driver] high_actor rc=$RC $(date '+%T')"
if [ $RC -ne 0 ] || [ ! -f "$HA" ]; then echo "[driver] high_actor 失败,中止"; exit 1; fi

# ===== 3) 起 residual run(照搬 bp_bc01 金标准,只换 hiql3 两件套 + 复用 offcache + 新 name/dir)=====
echo "[driver] 3/3 residual run(500k,卡2)... $(date '+%T')"
$PY -u -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreePieceAssembly \
  --base_wandb_id /mnt/mnt/data/resfit/resfit/out/piecce/best \
  --dataset "$DS" --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw --stage_balanced --reward_shaping none \
  --offline_dataset_path "$HDF5" --offline_fraction 0.5 --offline_stage_cache "$STAGES" \
  --offline_base_mode base_policy --demo_bc_coef 0.1 \
  --offline_buffer_cache outputs_chunk/aligned_bp_offcache \
  --subgoal_conditioned --gc_value_ckpt "$GCV" --high_actor_ckpt "$HA" \
  --subgoal_state30_cache "$STATE30" --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual --wandb_name three_piece_hiql3_bp_bc01 \
  --output_dir outputs_chunk/three_piece_hiql3_bp_bc01 \
  > outputs_chunk/three_piece_hiql3_bp_bc01.log 2>&1
echo "[driver] residual run 退出 rc=$? $(date '+%T')"
echo "[driver] ALL DONE $(date '+%F %T')"
