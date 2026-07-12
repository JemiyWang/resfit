#!/usr/bin/env bash
# three_piece sg8 bc005:参数对齐 three_piece_actfeat_bp_bc01_hiqlv512_sg8_bcdecay,只改 BC:
#   demo_bc_coef 0.1 -> 0.05,且【不要 bcdecay】(去掉 --bc_coef_final,BC 固定 0.05)。
# subgoal_way_steps 仍是 8 → 与 sg8 完全相同,故 high_actor(sg8)与 offcache(sg8)可直接复用;
#   bc_coef 只是训练 loss 权重、不进 offcache 签名(已核实),所以【什么都不用重建/重训】。
# main-train-only:全部 gate 复用 gc_value/act_feat/stage/high_actor(sg8)/offcache(sg8)。
# 用法: setsid bash run_three_piece_hiqlv512_sg8_bc005.sh <gpu> > three_piece_hiqlv512_sg8_bc005.log 2>&1 < /dev/null &
set -u
GPU=${1:-1}
export PATH="/root/miniconda3/bin:$PATH"
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export CUDA_VISIBLE_DEVICES="$GPU"
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_OFFLINE=1
export PYTHONPATH=/mnt/mnt/data/resfit
cd /mnt/mnt/data/resfit || exit 3
RUN="conda run -n residual --no-capture-output python -u"

HDF5=resfit/dataset/two_arm_three_piece_assembly.hdf5
DS=ankile/dexmg-two-arm-three-piece-assembly
STAGES=outputs_chunk/three_piece_stages.npz                              # 复用
CACHE=outputs_chunk/three_piece_act_feat.npz                             # 复用
BASE=resfit/out/piecce/best
GCV=outputs_chunk/three_piece_gc_value_actfeat_hiqlv512.pt              # 复用
HIA=outputs_chunk/three_piece_high_actor_actfeat_hiqlv512_sg8.pt        # 复用(way_steps 8,与 sg8 同)
OFFCACHE=outputs_chunk/three_piece_actfeat_bp_hiqlv512_sg8_offcache     # 复用(bc 不进签名,直接命中)
OUT=outputs_chunk/three_piece_actfeat_bp_bc005_hiqlv512_sg8             # 新
WAY=8
gate(){ [ -s "$1" ] || { echo "[bc005] FATAL 复用产物缺失: $1"; exit 4; }; }

echo "[bc005] gate 复用产物(gc_value/act_feat/stage/high_actor/offcache)..."
gate "$CACHE"; gate "$STAGES"; gate "$GCV"; gate "$HIA"; gate "$OFFCACHE/buffer_meta.json"
echo "[bc005] START three_piece hiqlv512 sg${WAY} bc005(固定 0.05)  GPU=$CUDA_VISIBLE_DEVICES  $(date '+%F %T')"

# main-train-only(对齐 sg8 主训,仅 demo_bc_coef 0.05 + 去掉 bc_coef_final + 新 out/name)
$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreePieceAssembly --base_wandb_id "$BASE" \
  --dataset "$DS" \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw --stage_balanced --reward_shaping none \
  --offline_dataset_path "$HDF5" \
  --offline_fraction 0.5 --offline_stage_cache "$STAGES" \
  --offline_base_mode base_policy --demo_bc_coef 0.05 \
  --offline_buffer_cache "$OFFCACHE" \
  --subgoal_conditioned --gc_value_ckpt "$GCV" --high_actor_ckpt "$HIA" --act_feat_cache "$CACHE" \
  --subgoal_way_steps "$WAY" \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual --wandb_name three_piece_actfeat_bp_bc005_hiqlv512_sg8 \
  --output_dir "$OUT"
echo "[bc005] ALL DONE three_piece hiqlv512 sg${WAY} bc005 $(date '+%F %T')  out=$OUT"
