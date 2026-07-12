#!/usr/bin/env bash
# three_piece sg8 bcdecay:以正在跑的 three_piece_actfeat_bp_bc01_hiqlv512_sg15 为基准,只改两处:
#   ① subgoal_way_steps 15 -> 8(high_actor --way_steps 与 主训 --subgoal_way_steps 两处同步改)
#   ② demo-BC 系数固定 0.1 -> 0.1 线性衰减到 0(新增 --bc_coef_final 0.0)
# 复用(与 way_steps 无关):gc_value(three_piece_gc_value_actfeat_hiqlv512.pt)/ act_feat / stage_cache。
# 重做:high_actor(way_steps 8,新后缀 _sg8)+ offcache(新后缀 _sg8,主训启动时自动重建)。
# 其余与 run_three_piece_hiqlv512_sg15.sh 逐字一致(含 hdf5 + --stage_balanced + base resfit/out/piecce/best)。
# 产物全用新后缀,绝不覆盖 sg15 产物/offcache/best.pt。
# 用法: setsid bash run_three_piece_hiqlv512_sg8_bcdecay.sh <gpu> > three_piece_hiqlv512_sg8_bcdecay.log 2>&1 < /dev/null &
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

HDF5=resfit/dataset/two_arm_three_piece_assembly.hdf5
DS=ankile/dexmg-two-arm-three-piece-assembly
STAGES=outputs_chunk/three_piece_stages.npz                              # 复用(与 way_steps 无关)
CACHE=outputs_chunk/three_piece_act_feat.npz                             # 复用(与 way_steps 无关)
BASE=resfit/out/piecce/best
GCV=outputs_chunk/three_piece_gc_value_actfeat_hiqlv512.pt              # 复用(与 way_steps 无关,存在即跳过)
HIA=outputs_chunk/three_piece_high_actor_actfeat_hiqlv512_sg8.pt        # 新(way_steps 8 重训)
OFFCACHE=outputs_chunk/three_piece_actfeat_bp_hiqlv512_sg8_offcache     # 新(会重建)
OUT=outputs_chunk/three_piece_actfeat_bp_bc01_hiqlv512_sg8_bcdecay      # 新
WAY=8
gate(){ [ -s "$1" ] || { echo "[driver] FATAL 产物缺失: $1 (上一步失败,停)"; exit 4; }; }

echo "[driver] START three_piece hiqlv512 sg${WAY} bcdecay  GPU=$CUDA_VISIBLE_DEVICES  $(date '+%F %T')"
echo "[driver] GCV=$GCV (复用)  HIA=$HIA (重训)  OUT=$OUT"

# 1) gc_value:复用已有(与 way_steps 无关)
gate "$CACHE"; gate "$STAGES"; gate "$GCV"
echo "[driver] 1) gc_value 复用现有(跳过重训): $GCV  $(date '+%T')"

# 2) high_actor:way_steps 8 重训 -> 新 ckpt
echo "[driver] 2) high_actor 重训(way_steps ${WAY})... $(date '+%T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
  --state_mode act_feat --act_base_ckpt "$BASE" \
  --hdf5 "$HDF5" --dataset "$DS" --stage_cache "$STAGES" \
  --act_feat_cache "$CACHE" --gc_value_ckpt "$GCV" --output "$HIA" \
  --way_steps "$WAY"
gate "$HIA"; echo "[driver] 2) high_actor DONE $(date '+%T')"

# 3) 主训 500k(与 sg15 step3 逐字一致,仅 way_steps 8 + 新增 bc_coef_final 0.0 + 新 offcache/out/name)
echo "[driver] 3) 主训 500k(base_policy,hdf5+stage_balanced,subgoal_way_steps ${WAY},bc 0.1->0)... $(date '+%T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task TwoArmThreePieceAssembly --base_wandb_id "$BASE" \
  --dataset "$DS" \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw --stage_balanced --reward_shaping none \
  --offline_dataset_path "$HDF5" \
  --offline_fraction 0.5 --offline_stage_cache "$STAGES" \
  --offline_base_mode base_policy --demo_bc_coef 0.1 --bc_coef_final 0.0 \
  --offline_buffer_cache "$OFFCACHE" \
  --subgoal_conditioned --gc_value_ckpt "$GCV" --high_actor_ckpt "$HIA" --act_feat_cache "$CACHE" \
  --subgoal_way_steps "$WAY" \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual --wandb_name three_piece_actfeat_bp_bc01_hiqlv512_sg8_bcdecay \
  --output_dir "$OUT"
echo "[driver] ALL DONE three_piece hiqlv512 sg${WAY} bcdecay $(date '+%F %T')  out=$OUT"
