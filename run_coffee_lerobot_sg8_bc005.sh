#!/usr/bin/env bash
# coffee sg8 bc005:参数对齐 coffee_actfeat_lerobot_hiql_aligned_bp_bc01_sg8_bcdecay,只改 BC:
#   demo_bc_coef 0.1 -> 0.05,且【不要 bcdecay】(去掉 --bc_coef_final,BC 固定 0.05)。
# subgoal_way_steps 仍是 8 → high_actor(coffee sg8)与 offcache(coffee sg8)直接复用;
#   bc 只是 loss 权重、不进 offcache 签名(已核实)→【什么都不用重建/重训】,main-train-only。
# 用法: setsid bash run_coffee_lerobot_sg8_bc005.sh <gpu> > coffee_lerobot_sg8_bc005.log 2>&1 < /dev/null &
set -u
GPU=${1:-2}
export PATH="/root/miniconda3/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH=/mnt/mnt/data/resfit
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
cd /mnt/mnt/data/resfit || exit 3
source resfit/lerobot/shell/torchcodec_env.sh
RUN="conda run -n residual --no-capture-output python -u"

TASK=TwoArmCoffee
SUB=dexmg-two-arm-coffee
DS=ankile/$SUB
ROOT=/mnt/mnt/data/resfit/resfit/dataset/ankile/$SUB
BASE=/mnt/mnt/data/resfit/resfit/out/coffee/best
CACHE=outputs_chunk/coffee_act_feat_lerobot_hiql_aligned.npz                # 复用
GC=outputs_chunk/coffee_gc_value_actfeat_lerobot_hiql_aligned.pt            # 复用
HA=outputs_chunk/coffee_high_actor_actfeat_lerobot_hiql_aligned_sg8.pt      # 复用(way_steps 8,与 sg8 同)
OFFCACHE=outputs_chunk/coffee_actfeat_lerobot_hiql_aligned_bp_sg8_offcache  # 复用(bc 不进签名,直接命中)
OUT=outputs_chunk/coffee_actfeat_lerobot_hiql_aligned_bp_bc005_sg8          # 新
WAY=8
gate(){ [ -s "$1" ] || { echo "[coffee-bc005] FATAL 复用产物缺失: $1"; exit 4; }; }

echo "[coffee-bc005] gate 复用产物(gc_value/act_feat/high_actor/offcache)..."
gate "$CACHE"; gate "$GC"; gate "$HA"; gate "$OFFCACHE/buffer_meta.json"
echo "[coffee-bc005] START coffee lerobot sg${WAY} bc005(固定 0.05)  GPU=$CUDA_VISIBLE_DEVICES  $(date '+%F %T')"

# main-train-only(对齐 coffee sg8 主训,仅 demo_bc_coef 0.05 + 去掉 bc_coef_final + 新 out/name)
$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task "$TASK" --base_wandb_id "$BASE" --dataset "$DS" \
  --data_source lerobot --lerobot_root "$ROOT" \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw \
  --reward_shaping none \
  --offline_fraction 0.5 --offline_base_mode base_policy --demo_bc_coef 0.05 \
  --offline_buffer_cache "$OFFCACHE" \
  --subgoal_conditioned --gc_value_ckpt "$GC" --high_actor_ckpt "$HA" --act_feat_cache "$CACHE" \
  --subgoal_way_steps "$WAY" \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual --wandb_name coffee_actfeat_lerobot_hiql_aligned_bp_bc005_sg8 \
  --output_dir "$OUT"
echo "[coffee-bc005] ALL DONE coffee lerobot sg${WAY} bc005 $(date '+%F %T')  out=$OUT"
