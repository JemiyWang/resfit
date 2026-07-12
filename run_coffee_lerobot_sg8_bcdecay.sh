#!/usr/bin/env bash
# coffee sg8 bcdecay:参数对齐 three_piece_actfeat_bp_bc01_hiqlv512_sg8_bcdecay。
#   在已跑完的 coffee(lerobot hiql_aligned,sg25,bc 固定 0.1)基础上只改两处:
#   ① subgoal_way_steps 25 -> 8(high_actor --way_steps 与 主训 --subgoal_way_steps 两处)
#   ② demo-BC 0.1 固定 -> 0.1 线性衰减到 0(新增 --bc_coef_final 0.0)
# 复用(与 way_steps 无关):gc_value(coffee_..._hiql_aligned.pt,=hiqlv512 配方)+ act_feat 缓存。
# 重做:high_actor(way_steps 8,新后缀 _sg8)+ offcache(新后缀 _sg8,主训启动自动建)。
# 其余照 coffee lerobot 路逐字:base=coffee/best、action_scale 0.05、no-stage、value 配方 hiqlv512。
# 用法: setsid bash run_coffee_lerobot_sg8_bcdecay.sh <gpu> > coffee_lerobot_sg8_bcdecay.log 2>&1 < /dev/null &
set -uo pipefail
GPU=${1:-6}
export PATH="/root/miniconda3/bin:$PATH"
export CUDA_VISIBLE_DEVICES="$GPU"
export PYTHONPATH=/mnt/mnt/data/resfit
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1 HF_ENDPOINT=https://hf-mirror.com
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 NUMEXPR_NUM_THREADS=8
cd /mnt/mnt/data/resfit || exit 3
source resfit/lerobot/shell/torchcodec_env.sh      # lerobot 解码走 torchcodec GPU 路(命门)
RUN="conda run -n residual --no-capture-output python -u"

TASK=TwoArmCoffee
SUB=dexmg-two-arm-coffee
DS=ankile/$SUB
ROOT=/mnt/mnt/data/resfit/resfit/dataset/ankile/$SUB
BASE=/mnt/mnt/data/resfit/resfit/out/coffee/best
DUMMY=/tmp/UNUSED_dummy.hdf5
CACHE=outputs_chunk/coffee_act_feat_lerobot_hiql_aligned.npz            # 复用
GC=outputs_chunk/coffee_gc_value_actfeat_lerobot_hiql_aligned.pt        # 复用(与 way_steps 无关)
HA=outputs_chunk/coffee_high_actor_actfeat_lerobot_hiql_aligned_sg8.pt  # 新(way_steps 8)
OFFCACHE=outputs_chunk/coffee_actfeat_lerobot_hiql_aligned_bp_sg8_offcache   # 新(会重建)
OUT=outputs_chunk/coffee_actfeat_lerobot_hiql_aligned_bp_bc01_sg8_bcdecay    # 新
WAY=8
gate(){ [ -s "$1" ] || { echo "[coffee-sg8] FATAL 复用产物缺失: $1"; exit 4; }; }

echo "[coffee-sg8] gate 复用产物(gc_value/act_feat)..."
gate "$CACHE"; gate "$GC"
echo "[coffee-sg8] START coffee ($TASK) lerobot sg${WAY} bcdecay  GPU=$CUDA_VISIBLE_DEVICES  $(date '+%F %T')"

echo "[coffee-sg8] high_actor 重训(way_steps ${WAY})... $(date '+%T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
  --hdf5 "$DUMMY" --state_mode act_feat --data_source lerobot --lerobot_root "$ROOT" \
  --act_base_ckpt "$BASE" --dataset "$DS" \
  --act_feat_cache "$CACHE" --gc_value_ckpt "$GC" --output "$HA" \
  --way_steps "$WAY"
gate "$HA"; echo "[coffee-sg8] high_actor DONE $(date '+%T')"

echo "[coffee-sg8] 主训 500k(对齐 three_piece sg8:lerobot no-stage,subgoal_way_steps ${WAY},bc 0.1->0)... $(date '+%T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task "$TASK" --base_wandb_id "$BASE" --dataset "$DS" \
  --data_source lerobot --lerobot_root "$ROOT" \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw \
  --reward_shaping none \
  --offline_fraction 0.5 --offline_base_mode base_policy --demo_bc_coef 0.1 --bc_coef_final 0.0 \
  --offline_buffer_cache "$OFFCACHE" \
  --subgoal_conditioned --gc_value_ckpt "$GC" --high_actor_ckpt "$HA" --act_feat_cache "$CACHE" \
  --subgoal_way_steps "$WAY" \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual --wandb_name coffee_actfeat_lerobot_hiql_aligned_bp_bc01_sg8_bcdecay \
  --output_dir "$OUT"
echo "[coffee-sg8] ALL DONE coffee lerobot sg${WAY} bcdecay $(date '+%F %T')  out=$OUT"
