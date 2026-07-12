#!/usr/bin/env bash
# pouring 残差重跑:配置对齐 lifttray_actfeat_hdf5_bp_bc01_hiqlv512_sg15_bcdecay
#   —— value 换 HIQL-aligned(concat rep / hidden=512 / layers=3 / LN / hiql loss+mask /
#      geometric future),subgoal_way_steps 25->15,demo-BC 系数 0.1 线性衰减到 0
#      (--bc_coef_final 0.0,500k 步内归零)。
# 在此基础上,把 action_scale / actor_lr 调成参考项目 residual-offpolicy-rl 的值:
#   action_scale 0.05 -> 0.2  (参考 pouring v2/utd8 变体值;dexmg 默认/v3 是 0.1,本次按 0.2)
#   actor_lr     1e-6 -> 1e-6 (参考 residual_td3 dexmg 默认值,与原 resfit 一致,实为不变)
# pouring 特有、保持不动:hdf5 路 + no-stage(绝不加 --stage_balanced/--offline_stage_cache),
#   task=TwoArmPouring / 数据=two_arm_pouring.hdf5 / base=run_anw5pphu_best:v2。
# 产物全用新后缀 _hiqlv512(离线两件套)/_hiqlv512_sg15(offcache)/_..._as02(主训),
#   绝不覆盖原 pouring_*_hdf5 产物与 pouring_actfeat_hdf5_bp_bc01/best.pt(已跑完 0.94 结果)。
# 三步串行:gc_value(复用 act_feat 缓存) -> high_actor(命中缓存) -> 主训 500k(no-stage)。
# 用法: setsid bash run_pouring_hiqlv512_sg15_bcdecay_as02.sh <gpu> > pouring_hiqlv512_sg15_bcdecay_as02.log 2>&1 < /dev/null &
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

TASK=TwoArmPouring
DS=ankile/dexmg-two-arm-pouring
HDF5=resfit/dataset/two_arm_pouring.hdf5
BASE=/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2
CACHE=outputs_chunk/pouring_act_feat_hdf5.npz                                    # 复用现有特征缓存(686M)
GC=outputs_chunk/pouring_gc_value_actfeat_hdf5_hiqlv512.pt                       # 新
HA=outputs_chunk/pouring_high_actor_actfeat_hdf5_hiqlv512.pt                     # 新
OFFCACHE=outputs_chunk/pouring_actfeat_hdf5_bp_hiqlv512_sg15_offcache           # 新(会重建)
OUT=outputs_chunk/pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_bcdecay_as02        # 新
ACTION_SCALE=0.2
gate(){ [ -s "$1" ] || { echo "[driver] FATAL 产物缺失: $1 (上一步失败,停)"; exit 4; }; }

echo "[driver] START pouring ($TASK) hiqlv512 sg15 bcdecay action_scale=$ACTION_SCALE  GPU=$CUDA_VISIBLE_DEVICES  $(date '+%F %T')"
echo "[driver] GC=$GC"
echo "[driver] OUT=$OUT"

echo "[driver] 1) gc_value HIQL-aligned(concat/512/3/LN/hiql,复用 act_feat 缓存)... $(date '+%T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_hiql_gc_value \
  --hdf5 "$HDF5" --state_mode act_feat \
  --act_base_ckpt "$BASE" --dataset "$DS" \
  --act_feat_cache "$CACHE" --output "$GC" \
  --rep_dim 10 \
  --value_rep_mode concat \
  --value_hidden 512 \
  --value_layers 3 \
  --use_layer_norm 1 \
  --value_loss_mode hiql \
  --value_mask_mode hiql \
  --goal_future_mode geometric
gate "$CACHE"; gate "$GC"; echo "[driver] 1) gc_value DONE $(date '+%T')"

echo "[driver] 2) high_actor(命中缓存,way_steps 15)... $(date '+%T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
  --hdf5 "$HDF5" --state_mode act_feat \
  --act_base_ckpt "$BASE" --dataset "$DS" \
  --act_feat_cache "$CACHE" --gc_value_ckpt "$GC" --output "$HA" \
  --way_steps 15
gate "$HA"; echo "[driver] 2) high_actor DONE $(date '+%T')"

echo "[driver] 3) 主训 500k(hdf5,base_policy,no-stage,subgoal_way_steps 15,bc 0.1->0,action_scale $ACTION_SCALE)... $(date '+%T')"
$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task "$TASK" --base_wandb_id "$BASE" --dataset "$DS" \
  --offline_dataset_path "$HDF5" \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --action_scale "$ACTION_SCALE" --actor_lr 1e-6 --actor raw \
  --reward_shaping none \
  --offline_fraction 0.5 --offline_base_mode base_policy --demo_bc_coef 0.1 --bc_coef_final 0.0 \
  --offline_buffer_cache "$OFFCACHE" \
  --subgoal_conditioned --gc_value_ckpt "$GC" --high_actor_ckpt "$HA" --act_feat_cache "$CACHE" \
  --subgoal_way_steps 15 \
  --total_env_steps 500000 \
  --wandb_project dexmg-chunk-residual --wandb_name pouring_actfeat_hdf5_bp_bc01_hiqlv512_sg15_bcdecay_as02 \
  --output_dir "$OUT"
echo "[driver] ALL DONE pouring hiqlv512 sg15 bcdecay as02 $(date '+%F %T')  out=$OUT"
