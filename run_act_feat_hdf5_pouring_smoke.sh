#!/usr/bin/env bash
# pouring hdf5 路线端到端 smoke：验证全链能跑通(非训好模型)。act_feat cache 走全集(hdf5 省解码
# ~2min;主训 act_feat subgoal 要求 cache 覆盖全部 demo),gc/high 少步,主训 base_policy 40 步。
# 用法: CUDA_VISIBLE_DEVICES=N MUJOCO_EGL_DEVICE_ID=N bash run_act_feat_hdf5_pouring_smoke.sh
set -euo pipefail
export PYTHONPATH=/mnt/mnt/data/resfit
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
cd /mnt/mnt/data/resfit

TASK=TwoArmPouring; DS=ankile/dexmg-two-arm-pouring
HDF5=resfit/dataset/two_arm_pouring.hdf5
BASE=/mnt/mnt/data/wjm/residual/residual-offpolicy-rl/artifacts/run_anw5pphu_best:v2
CACHE=/tmp/pouring_hdf5_smoke_actfeat.npz
GC=/tmp/pouring_hdf5_smoke_gc.pt
HA=/tmp/pouring_hdf5_smoke_high.pt
RUN="conda run -n residual --no-capture-output python -u"

echo "=== [1] gc_value (hdf5 全集 cache 首建 ~2min + 少步) $(date '+%T') ==="
$RUN -m resfit.rl_finetuning.chunk_residual.train_hiql_gc_value \
  --hdf5 "$HDF5" --state_mode act_feat --act_base_ckpt "$BASE" --dataset "$DS" \
  --steps 80 --act_feat_cache "$CACHE" --output "$GC"
[ -s "$CACHE" ] && [ -s "$GC" ] || { echo "FATAL gc_value/cache"; exit 1; }

echo "=== [2] high_actor (命中 cache + 少步) $(date '+%T') ==="
$RUN -m resfit.rl_finetuning.chunk_residual.train_hiql_high_actor \
  --hdf5 "$HDF5" --state_mode act_feat --act_base_ckpt "$BASE" --dataset "$DS" \
  --steps 80 --act_feat_cache "$CACHE" --gc_value_ckpt "$GC" --output "$HA"
[ -s "$HA" ] || { echo "FATAL high_actor"; exit 1; }

echo "=== [3] 主训 smoke 40 步 (hdf5 base_policy, 对齐 three_piece 超参) $(date '+%T') ==="
$RUN -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
  --task "$TASK" --base_wandb_id "$BASE" --dataset "$DS" \
  --offline_dataset_path "$HDF5" \
  --subgoal_conditioned --reward_shaping none \
  --offline_base_mode base_policy \
  --gc_value_ckpt "$GC" --high_actor_ckpt "$HA" --act_feat_cache "$CACHE" \
  --offline_num_demos 4 --offline_fraction 0.5 --demo_bc_coef 0.1 \
  --action_scale 0.05 --actor_lr 1e-6 --actor raw \
  --chunk_length 1 --base_action_mode queue --base_n_action_steps 10 \
  --total_env_steps 40 --learning_starts 10 --eval_every_env_steps 30 \
  --eval_num_envs 2 --eval_num_episodes 2 --utd 1 \
  --wandb_mode disabled --output_dir /tmp/pouring_hdf5_smoke_out
echo "[smoke] pouring hdf5+base_policy 全链 DONE $(date '+%T')"
