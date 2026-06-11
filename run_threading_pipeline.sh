#!/usr/bin/env bash
# TwoArmThreading 残差 RL 主训(对齐 three_piece 稳定线:stage+object-aware+HIQL value potential)
# 用法: bash run_threading_pipeline.sh <gpu> [smoke]
#   <gpu>   : CUDA 逻辑卡号(空闲卡)
#   smoke   : 传 "smoke" 则少量 demo+步数+关 wandb 验接线;否则全量长跑
set -euo pipefail
GPU=${1:?usage: run_threading_pipeline.sh <gpu> [smoke]}
MODE=${2:-full}

# 关键约定(已核实):
#  - --base_wandb_id 传本地目录 resfit/out/threading/best(含 best/ 的 config.json+model.safetensors)
#  - 不传 --state_mode:env_state_mode 由 value.pt 的 state_mode(eef_piece)自动派生
#  - 金标准默认已生效:actor=raw / chunk_length=1 / actor_lr=1e-6 / stage_balanced=True /
#    offline_base_mode=base_policy / base_action_mode=queue / base_n_action_steps=None
BASE=resfit/out/threading/best
VALUE=outputs_chunk/two_arm_threading_value.pt
STAGES=outputs_chunk/two_arm_threading_stages.npz
HDF5=resfit/dataset/two_arm_threading.hdf5
DATASET=ankile/dexmg-two-arm-threading

COMMON=(
  --task TwoArmThreading
  --base_wandb_id "$BASE"
  --dataset "$DATASET"
  --offline_dataset_path "$HDF5"
  --offline_stage_cache "$STAGES"
  --reward_shaping potential --potential_source hiql
  --hiql_value_ckpt "$VALUE"
  --action_scale 0.05 --offline_fraction 0.5
  --demo_bc_coef 0.1
)

# OMP/MKL 限线程:防 CPU 侧(offline buffer 的 base forward / 小算子)开满核霸占机器
# (train_hiql_value 曾因此 352 线程霸 81 核;主训虽 GPU 为主,buffer build 仍有重 CPU 段)
export OMP_NUM_THREADS=16 MKL_NUM_THREADS=16 OPENBLAS_NUM_THREADS=16 NUMEXPR_NUM_THREADS=16
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl CUDA_VISIBLE_DEVICES="$GPU" HF_HUB_OFFLINE=1

if [ "$MODE" = "smoke" ]; then
  echo "[smoke] 少量 demo + 步数 + 关 wandb(只验接线)"
  conda run -n residual --no-capture-output python -u \
    -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
    "${COMMON[@]}" \
    --offline_num_demos 20 --smoke --wandb_mode disabled \
    --offline_buffer_cache outputs_chunk/threading_offcache_smoke \
    --output_dir outputs_chunk/threading_smoke \
    2>&1 | tee outputs_chunk/threading_smoke.log
else
  echo "[full] 全量长跑"
  conda run -n residual --no-capture-output python -u \
    -m resfit.rl_finetuning.chunk_residual.train_chunk_residual \
    "${COMMON[@]}" \
    --offline_buffer_cache outputs_chunk/threading_offcache \
    --wandb_project dexmg-chunk-residual --wandb_name threading_best_pothiql \
    --output_dir outputs_chunk/threading_best_pothiql \
    2>&1 | tee threading_best_pothiql.log
fi
